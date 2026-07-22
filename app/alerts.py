from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from app.config import Settings
from app.stream_state import CameraFrame

# Keep the status snapshot and per-event detail bounded so a busy stream can't
# grow the payload without limit.
MAX_RECENT_EVENTS = 20
MAX_MATCH_DETAIL = 20

ClientFactory = Callable[[], httpx.AsyncClient]


@dataclass(frozen=True)
class AlertRule:
    """A single rule evaluated against every detection.

    ``classes`` empty means "any label"; otherwise a box counts only when its
    detection label is in the set. ``min_confidence`` filters weak boxes before
    counting, ``min_count`` is how many matching boxes trigger the rule, and
    ``cooldown_sec`` is the minimum gap between two fires of the same rule.
    """

    name: str
    classes: frozenset[str]
    min_count: int
    min_confidence: float
    cooldown_sec: float

    def matches_label(self, label: str) -> bool:
        return not self.classes or label in self.classes

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "classes": sorted(self.classes),
            "min_count": self.min_count,
            "min_confidence": self.min_confidence,
            "cooldown_sec": self.cooldown_sec,
        }


def _parse_rule(item: Any, index: int, default_cooldown: float) -> AlertRule:
    if not isinstance(item, dict):
        raise ValueError(f"alert rule #{index} must be an object")

    name = str(item.get("name") or "").strip()
    if not name:
        raise ValueError(f"alert rule #{index} requires a non-empty name")

    classes_raw = item.get("classes")
    if classes_raw is None:
        classes: frozenset[str] = frozenset()
    elif isinstance(classes_raw, str):
        classes = frozenset(part.strip() for part in classes_raw.split(",") if part.strip())
    elif isinstance(classes_raw, (list, tuple)):
        classes = frozenset(str(part).strip() for part in classes_raw if str(part).strip())
    else:
        raise ValueError(f"alert rule {name!r}: classes must be a list or comma string")

    min_count = _positive_int(item.get("min_count", 1), name, "min_count")
    min_confidence = _unit_float(item.get("min_confidence", 0.0), name, "min_confidence")
    cooldown_sec = _non_negative_float(
        item.get("cooldown_sec", default_cooldown), name, "cooldown_sec"
    )
    return AlertRule(
        name=name,
        classes=classes,
        min_count=min_count,
        min_confidence=min_confidence,
        cooldown_sec=cooldown_sec,
    )


def _positive_int(value: Any, rule: str, field: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"alert rule {rule!r}: {field} must be an integer") from exc
    if number < 1:
        raise ValueError(f"alert rule {rule!r}: {field} must be >= 1")
    return number


def _unit_float(value: Any, rule: str, field: str) -> float:
    number = _to_float(value, rule, field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"alert rule {rule!r}: {field} must be between 0.0 and 1.0")
    return number


def _non_negative_float(value: Any, rule: str, field: str) -> float:
    number = _to_float(value, rule, field)
    if number < 0.0:
        raise ValueError(f"alert rule {rule!r}: {field} must be >= 0")
    return number


def _to_float(value: Any, rule: str, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"alert rule {rule!r}: {field} must be a number") from exc


def build_rules(data: Any, default_cooldown: float) -> tuple[AlertRule, ...]:
    """Validate a decoded rules list into ``AlertRule`` objects (fail loud)."""
    if not isinstance(data, list):
        raise ValueError("alert rules must be a JSON array of rule objects")
    rules = tuple(_parse_rule(item, index, default_cooldown) for index, item in enumerate(data))
    names = [rule.name for rule in rules]
    if len(names) != len(set(names)):
        raise ValueError("alert rule names must be unique")
    return rules


def parse_alert_rules(raw: str, default_cooldown: float) -> tuple[AlertRule, ...]:
    raw = (raw or "").strip()
    if not raw:
        return ()
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"ALERT_RULES must be valid JSON: {exc}") from exc
    return build_rules(data, default_cooldown)


class AlertEngine:
    """Evaluates detections against rules and dispatches fired alerts.

    Rules come from ``ALERT_RULES`` at startup and can be replaced at runtime via
    ``set_rules``. A fired alert is returned to the caller (to broadcast to
    viewers) and, when a webhook is configured, queued for background delivery.
    The single detection worker calls ``process`` serially; the lock guards the
    shared state against concurrent snapshot reads and runtime rule swaps.
    """

    def __init__(self, settings: Settings, client_factory: ClientFactory | None = None) -> None:
        self.settings = settings
        self._rules = parse_alert_rules(settings.alert_rules_json, settings.alert_cooldown_sec)
        self.webhook_enabled = bool(settings.alert_webhook_url)
        self._client_factory = client_factory or self._default_client
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._worker: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        # Cooldown bookkeeping keyed by rule name, using a monotonic clock so it
        # is immune to wall-clock jumps.
        self._last_fired_monotonic: dict[str, float] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=MAX_RECENT_EVENTS)
        self.events_fired = 0
        self.webhook_sent = 0
        self.webhook_failed = 0
        self.webhook_dropped = 0
        self.last_webhook_error: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._rules)

    async def start(self) -> None:
        if self.webhook_enabled and self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="alert-webhook")

    async def stop(self) -> None:
        worker = self._worker
        if worker is None:
            return
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        self._worker = None

    async def process(
        self, frame: CameraFrame, detection: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Evaluate one detection; return the alerts that fired this frame."""
        if not self._rules or detection.get("error"):
            return []

        boxes = detection.get("boxes") or []
        now = time.monotonic()
        fired: list[dict[str, Any]] = []
        async with self._lock:
            for rule in self._rules:
                matches = [
                    box
                    for box in boxes
                    if rule.matches_label(str(box.get("label", "")))
                    and float(box.get("confidence") or 0.0) >= rule.min_confidence
                ]
                if len(matches) < rule.min_count:
                    continue
                last = self._last_fired_monotonic.get(rule.name)
                if last is not None and (now - last) < rule.cooldown_sec:
                    continue
                self._last_fired_monotonic[rule.name] = now
                event = self._build_event(rule, matches, detection)
                self._recent.appendleft(event)
                self.events_fired += 1
                fired.append(event)

        for event in fired:
            await self._enqueue_webhook(event)
        return fired

    async def set_rules(self, data: Any) -> None:
        rules = build_rules(data, self.settings.alert_cooldown_sec)
        names = {rule.name for rule in rules}
        async with self._lock:
            self._rules = rules
            # Drop cooldown state for rules that no longer exist so a removed and
            # re-added rule doesn't inherit a stale cooldown.
            self._last_fired_monotonic = {
                name: ts for name, ts in self._last_fired_monotonic.items() if name in names
            }

    async def drain(self) -> None:
        if self.webhook_enabled:
            await self._queue.join()

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "enabled": self.enabled,
                "webhook_enabled": self.webhook_enabled,
                "webhook_configured": bool(self.settings.alert_webhook_url),
                "rules": [rule.public() for rule in self._rules],
                "events_fired": self.events_fired,
                "recent_events": list(self._recent),
                "webhook_sent": self.webhook_sent,
                "webhook_failed": self.webhook_failed,
                "webhook_dropped": self.webhook_dropped,
                "queue_depth": self._queue.qsize(),
                "last_webhook_error": self.last_webhook_error,
            }

    def _build_event(
        self, rule: AlertRule, matches: list[dict[str, Any]], detection: dict[str, Any]
    ) -> dict[str, Any]:
        fired_at = time.time()
        detail = [
            {
                "label": str(box.get("label", "")),
                "confidence": box.get("confidence"),
                "track_id": box.get("track_id"),
            }
            for box in matches[:MAX_MATCH_DETAIL]
        ]
        track_ids = [
            box.get("track_id") for box in matches if box.get("track_id") is not None
        ]
        return {
            "type": "alert",
            "rule": rule.name,
            "count": len(matches),
            "frame_id": detection.get("frame_id"),
            "fired_at": fired_at,
            "fired_at_iso": datetime.fromtimestamp(fired_at, timezone.utc).isoformat(),
            "matches": detail,
            "track_ids": track_ids,
        }

    async def _enqueue_webhook(self, event: dict[str, Any]) -> None:
        if not self.webhook_enabled:
            return
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                async with self._lock:
                    self.webhook_dropped += 1
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(event)

    async def _run(self) -> None:
        async with self._client_factory() as client:
            while True:
                event = await self._queue.get()
                try:
                    await self._post_with_retries(client, event)
                except Exception as exc:
                    async with self._lock:
                        self.webhook_failed += 1
                        self.last_webhook_error = str(exc)
                finally:
                    self._queue.task_done()

    async def _post_with_retries(self, client: httpx.AsyncClient, event: dict[str, Any]) -> None:
        attempts = self.settings.alert_webhook_retries + 1
        last_error: str | None = None
        payload = {"source": "yolo-elf", **event}
        for attempt in range(attempts):
            try:
                response = await client.post(
                    self.settings.alert_webhook_url, json=payload, headers=self._headers()
                )
                response.raise_for_status()
            except Exception as exc:
                last_error = str(exc)
                if attempt < attempts - 1:
                    await asyncio.sleep(min(2.0, 0.2 * (2**attempt)))
                continue
            async with self._lock:
                self.webhook_sent += 1
                self.last_webhook_error = None
            return
        raise RuntimeError(last_error or "Alert webhook delivery failed")

    def _headers(self) -> dict[str, str]:
        if not self.settings.alert_webhook_token:
            return {}
        return {"Authorization": f"Bearer {self.settings.alert_webhook_token}"}

    def _default_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.settings.alert_webhook_timeout)
