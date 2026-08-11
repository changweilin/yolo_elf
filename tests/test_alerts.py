import asyncio
import json

import httpx
import pytest

from app.alerts import AlertEngine, parse_alert_rules
from app.stream_state import CameraFrame
from helpers import box as _box
from helpers import detection as _detection
from helpers import settings_from


def _engine(monkeypatch, rules=None, **env):
    if rules is not None:
        env["ALERT_RULES"] = json.dumps(rules)
    return AlertEngine(settings_from(monkeypatch, **env))


def _frame():
    return CameraFrame(frame_id=1, jpeg=b"jpeg", received_at=1710000000.0)


# --- rule parsing / validation -------------------------------------------------


def test_parse_alert_rules_empty_is_disabled():
    assert parse_alert_rules("", 15.0) == ()
    assert parse_alert_rules("   ", 15.0) == ()


def test_parse_alert_rules_reads_fields():
    rules = parse_alert_rules(
        json.dumps(
            [{"name": "crowd", "classes": ["person"], "min_count": 3, "cooldown_sec": 30}]
        ),
        15.0,
    )
    assert len(rules) == 1
    assert rules[0].name == "crowd"
    assert rules[0].classes == frozenset({"person"})
    assert rules[0].min_count == 3
    assert rules[0].cooldown_sec == 30.0


def test_rule_cooldown_defaults_to_engine_default():
    rules = parse_alert_rules(json.dumps([{"name": "any"}]), 42.0)
    assert rules[0].cooldown_sec == 42.0
    assert rules[0].classes == frozenset()
    assert rules[0].min_count == 1


@pytest.mark.parametrize(
    "raw",
    [
        "{not json",
        json.dumps({"name": "x"}),  # object, not an array
        json.dumps([{"classes": ["person"]}]),  # missing name
        json.dumps([{"name": "x", "min_count": 0}]),
        json.dumps([{"name": "x", "min_confidence": 1.5}]),
        json.dumps([{"name": "x", "cooldown_sec": -1}]),
        json.dumps([{"name": "dup"}, {"name": "dup"}]),
    ],
)
def test_parse_alert_rules_rejects_invalid(raw):
    with pytest.raises(ValueError):
        parse_alert_rules(raw, 15.0)


# --- evaluation ---------------------------------------------------------------


def test_process_fires_when_count_reached(monkeypatch):
    engine = _engine(
        monkeypatch, [{"name": "two-people", "classes": ["person"], "min_count": 2, "cooldown_sec": 0}]
    )
    boxes = [_box("person", track_id=4), _box("person", track_id=7), _box("dog")]

    events = asyncio.run(engine.process(_frame(), _detection(boxes)))

    assert len(events) == 1
    event = events[0]
    assert event["type"] == "alert"
    assert event["rule"] == "two-people"
    assert event["count"] == 2
    assert event["track_ids"] == [4, 7]
    assert {m["label"] for m in event["matches"]} == {"person"}


def test_process_does_not_fire_below_count(monkeypatch):
    engine = _engine(
        monkeypatch, [{"name": "two-people", "classes": ["person"], "min_count": 2}]
    )
    events = asyncio.run(engine.process(_frame(), _detection([_box("person")])))
    assert events == []


def test_process_filters_by_min_confidence(monkeypatch):
    engine = _engine(
        monkeypatch,
        [{"name": "sure-person", "classes": ["person"], "min_confidence": 0.5, "cooldown_sec": 0}],
    )
    boxes = [_box("person", confidence=0.4), _box("person", confidence=0.92)]

    events = asyncio.run(engine.process(_frame(), _detection(boxes)))

    assert events[0]["count"] == 1


def test_empty_classes_matches_any_label(monkeypatch):
    engine = _engine(monkeypatch, [{"name": "anything", "classes": [], "cooldown_sec": 0}])
    events = asyncio.run(engine.process(_frame(), _detection([_box("giraffe")])))
    assert events[0]["rule"] == "anything"


def test_process_scopes_rule_to_zone(monkeypatch):
    engine = _engine(
        monkeypatch,
        [{"name": "door-person", "classes": ["person"], "zone": "door", "cooldown_sec": 0}],
    )
    boxes = [
        {"label": "person", "confidence": 0.9, "track_id": 1, "zones": ["door"]},
        {"label": "person", "confidence": 0.9, "track_id": 2, "zones": ["lobby"]},
        {"label": "person", "confidence": 0.9, "track_id": 3, "zones": []},
    ]

    events = asyncio.run(engine.process(_frame(), _detection(boxes)))

    assert len(events) == 1
    assert events[0]["count"] == 1
    assert events[0]["zone"] == "door"
    assert events[0]["track_ids"] == [1]


def test_error_detection_never_fires(monkeypatch):
    engine = _engine(monkeypatch, [{"name": "anything", "cooldown_sec": 0}])
    events = asyncio.run(
        engine.process(_frame(), _detection([_box("person")], error="bad frame"))
    )
    assert events == []


def test_cooldown_blocks_then_expires(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr("app.alerts.time.monotonic", lambda: clock["t"])
    engine = _engine(monkeypatch, [{"name": "person", "classes": ["person"], "cooldown_sec": 30}])

    async def run():
        first = await engine.process(_frame(), _detection([_box("person")]))
        clock["t"] = 1010.0  # within cooldown
        second = await engine.process(_frame(), _detection([_box("person")]))
        clock["t"] = 1041.0  # past the 30s cooldown
        third = await engine.process(_frame(), _detection([_box("person")]))
        return first, second, third

    first, second, third = asyncio.run(run())
    assert len(first) == 1
    assert second == []
    assert len(third) == 1
    assert engine.events_fired == 2


def test_set_rules_swaps_at_runtime(monkeypatch):
    engine = _engine(monkeypatch, None)
    assert engine.enabled is False

    async def run():
        await engine.set_rules([{"name": "person", "classes": ["person"], "cooldown_sec": 0}])
        fired = await engine.process(_frame(), _detection([_box("person")]))
        snap = await engine.snapshot()
        return fired, snap

    fired, snap = asyncio.run(run())
    assert engine.enabled is True
    assert len(fired) == 1
    assert snap["enabled"] is True
    assert snap["rules"][0]["name"] == "person"
    assert snap["events_fired"] == 1
    assert snap["recent_events"][0]["rule"] == "person"


def test_set_rules_rejects_invalid(monkeypatch):
    engine = _engine(monkeypatch, None)
    with pytest.raises(ValueError):
        asyncio.run(engine.set_rules([{"min_count": 2}]))


# --- webhook dispatch ---------------------------------------------------------


def test_webhook_posts_fired_alert(monkeypatch):
    settings = settings_from(
        monkeypatch,
        ALERT_RULES=json.dumps([{"name": "person", "classes": ["person"], "cooldown_sec": 0}]),
        ALERT_WEBHOOK_URL="https://hooks.example/alert",
        ALERT_WEBHOOK_TOKEN="secret",
    )
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(204)

    def client_factory():
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler), timeout=settings.alert_webhook_timeout
        )

    async def run():
        engine = AlertEngine(settings, client_factory=client_factory)
        await engine.start()
        try:
            await engine.process(_frame(), _detection([_box("person", track_id=9)]))
            await asyncio.wait_for(engine.drain(), timeout=1)
            return await engine.snapshot()
        finally:
            await engine.stop()

    snap = asyncio.run(run())

    assert snap["webhook_enabled"] is True
    assert snap["webhook_sent"] == 1
    assert snap["webhook_failed"] == 0
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert payload["source"] == "yolo-elf"
    assert payload["rule"] == "person"
    assert payload["track_ids"] == [9]
