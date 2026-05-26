from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from app.config import Settings
from app.stream_state import CameraFrame


@dataclass(frozen=True)
class RemoteStorageRecord:
    frame_id: int
    received_at: float
    detection: dict[str, Any]
    jpeg: bytes | None


ClientFactory = Callable[[], httpx.AsyncClient]


class RemoteStorage:
    def __init__(
        self, settings: Settings, client_factory: ClientFactory | None = None
    ) -> None:
        self.settings = settings
        self.enabled = bool(settings.remote_storage_url)
        self._client_factory = client_factory or self._default_client
        self._queue: asyncio.Queue[RemoteStorageRecord] = asyncio.Queue(
            maxsize=settings.remote_storage_queue_size
        )
        self._worker: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self.records_enqueued = 0
        self.records_uploaded = 0
        self.records_failed = 0
        self.records_dropped = 0
        self.last_error: str | None = None
        self.last_attempt_at: float | None = None
        self.last_uploaded_at: float | None = None

    async def start(self) -> None:
        if not self.enabled or self._worker is not None:
            return
        self._worker = asyncio.create_task(self._run(), name="remote-storage")

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

    async def submit(self, frame: CameraFrame, detection: dict[str, Any]) -> None:
        if not self.enabled:
            return

        record = RemoteStorageRecord(
            frame_id=frame.frame_id,
            received_at=frame.received_at,
            detection=detection,
            jpeg=frame.jpeg if self.settings.remote_storage_include_frame else None,
        )

        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                async with self._lock:
                    self.records_dropped += 1
            except asyncio.QueueEmpty:
                pass

        self._queue.put_nowait(record)
        async with self._lock:
            self.records_enqueued += 1

    async def drain(self) -> None:
        if self.enabled:
            await self._queue.join()

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "enabled": self.enabled,
                "endpoint_configured": bool(self.settings.remote_storage_url),
                "include_frame": self.settings.remote_storage_include_frame,
                "queue_depth": self._queue.qsize(),
                "queue_size": self.settings.remote_storage_queue_size,
                "records_enqueued": self.records_enqueued,
                "records_uploaded": self.records_uploaded,
                "records_failed": self.records_failed,
                "records_dropped": self.records_dropped,
                "last_error": self.last_error,
                "last_attempt_at": self.last_attempt_at,
                "last_uploaded_at": self.last_uploaded_at,
            }

    async def _run(self) -> None:
        async with self._client_factory() as client:
            while True:
                record = await self._queue.get()
                try:
                    await self._upload_with_retries(client, record)
                except Exception as exc:
                    async with self._lock:
                        self.records_failed += 1
                        self.last_error = str(exc)
                finally:
                    self._queue.task_done()

    async def _upload_with_retries(
        self, client: httpx.AsyncClient, record: RemoteStorageRecord
    ) -> None:
        attempts = self.settings.remote_storage_retries + 1
        last_error: str | None = None
        for attempt in range(attempts):
            async with self._lock:
                self.last_attempt_at = _utc_timestamp()
            try:
                response = await client.post(
                    self.settings.remote_storage_url,
                    json=self._payload(record),
                    headers=self._headers(),
                )
                response.raise_for_status()
            except Exception as exc:
                last_error = str(exc)
                if attempt < attempts - 1:
                    await asyncio.sleep(min(2.0, 0.2 * (2**attempt)))
                continue

            async with self._lock:
                self.records_uploaded += 1
                self.last_error = None
                self.last_uploaded_at = _utc_timestamp()
            return

        raise RuntimeError(last_error or "Remote storage upload failed")

    def _payload(self, record: RemoteStorageRecord) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": "yolo-elf",
            "frame_id": record.detection.get("frame_id", record.frame_id),
            "received_at": record.received_at,
            "received_at_iso": datetime.fromtimestamp(
                record.received_at, timezone.utc
            ).isoformat(),
            "detection": record.detection,
        }
        if record.jpeg is not None:
            payload["frame"] = {
                "content_type": "image/jpeg",
                "byte_length": len(record.jpeg),
                "base64": base64.b64encode(record.jpeg).decode("ascii"),
            }
        return payload

    def _headers(self) -> dict[str, str]:
        if not self.settings.remote_storage_token:
            return {}
        return {"Authorization": f"Bearer {self.settings.remote_storage_token}"}

    def _default_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.settings.remote_storage_timeout)


def _utc_timestamp() -> float:
    return datetime.now(timezone.utc).timestamp()
