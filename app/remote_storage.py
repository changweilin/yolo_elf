from __future__ import annotations

import asyncio
import base64
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from app.config import Settings
from app.recordings import RecordingRecord
from app.stream_state import CameraFrame


@dataclass(frozen=True)
class RemoteStorageRecord:
    frame_id: int
    received_at: float
    detection: dict[str, Any]
    jpeg: bytes | None


ClientFactory = Callable[[], httpx.AsyncClient]
RemoteStorageItem = RemoteStorageRecord | RecordingRecord


class RemoteStorage:
    def __init__(
        self, settings: Settings, client_factory: ClientFactory | None = None
    ) -> None:
        self.settings = settings
        self.events_enabled = bool(settings.remote_storage_url)
        self.recordings_enabled = bool(settings.remote_storage_recording_url)
        self.enabled = self.events_enabled or self.recordings_enabled
        self._client_factory = client_factory or self._default_client
        self._queue: asyncio.Queue[RemoteStorageItem] = asyncio.Queue(
            maxsize=settings.remote_storage_queue_size
        )
        self._worker: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self.records_enqueued = 0
        self.records_uploaded = 0
        self.records_failed = 0
        self.records_dropped = 0
        self.recordings_enqueued = 0
        self.recordings_uploaded = 0
        self.recordings_failed = 0
        self.recordings_dropped = 0
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
        if not self.events_enabled:
            return

        record = RemoteStorageRecord(
            frame_id=frame.frame_id,
            received_at=frame.received_at,
            detection=detection,
            jpeg=frame.jpeg if self.settings.remote_storage_include_frame else None,
        )

        await self._enqueue(record)

    async def submit_recording(self, recording: RecordingRecord) -> dict[str, Any]:
        if not self.recordings_enabled:
            return {"status": "disabled", "endpoint_configured": False}

        await self._enqueue(recording)
        return {"status": "queued", "endpoint_configured": True}

    async def _enqueue(self, item: RemoteStorageItem) -> None:
        if self._queue.full():
            try:
                dropped = self._queue.get_nowait()
                self._queue.task_done()
                async with self._lock:
                    if isinstance(dropped, RecordingRecord):
                        self.recordings_dropped += 1
                    else:
                        self.records_dropped += 1
            except asyncio.QueueEmpty:
                pass

        self._queue.put_nowait(item)
        async with self._lock:
            if isinstance(item, RecordingRecord):
                self.recordings_enqueued += 1
            else:
                self.records_enqueued += 1

    async def drain(self) -> None:
        if self.enabled:
            await self._queue.join()

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "enabled": self.enabled,
                "endpoint_configured": bool(self.settings.remote_storage_url),
                "recording_endpoint_configured": bool(
                    self.settings.remote_storage_recording_url
                ),
                "include_frame": self.settings.remote_storage_include_frame,
                "queue_depth": self._queue.qsize(),
                "queue_size": self.settings.remote_storage_queue_size,
                "records_enqueued": self.records_enqueued,
                "records_uploaded": self.records_uploaded,
                "records_failed": self.records_failed,
                "records_dropped": self.records_dropped,
                "recordings_enqueued": self.recordings_enqueued,
                "recordings_uploaded": self.recordings_uploaded,
                "recordings_failed": self.recordings_failed,
                "recordings_dropped": self.recordings_dropped,
                "last_error": self.last_error,
                "last_attempt_at": self.last_attempt_at,
                "last_uploaded_at": self.last_uploaded_at,
            }

    async def _run(self) -> None:
        async with self._client_factory() as client:
            while True:
                record = await self._queue.get()
                try:
                    if isinstance(record, RecordingRecord):
                        await self._upload_recording_with_retries(client, record)
                    else:
                        await self._upload_with_retries(client, record)
                except Exception as exc:
                    async with self._lock:
                        if isinstance(record, RecordingRecord):
                            self.recordings_failed += 1
                        else:
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

    async def _upload_recording_with_retries(
        self, client: httpx.AsyncClient, record: RecordingRecord
    ) -> None:
        attempts = self.settings.remote_storage_retries + 1
        last_error: str | None = None
        for attempt in range(attempts):
            async with self._lock:
                self.last_attempt_at = _utc_timestamp()
            try:
                with ExitStack() as stack:
                    recording_file = stack.enter_context(record.path.open("rb"))
                    files = {
                        "file": (
                            record.filename,
                            recording_file,
                            record.content_type,
                        )
                    }
                    if record.metadata_path is not None and record.metadata_path.exists():
                        metadata_file = stack.enter_context(record.metadata_path.open("rb"))
                        files["metadata"] = (
                            record.metadata_path.name,
                            metadata_file,
                            "application/json",
                        )
                    response = await client.post(
                        self.settings.remote_storage_recording_url,
                        data=self._recording_fields(record),
                        files=files,
                        headers=self._headers(),
                    )
                response.raise_for_status()
            except Exception as exc:
                last_error = str(exc)
                if attempt < attempts - 1:
                    await asyncio.sleep(min(2.0, 0.2 * (2**attempt)))
                continue

            async with self._lock:
                self.recordings_uploaded += 1
                self.last_error = None
                self.last_uploaded_at = _utc_timestamp()
            if not record.local_saved:
                record.path.unlink(missing_ok=True)
                if record.metadata_path is not None:
                    record.metadata_path.unlink(missing_ok=True)
            return

        raise RuntimeError(last_error or "Remote recording upload failed")

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

    def _recording_fields(self, record: RecordingRecord) -> dict[str, str]:
        fields = {
            "source": "yolo-elf",
            "type": "recording",
            "recording_id": record.recording_id,
            "filename": record.filename,
            "content_type": record.content_type,
            "byte_length": str(record.byte_length),
            "storage_mode": record.storage_mode,
            "local_saved": "1" if record.local_saved else "0",
            "metadata_byte_length": str(_file_size(record.metadata_path)),
            "created_at": str(record.created_at),
            "created_at_iso": datetime.fromtimestamp(
                record.created_at, timezone.utc
            ).isoformat(),
        }
        if record.duration_ms is not None:
            fields["duration_ms"] = str(record.duration_ms)
        if record.started_at:
            fields["started_at"] = record.started_at
        return fields

    def _headers(self) -> dict[str, str]:
        if not self.settings.remote_storage_token:
            return {}
        return {"Authorization": f"Bearer {self.settings.remote_storage_token}"}

    def _default_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.settings.remote_storage_timeout)


def _utc_timestamp() -> float:
    return datetime.now(timezone.utc).timestamp()


def _file_size(path) -> int:
    if path is None:
        return 0
    try:
        return path.stat().st_size
    except OSError:
        return 0
