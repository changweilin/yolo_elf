import asyncio
import base64
import json

import httpx

from app.config import get_settings
from app.recordings import RecordingRecord
from app.remote_storage import RemoteStorage
from app.stream_state import CameraFrame


REMOTE_ENV = [
    "RECORDING_ENABLED",
    "RECORDING_STORAGE_DIR",
    "RECORDING_MAX_BYTES",
    "REMOTE_STORAGE_URL",
    "REMOTE_STORAGE_TOKEN",
    "REMOTE_STORAGE_INCLUDE_FRAME",
    "REMOTE_STORAGE_RECORDING_URL",
    "REMOTE_STORAGE_QUEUE_SIZE",
    "REMOTE_STORAGE_TIMEOUT",
    "REMOTE_STORAGE_RETRIES",
]


def clear_remote_env(monkeypatch):
    for name in REMOTE_ENV:
        monkeypatch.delenv(name, raising=False)


def test_remote_storage_is_disabled_without_endpoint(monkeypatch):
    clear_remote_env(monkeypatch)
    storage = RemoteStorage(get_settings())

    async def run():
        await storage.start()
        await storage.submit(
            CameraFrame(frame_id=1, jpeg=b"jpeg", received_at=1710000000.0),
            {"frame_id": 1, "boxes": []},
        )
        return await storage.snapshot()

    status = asyncio.run(run())

    assert status["enabled"] is False
    assert status["records_enqueued"] == 0
    assert status["queue_depth"] == 0


def test_remote_storage_posts_detection_payload(monkeypatch):
    clear_remote_env(monkeypatch)
    monkeypatch.setenv("REMOTE_STORAGE_URL", "https://storage.example/events")
    monkeypatch.setenv("REMOTE_STORAGE_TOKEN", "secret-token")
    monkeypatch.setenv("REMOTE_STORAGE_INCLUDE_FRAME", "1")
    settings = get_settings()
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(204)

    def client_factory():
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            timeout=settings.remote_storage_timeout,
        )

    async def run():
        storage = RemoteStorage(settings, client_factory=client_factory)
        await storage.start()
        try:
            await storage.submit(
                CameraFrame(frame_id=3, jpeg=b"jpeg-bytes", received_at=1710000000.0),
                {"frame_id": 3, "boxes": [{"label": "person"}]},
            )
            await asyncio.wait_for(storage.drain(), timeout=1)
            return await storage.snapshot()
        finally:
            await storage.stop()

    status = asyncio.run(run())

    assert status["records_uploaded"] == 1
    assert status["records_failed"] == 0
    assert len(requests) == 1
    request = requests[0]
    payload = json.loads(request.content)
    assert str(request.url) == "https://storage.example/events"
    assert request.headers["authorization"] == "Bearer secret-token"
    assert payload["source"] == "yolo-elf"
    assert payload["frame_id"] == 3
    assert payload["detection"]["boxes"] == [{"label": "person"}]
    assert payload["frame"]["content_type"] == "image/jpeg"
    assert payload["frame"]["byte_length"] == len(b"jpeg-bytes")
    assert payload["frame"]["base64"] == base64.b64encode(b"jpeg-bytes").decode("ascii")


def test_remote_storage_posts_recording_payload(monkeypatch, tmp_path):
    clear_remote_env(monkeypatch)
    monkeypatch.setenv("REMOTE_STORAGE_RECORDING_URL", "https://storage.example/recordings")
    monkeypatch.setenv("REMOTE_STORAGE_TOKEN", "secret-token")
    settings = get_settings()
    recording_path = tmp_path / "rec-test.webm"
    recording_path.write_bytes(b"webm-bytes")
    metadata_path = tmp_path / "rec-test.detections.json"
    metadata_path.write_text('{"detections":[{"boxes":[{"xywh":[1,2,3,4]}]}]}', encoding="utf-8")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(201)

    def client_factory():
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            timeout=settings.remote_storage_timeout,
        )

    async def run():
        storage = RemoteStorage(settings, client_factory=client_factory)
        await storage.start()
        try:
            result = await storage.submit_recording(
                RecordingRecord(
                    recording_id="rec-test",
                    filename="rec-test.webm",
                    path=recording_path,
                    content_type="video/webm",
                    byte_length=len(b"webm-bytes"),
                    duration_ms=1200,
                    started_at="2026-05-27T08:00:00.000Z",
                    created_at=1710000000.0,
                    storage_mode="both",
                    local_saved=True,
                    metadata_path=metadata_path,
                )
            )
            await asyncio.wait_for(storage.drain(), timeout=1)
            return result, await storage.snapshot()
        finally:
            await storage.stop()

    result, status = asyncio.run(run())

    assert result["status"] == "queued"
    assert status["enabled"] is True
    assert status["endpoint_configured"] is False
    assert status["recording_endpoint_configured"] is True
    assert status["recordings_uploaded"] == 1
    assert status["recordings_failed"] == 0
    assert len(requests) == 1
    request = requests[0]
    body = request.content
    assert str(request.url) == "https://storage.example/recordings"
    assert request.headers["authorization"] == "Bearer secret-token"
    assert "multipart/form-data" in request.headers["content-type"]
    assert b'name="recording_id"\r\n\r\nrec-test' in body
    assert b'name="duration_ms"\r\n\r\n1200' in body
    assert b'name="storage_mode"\r\n\r\nboth' in body
    assert b'name="local_saved"\r\n\r\n1' in body
    assert b'name="metadata_byte_length"\r\n\r\n' in body
    assert b'filename="rec-test.webm"' in body
    assert b'filename="rec-test.detections.json"' in body
    assert b'"xywh":[1,2,3,4]' in body
    assert b"webm-bytes" in body


def test_remote_only_recording_staging_file_is_removed_after_upload(monkeypatch, tmp_path):
    clear_remote_env(monkeypatch)
    monkeypatch.setenv("REMOTE_STORAGE_RECORDING_URL", "https://storage.example/recordings")
    settings = get_settings()
    recording_path = tmp_path / "rec-test.webm"
    recording_path.write_bytes(b"webm-bytes")
    metadata_path = tmp_path / "rec-test.detections.json"
    metadata_path.write_text('{"detections":[]}', encoding="utf-8")

    def handler(_request):
        return httpx.Response(201)

    def client_factory():
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            timeout=settings.remote_storage_timeout,
        )

    async def run():
        storage = RemoteStorage(settings, client_factory=client_factory)
        await storage.start()
        try:
            await storage.submit_recording(
                RecordingRecord(
                    recording_id="rec-test",
                    filename="rec-test.webm",
                    path=recording_path,
                    content_type="video/webm",
                    byte_length=len(b"webm-bytes"),
                    duration_ms=None,
                    started_at=None,
                    created_at=1710000000.0,
                    storage_mode="remote",
                    local_saved=False,
                    metadata_path=metadata_path,
                )
            )
            await asyncio.wait_for(storage.drain(), timeout=1)
            return await storage.snapshot()
        finally:
            await storage.stop()

    status = asyncio.run(run())

    assert status["recordings_uploaded"] == 1
    assert not recording_path.exists()
    assert not metadata_path.exists()
