import asyncio
import base64
import json

import httpx

from app.config import get_settings
from app.remote_storage import RemoteStorage
from app.stream_state import CameraFrame


REMOTE_ENV = [
    "REMOTE_STORAGE_URL",
    "REMOTE_STORAGE_TOKEN",
    "REMOTE_STORAGE_INCLUDE_FRAME",
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
