from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.detector import DetectionError, YoloDetector, detection_error_payload
from app.recordings import RecordingStore, recording_storage_mode
from app.remote_storage import RemoteStorage
from app.stream_state import CameraFrame, StreamHub


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    detector = YoloDetector(resolved_settings)
    hub = StreamHub(resolved_settings)
    recording_store = RecordingStore(resolved_settings)
    remote_storage = RemoteStorage(resolved_settings)

    async def detection_worker() -> None:
        while True:
            frame: CameraFrame = await hub.frame_queue.get()
            processing_started_at = time.time()
            try:
                detection = await asyncio.to_thread(detector.detect, frame.jpeg, frame.frame_id)
            except DetectionError as exc:
                detection = detection_error_payload(frame.frame_id, str(exc))
            except Exception as exc:
                detection = detection_error_payload(
                    frame.frame_id, f"Unexpected detector error: {exc}"
                )
            finally:
                hub.frame_queue.task_done()

            await hub.publish_detection(frame, detection, processing_started_at)
            await remote_storage.submit(frame, detection)

    async def status_payload() -> dict[str, Any]:
        status = await hub.snapshot(detector.status())
        status["recordings"] = await recording_store.snapshot()
        status["remote_storage"] = await remote_storage.snapshot()
        return status

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        api.state.settings = resolved_settings
        api.state.detector = detector
        api.state.hub = hub
        api.state.recording_store = recording_store
        api.state.remote_storage = remote_storage
        if resolved_settings.yolo_warmup:
            await asyncio.to_thread(detector.warmup)
        await remote_storage.start()
        worker = asyncio.create_task(detection_worker())
        try:
            yield
        finally:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
            await remote_storage.stop()

    api = FastAPI(title="YOLO Elf", version="0.1.0", lifespan=lifespan)
    api.mount("/static", StaticFiles(directory=resolved_settings.static_dir), name="static")

    @api.get("/")
    async def root() -> RedirectResponse:
        return RedirectResponse("/phone")

    @api.get("/phone")
    async def phone() -> FileResponse:
        return FileResponse(
            resolved_settings.static_dir / "phone.html",
            headers={"Cache-Control": "no-store"},
        )

    @api.get("/viewer")
    async def viewer() -> FileResponse:
        return FileResponse(resolved_settings.static_dir / "viewer.html")

    @api.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/api/status")
    async def api_status() -> dict[str, Any]:
        return await status_payload()

    @api.post("/api/recordings")
    async def api_recordings(request: Request) -> dict[str, Any]:
        storage_mode = recording_storage_mode(
            request.headers.get("x-yolo-elf-storage-mode")
        )
        wants_remote = storage_mode in {"remote", "both"}
        if wants_remote and not remote_storage.recordings_enabled:
            raise HTTPException(
                status_code=400,
                detail="Remote recording storage is not configured",
            )

        recording = await recording_store.save_request(request, storage_mode)
        remote_result = (
            await remote_storage.submit_recording(recording)
            if wants_remote
            else {"status": "skipped", "endpoint_configured": False}
        )
        return {
            "type": "recording",
            "recording": recording.public_payload(),
            "remote_storage": remote_result,
        }

    @api.get("/api/recordings/{recording_id}")
    async def api_recording(recording_id: str) -> FileResponse:
        path, media_type = recording_store.resolve(recording_id)
        return FileResponse(path, media_type=media_type, filename=path.name)

    @api.websocket("/ws/camera")
    async def camera_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        await hub.set_camera(websocket)
        await websocket.send_json(
            {
                "type": "config",
                "capture": {
                    "width": resolved_settings.capture_width,
                    "height": resolved_settings.capture_height,
                    "fps": resolved_settings.frame_fps,
                    "jpeg_quality": resolved_settings.jpeg_quality,
                    "max_frame_bytes": resolved_settings.max_frame_bytes,
                },
                "recording": {
                    "enabled": resolved_settings.recording_enabled,
                    "max_bytes": resolved_settings.recording_max_bytes,
                    "remote_upload_enabled": remote_storage.recordings_enabled,
                    "storage_modes": ["remote", "local", "both"],
                },
            }
        )
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    try:
                        await hub.submit_frame(message["bytes"])
                    except ValueError as exc:
                        await websocket.send_json(
                            {"type": "error", "message": str(exc), "retryable": True}
                        )
                elif message.get("text") is not None:
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            pass
        finally:
            await hub.clear_camera(websocket)

    @api.websocket("/ws/viewer")
    async def viewer_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        await hub.add_viewer(websocket)
        await websocket.send_json({"type": "status", "status": await status_payload()})
        if not await hub.send_latest_to_viewer(websocket):
            await hub.remove_viewer(websocket)
            return
        try:
            while True:
                await websocket.receive_text()
                await websocket.send_json({"type": "status", "status": await status_payload()})
        except WebSocketDisconnect:
            pass
        finally:
            await hub.remove_viewer(websocket)

    return api


app = create_app()
