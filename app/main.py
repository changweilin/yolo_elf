from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles

from app.alerts import AlertEngine
from app.auth import COOKIE_NAME, AuthGuard
from app.config import Settings, get_settings
from app.detector import DetectionError, YoloDetector, detection_error_payload
from app.events import EventStore, SightingAggregator
from app.metrics import CONTENT_TYPE as METRICS_CONTENT_TYPE, render_metrics
from app.recordings import RecordingStore, recording_storage_mode
from app.remote_storage import RemoteStorage
from app.stream_state import StreamRegistry
from app.vlm import VlmEngine
from app.zones import ZoneEngine

# Private close code (4000-4999 is reserved for applications) telling a client
# its camera_id is not in the CAMERAS allowlist. Distinct from 1008, which the
# pages treat as "session expired, go log in".
UNKNOWN_CAMERA_CLOSE_CODE = 4404


async def _apply_camera_text(registry: StreamRegistry, camera_id: str, text: str) -> bool:
    """Handle a text frame from the phone. Returns True if it was a known message."""
    try:
        message = json.loads(text)
    except (ValueError, TypeError):
        return False
    if not isinstance(message, dict) or message.get("type") != "client_state":
        return False
    await registry.set_camera_state(
        message.get("storage_mode"), message.get("recording"), camera_id
    )
    return True


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    detector = YoloDetector(resolved_settings)
    vlm_engine = VlmEngine(resolved_settings)
    registry = StreamRegistry(resolved_settings)
    recording_store = RecordingStore(resolved_settings)
    remote_storage = RemoteStorage(resolved_settings)
    alert_engine = AlertEngine(resolved_settings)
    zone_engine = ZoneEngine(resolved_settings)
    event_store = EventStore(resolved_settings)
    sightings = SightingAggregator()
    auth_guard = AuthGuard(resolved_settings)

    async def detection_worker() -> None:
        # One worker for every camera: the GPU is the bottleneck, so streams are
        # serialized here rather than each fighting for the device from its own
        # thread. The registry's ready queue hands them out round-robin, and each
        # camera's single-slot queue means a backlog is dropped, not queued.
        while True:
            camera_id, frame = await registry.next_frame()
            processing_started_at = time.time()
            try:
                detection = await asyncio.to_thread(
                    detector.detect,
                    frame.jpeg,
                    frame.frame_id,
                    registry.tracker_key(camera_id),
                )
            except DetectionError as exc:
                detection = detection_error_payload(frame.frame_id, str(exc))
            except Exception as exc:
                detection = detection_error_payload(
                    frame.frame_id, f"Unexpected detector error: {exc}"
                )

            detection["camera_id"] = camera_id
            # Tag boxes with their ROI zones before anyone consumes the detection
            # so viewers, recordings and zone-scoped alert rules all agree.
            zone_engine.annotate(detection, camera_id)
            sightings.observe(detection, time.time(), camera_id)
            await registry.publish_detection(frame, detection, processing_started_at, camera_id)
            await remote_storage.submit(frame, detection)
            events = await alert_engine.process(frame, detection, camera_id)
            if events:
                await registry.broadcast_alert(events)

    async def vlm_worker() -> None:
        # Additive VLM channel. On its own slow cadence it samples each camera's
        # newest *processed* frame (registry.latest_frame) and broadcasts
        # open-vocabulary boxes as a distinct `vlm` message. Decoupled from the
        # per-frame detection worker so a slow Florence pass never blocks YOLO,
        # and reading the retained frame means it never steals from the ready
        # queue. Per-tick errors are recorded in vlm_engine.status() and skipped
        # so a transient failure never tears the worker down.
        interval = resolved_settings.vlm_interval_sec
        while True:
            await asyncio.sleep(interval)
            for camera_id in registry.camera_ids:
                frame = registry.latest_frame(camera_id)
                if frame is None:
                    continue
                try:
                    result = await asyncio.to_thread(vlm_engine.analyze, frame.jpeg)
                except Exception:
                    # VlmError (load/inference) is already recorded in
                    # vlm_engine.status(); skip this tick and try next interval.
                    continue
                await registry.broadcast_vlm(
                    camera_id,
                    {
                        "type": "vlm",
                        "camera_id": camera_id,
                        "frame_id": frame.frame_id,
                        **result,
                    },
                )

    async def event_flush_worker() -> None:
        # Finalize and persist sightings whose track has gone quiet. Decoupled
        # from the frame rate so SQLite writes stay infrequent and batched.
        interval = min(2.0, resolved_settings.event_expiry_sec)
        while True:
            await asyncio.sleep(interval)
            expired = sightings.take_expired(time.time(), resolved_settings.event_expiry_sec)
            await event_store.write(expired)

    async def status_payload(camera_id: str | None = None) -> dict[str, Any]:
        status = await registry.snapshot(detector.status())
        status["recordings"] = await recording_store.snapshot()
        status["remote_storage"] = await remote_storage.snapshot()
        status["alerts"] = await alert_engine.snapshot()
        status["zones"] = zone_engine.snapshot(camera_id)
        status["events"] = {**event_store.snapshot(), "active_sightings": sightings.active_count}
        status["vlm"] = vlm_engine.status()
        return status

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        api.state.settings = resolved_settings
        api.state.detector = detector
        api.state.vlm_engine = vlm_engine
        api.state.registry = registry
        api.state.recording_store = recording_store
        api.state.remote_storage = remote_storage
        api.state.alert_engine = alert_engine
        api.state.zone_engine = zone_engine
        api.state.event_store = event_store
        api.state.sightings = sightings
        api.state.auth_guard = auth_guard
        if resolved_settings.yolo_warmup:
            await asyncio.to_thread(detector.warmup)
        await remote_storage.start()
        await alert_engine.start()
        worker = asyncio.create_task(detection_worker())
        flusher = (
            asyncio.create_task(event_flush_worker())
            if resolved_settings.event_log_enabled
            else None
        )
        vlm_task = None
        if resolved_settings.vlm_enabled:
            # Load Florence-2 off the event loop so a slow first load never
            # blocks startup; the worker tolerates it not being ready yet.
            asyncio.create_task(asyncio.to_thread(vlm_engine.preload))
            vlm_task = asyncio.create_task(vlm_worker())
        try:
            yield
        finally:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
            if vlm_task is not None:
                vlm_task.cancel()
                try:
                    await vlm_task
                except asyncio.CancelledError:
                    pass
            if flusher is not None:
                flusher.cancel()
                try:
                    await flusher
                except asyncio.CancelledError:
                    pass
                # Persist whatever was still in flight so nothing is lost on stop.
                await event_store.write(sightings.take_all())
            await remote_storage.stop()
            await alert_engine.stop()

    api = FastAPI(title="YOLO Elf", version="0.1.0", lifespan=lifespan)
    api.mount("/static", StaticFiles(directory=resolved_settings.static_dir), name="static")

    # Paths reachable without a session so the login flow can bootstrap: the
    # login page + its static assets, the liveness probe, and the credential
    # exchange itself. Everything else is gated when auth is enabled.
    auth_exempt_paths = {"/login", "/api/login", "/api/logout", "/health", "/favicon.ico"}

    def _is_auth_exempt(path: str) -> bool:
        return path in auth_exempt_paths or path.startswith("/static/")

    def _resolve_camera(camera_id: str | None) -> str:
        """Map an optional ``camera_id`` query param onto a configured camera."""
        try:
            return registry.resolve_camera_id(camera_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _safe_next(target: str | None) -> str:
        # Local paths only, never a scheme/host, to avoid an open redirect.
        if target and target.startswith("/") and not target.startswith("//"):
            return target
        return "/"

    def _set_session_cookie(response: JSONResponse, request: Request) -> None:
        response.set_cookie(
            COOKIE_NAME,
            auth_guard.issue_cookie(),
            max_age=auth_guard.ttl,
            httponly=True,
            samesite="lax",
            secure=request.url.scheme == "https",
            path="/",
        )

    @api.middleware("http")
    async def enforce_auth(request: Request, call_next):
        if auth_guard.enabled and not _is_auth_exempt(request.url.path):
            if not auth_guard.request_ok(request):
                # Browser navigations get bounced to the login page; API/other
                # clients get a plain 401 so they can present a credential.
                if request.method == "GET" and "text/html" in request.headers.get(
                    "accept", ""
                ):
                    return RedirectResponse(
                        f"/login?next={_safe_next(request.url.path)}"
                    )
                return JSONResponse(
                    {"detail": "Authentication required"}, status_code=401
                )
        return await call_next(request)

    @api.get("/login")
    async def login_page() -> FileResponse:
        return FileResponse(
            resolved_settings.static_dir / "login.html",
            headers={"Cache-Control": "no-store"},
        )

    @api.post("/api/login")
    async def api_login(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = None
        token = body.get("token") if isinstance(body, dict) else None
        if not auth_guard.verify_credential(token):
            # Never echo the attempted token back (keep it out of logs/responses).
            raise HTTPException(status_code=401, detail="Invalid token")
        response = JSONResponse({"status": "ok"})
        _set_session_cookie(response, request)
        return response

    @api.post("/api/logout")
    async def api_logout() -> JSONResponse:
        response = JSONResponse({"status": "ok"})
        response.delete_cookie(COOKIE_NAME, path="/")
        return response

    @api.get("/")
    async def root() -> RedirectResponse:
        return RedirectResponse("/phone")

    @api.get("/phone")
    async def phone() -> FileResponse:
        return FileResponse(
            resolved_settings.static_dir / "phone.html",
            headers={"Cache-Control": "no-store"},
        )

    @api.get("/recorder")
    async def recorder() -> FileResponse:
        # Device-neutral alias for the capture/recording page. Works on any
        # device with a camera (phone or desktop webcam).
        return FileResponse(
            resolved_settings.static_dir / "phone.html",
            headers={"Cache-Control": "no-store"},
        )

    @api.get("/viewer")
    async def viewer() -> FileResponse:
        return FileResponse(
            resolved_settings.static_dir / "viewer.html",
            headers={"Cache-Control": "no-store"},
        )

    @api.get("/settings")
    async def settings_page() -> FileResponse:
        return FileResponse(
            resolved_settings.static_dir / "settings.html",
            headers={"Cache-Control": "no-store"},
        )

    @api.get("/history")
    async def history_page() -> FileResponse:
        return FileResponse(
            resolved_settings.static_dir / "history.html",
            headers={"Cache-Control": "no-store"},
        )

    @api.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/api/status")
    async def api_status(camera_id: str | None = None) -> dict[str, Any]:
        return await status_payload(_resolve_camera(camera_id))

    @api.get("/metrics")
    async def metrics() -> PlainTextResponse:
        if not resolved_settings.metrics_enabled:
            raise HTTPException(status_code=404, detail="Metrics are disabled")
        return PlainTextResponse(
            render_metrics(await status_payload()), media_type=METRICS_CONTENT_TYPE
        )

    @api.post("/api/detector/mode")
    async def api_detector_mode(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            body = None
        mode = body.get("mode") if isinstance(body, dict) else None
        try:
            detector.set_mode(mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Detection mode is invalid") from exc
        # Warm the new preset's weights in the background so the client progress
        # bar can poll `/api/status` and tell when the switch is actually ready,
        # even while no frames are streaming. preload() swallows load errors.
        asyncio.create_task(asyncio.to_thread(detector.preload))
        return {"type": "detector_mode", "detector": detector.status()}

    @api.post("/api/detector/task")
    async def api_detector_task(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            body = None
        task = body.get("task") if isinstance(body, dict) else None
        # `tasks` (a list) runs several heads on every frame and wins when both
        # are present; `task` remains the original single-head contract.
        tasks = body.get("tasks") if isinstance(body, dict) else None
        try:
            if tasks is not None:
                detector.set_tasks(tasks)
            else:
                detector.set_task(task)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Same background preload as the mode switch: each task has its own
        # checkpoint, so the client's progress bar can poll /api/status.
        asyncio.create_task(asyncio.to_thread(detector.preload))
        return {"type": "detector_task", "detector": detector.status()}

    @api.get("/api/detector/config")
    async def api_detector_config_get() -> dict[str, Any]:
        return {"type": "detector_config", "detector": detector.status()}

    @api.post("/api/detector/config")
    async def api_detector_config(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            body = None
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Detector config is invalid")
        try:
            detector.update_config(body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Warm the (possibly newly named) weights in the background so the
        # settings-page progress bar can poll `/api/status` for readiness.
        asyncio.create_task(asyncio.to_thread(detector.preload))
        return {"type": "detector_config", "detector": detector.status()}

    @api.get("/api/alerts")
    async def api_alerts_get() -> dict[str, Any]:
        return {"type": "alerts", "alerts": await alert_engine.snapshot()}

    @api.post("/api/alerts")
    async def api_alerts_set(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            body = None
        # Accept either {"rules": [...]} or a bare [...] array. The webhook target
        # stays env-only and is never settable here (SSRF guard).
        rules = body.get("rules") if isinstance(body, dict) else body
        try:
            await alert_engine.set_rules(rules)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"type": "alerts", "alerts": await alert_engine.snapshot()}

    @api.get("/api/events")
    async def api_events(
        limit: int = 100,
        since: float | None = None,
        until: float | None = None,
        label: str | None = None,
        zone: str | None = None,
        camera_id: str | None = None,
    ) -> dict[str, Any]:
        events = await event_store.query(
            limit=limit,
            since=since,
            until=until,
            label=label,
            zone=zone,
            camera_id=_resolve_camera(camera_id) if camera_id else None,
        )
        return {"type": "events", "events": events, "count": len(events)}

    @api.get("/api/cameras")
    async def api_cameras() -> dict[str, Any]:
        status = await registry.snapshot()
        return {
            "type": "cameras",
            "cameras": registry.public_cameras(),
            "default_camera_id": registry.default_camera_id,
            "multi_camera": registry.multi_camera,
            "streams": status["cameras"],
        }

    @api.get("/api/zones")
    async def api_zones_get(camera_id: str | None = None) -> dict[str, Any]:
        return {"type": "zones", "zones": zone_engine.snapshot(_resolve_camera(camera_id))}

    @api.post("/api/zones")
    async def api_zones_set(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            body = None
        # Accept either {"zones": [...]} or a bare [...] array. A camera_id in
        # the body (or omitted, meaning the default camera) scopes the write, so
        # editing the front door's polygons never clears the back yard's.
        zones = body.get("zones") if isinstance(body, dict) else body
        camera_id = _resolve_camera(body.get("camera_id") if isinstance(body, dict) else None)
        try:
            zone_engine.set_zones(zones, camera_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"type": "zones", "zones": zone_engine.snapshot(camera_id)}

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

    @api.post("/api/recordings/{recording_id}/metadata")
    async def api_recording_metadata(recording_id: str, request: Request) -> dict[str, Any]:
        storage_mode = recording_storage_mode(
            request.headers.get("x-yolo-elf-storage-mode")
        )
        wants_remote = storage_mode in {"remote", "both"}
        if wants_remote and not remote_storage.recordings_enabled:
            raise HTTPException(
                status_code=400,
                detail="Remote recording storage is not configured",
            )

        metadata = await request.json()
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=400, detail="Recording metadata is invalid")

        return {
            "type": "recording_metadata",
            "metadata": await recording_store.save_metadata(
                recording_id, metadata, storage_mode
            ),
        }

    @api.get("/api/recordings/{recording_id}")
    async def api_recording(recording_id: str) -> FileResponse:
        path, media_type = recording_store.resolve(recording_id)
        return FileResponse(path, media_type=media_type, filename=path.name)

    @api.get("/api/recordings/{recording_id}/metadata")
    async def api_recording_metadata_file(recording_id: str) -> FileResponse:
        path = recording_store.resolve_metadata(recording_id)
        return FileResponse(path, media_type="application/json", filename=path.name)

    @api.websocket("/ws/camera")
    async def camera_socket(websocket: WebSocket, camera_id: str | None = None) -> None:
        # HTTP middleware never sees WS handshakes, so gate them here. The
        # browser sends the session cookie automatically on same-origin connects.
        if auth_guard.enabled and not auth_guard.websocket_ok(websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            # Accept first, then validate: a close before the handshake completes
            # reaches the browser as an opaque 1006, so the recorder would never
            # learn *why* it was rejected.
            resolved_camera_id = registry.resolve_camera_id(camera_id)
        except ValueError as exc:
            await websocket.send_json(
                {"type": "error", "message": str(exc), "retryable": False, "fatal": True}
            )
            await websocket.close(code=UNKNOWN_CAMERA_CLOSE_CODE)
            return
        channel = registry.channel(resolved_camera_id)
        await registry.set_camera(websocket, resolved_camera_id)
        await websocket.send_json(
            {
                "type": "config",
                "camera_id": resolved_camera_id,
                "display_name": channel.display_name,
                "cameras": registry.public_cameras(),
                "capture": {
                    "width": resolved_settings.capture_width,
                    "height": resolved_settings.capture_height,
                    "fps": resolved_settings.frame_fps,
                    "jpeg_quality": resolved_settings.jpeg_quality,
                    "max_frame_bytes": resolved_settings.max_frame_bytes,
                },
                "detector": detector.status(),
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
                        await registry.submit_frame(message["bytes"], resolved_camera_id)
                    except ValueError as exc:
                        await websocket.send_json(
                            {"type": "error", "message": str(exc), "retryable": True}
                        )
                elif message.get("text") is not None:
                    if not await _apply_camera_text(
                        registry, resolved_camera_id, message["text"]
                    ):
                        await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            pass
        finally:
            await registry.clear_camera(websocket, resolved_camera_id)

    @api.websocket("/ws/viewer")
    async def viewer_socket(websocket: WebSocket, camera_id: str | None = None) -> None:
        if auth_guard.enabled and not auth_guard.websocket_ok(websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            # No camera_id = watch every camera (the grid); a camera_id narrows
            # the socket to that one stream so a focused viewer isn't paying for
            # frames it will not draw.
            subscription = registry.resolve_camera_id(camera_id) if camera_id else None
        except ValueError as exc:
            await websocket.send_json(
                {"type": "error", "message": str(exc), "retryable": False, "fatal": True}
            )
            await websocket.close(code=UNKNOWN_CAMERA_CLOSE_CODE)
            return
        await registry.add_viewer(websocket, subscription)
        await websocket.send_json(
            {"type": "status", "status": await status_payload(subscription)}
        )
        if not await registry.send_latest_to_viewer(websocket):
            await registry.remove_viewer(websocket)
            return
        try:
            while True:
                await websocket.receive_text()
                await websocket.send_json(
                    {"type": "status", "status": await status_payload(subscription)}
                )
        except WebSocketDisconnect:
            pass
        finally:
            await registry.remove_viewer(websocket)

    return api


app = create_app()
