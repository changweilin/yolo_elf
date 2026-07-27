from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from fastapi import WebSocket

from app.config import Settings


@dataclass(frozen=True)
class CameraFrame:
    frame_id: int
    jpeg: bytes
    received_at: float


class StreamChannel:
    """All per-camera stream state: transport, counters and latest frame.

    One instance per entry in ``CAMERAS``. Counters are deliberately private to
    the channel so two recorders can never pollute each other's fps / drop
    numbers, and the single-slot ``frame_queue`` keeps the "always process the
    newest frame, discard the backlog" latency behaviour per stream.
    """

    def __init__(self, camera_id: str, display_name: str) -> None:
        self.camera_id = camera_id
        self.display_name = display_name
        self.frame_queue: asyncio.Queue[CameraFrame] = asyncio.Queue(maxsize=1)
        self.camera_client: WebSocket | None = None
        self.camera_storage_mode: str | None = None
        self.camera_recording: bool = False
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.last_inference_ms = 0.0
        self.last_queue_latency_ms = 0.0
        self.last_total_latency_ms = 0.0
        self.last_frame_bytes = 0
        self.last_received_at = 0.0
        self.last_processed_at = 0.0
        self.total_inference_ms = 0.0
        self.total_queue_latency_ms = 0.0
        self.total_total_latency_ms = 0.0
        self.last_error: str | None = None
        self.latest_detection: dict[str, Any] | None = None
        self.latest_frame: CameraFrame | None = None
        # Newest VLM channel payload (open-vocab boxes + optional caption),
        # retained so a viewer joining mid-stream can be primed. The VLM worker
        # writes it via broadcast_vlm; it never touches the YOLO frame path.
        self.latest_vlm: dict[str, Any] | None = None

    def snapshot(self, uptime_sec: float) -> dict[str, Any]:
        processed_count = self.frames_processed
        return {
            "camera_id": self.camera_id,
            "display_name": self.display_name,
            "camera_connected": self.camera_client is not None,
            "camera_storage_mode": self.camera_storage_mode,
            "camera_recording": self.camera_recording,
            "frames_received": self.frames_received,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "receive_fps": _rate(self.frames_received, uptime_sec),
            "process_fps": _rate(self.frames_processed, uptime_sec),
            "queue_depth": self.frame_queue.qsize(),
            "last_frame_bytes": self.last_frame_bytes,
            "last_inference_ms": self.last_inference_ms,
            "last_queue_latency_ms": self.last_queue_latency_ms,
            "last_total_latency_ms": self.last_total_latency_ms,
            "avg_inference_ms": _average(self.total_inference_ms, processed_count),
            "avg_queue_latency_ms": _average(self.total_queue_latency_ms, processed_count),
            "avg_total_latency_ms": _average(self.total_total_latency_ms, processed_count),
            "last_error": self.last_error,
            "latest_frame_id": (
                self.latest_detection.get("frame_id") if self.latest_detection else None
            ),
        }


class StreamRegistry:
    """Owns every :class:`StreamChannel` plus the viewers watching them.

    Scheduling: instead of polling N per-camera queues, submitters push the
    *camera id* onto one shared ready queue and the single detection worker pops
    ids from it. That is round-robin by construction and preserves the
    single-slot "newest frame wins" behaviour per camera. A camera already
    marked ready is never enqueued twice, so a fast stream cannot starve a slow
    one out of the rotation.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.channels: dict[str, StreamChannel] = {
            camera_id: StreamChannel(camera_id, display_name)
            for camera_id, display_name in settings.cameras
        }
        self.camera_ids: tuple[str, ...] = tuple(self.channels)
        self.default_camera_id: str = self.camera_ids[0]
        # Viewer -> the camera it subscribed to; ``None`` means "every camera".
        self.viewer_clients: dict[WebSocket, str | None] = {}
        self.started_at = time.time()
        self._ready_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=len(self.channels))
        self._ready_pending: set[str] = set()
        self._lock = asyncio.Lock()

    @property
    def multi_camera(self) -> bool:
        return len(self.channels) > 1

    def resolve_camera_id(self, camera_id: str | None) -> str:
        """Map a client-supplied id onto a known channel (fail loud otherwise)."""
        if camera_id is None or camera_id == "":
            return self.default_camera_id
        if camera_id not in self.channels:
            raise ValueError(f"Unknown camera_id {camera_id!r}")
        return camera_id

    def channel(self, camera_id: str | None = None) -> StreamChannel:
        return self.channels[self.resolve_camera_id(camera_id)]

    def latest_frame(self, camera_id: str) -> CameraFrame | None:
        """The newest processed frame for a camera, or ``None`` if none yet.

        The VLM worker samples this instead of pulling from the ready queue, so
        the slow VLM pass never competes with YOLO for live frames.
        """
        channel = self.channels.get(camera_id)
        return channel.latest_frame if channel else None

    def tracker_key(self, camera_id: str) -> str | None:
        """The detector's tracker slot for a camera.

        The first camera reuses the detector's primary model (so a single-camera
        deployment loads exactly one set of weights, unchanged); every extra
        camera gets its own instance so ultralytics' persistent tracker state
        never mixes track ids across streams.
        """
        if camera_id == self.default_camera_id:
            return None
        return camera_id

    def public_cameras(self) -> list[dict[str, str]]:
        return [
            {"camera_id": channel.camera_id, "display_name": channel.display_name}
            for channel in self.channels.values()
        ]

    async def set_camera(self, websocket: WebSocket | None, camera_id: str | None = None) -> None:
        channel = self.channel(camera_id)
        async with self._lock:
            channel.camera_client = websocket
            channel.camera_storage_mode = None
            channel.camera_recording = False

    async def clear_camera(self, websocket: WebSocket, camera_id: str | None = None) -> None:
        channel = self.channel(camera_id)
        async with self._lock:
            if channel.camera_client is websocket:
                channel.camera_client = None
                channel.camera_storage_mode = None
                channel.camera_recording = False

    async def set_camera_state(
        self,
        storage_mode: str | None,
        recording: bool | None,
        camera_id: str | None = None,
    ) -> None:
        channel = self.channel(camera_id)
        async with self._lock:
            if storage_mode in {"local", "remote", "both"}:
                channel.camera_storage_mode = storage_mode
            if recording is not None:
                channel.camera_recording = bool(recording)

    async def add_viewer(self, websocket: WebSocket, camera_id: str | None = None) -> None:
        subscription = self.resolve_camera_id(camera_id) if camera_id else None
        async with self._lock:
            self.viewer_clients[websocket] = subscription

    async def remove_viewer(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.viewer_clients.pop(websocket, None)

    async def submit_frame(self, jpeg: bytes, camera_id: str | None = None) -> int:
        if len(jpeg) > self.settings.max_frame_bytes:
            raise ValueError(
                f"Frame is too large: {len(jpeg)} bytes > {self.settings.max_frame_bytes} bytes"
            )

        channel = self.channel(camera_id)
        received_at = time.time()
        async with self._lock:
            channel.frames_received += 1
            frame_id = channel.frames_received
            channel.last_frame_bytes = len(jpeg)
            channel.last_received_at = received_at

        frame = CameraFrame(frame_id=frame_id, jpeg=jpeg, received_at=received_at)
        while channel.frame_queue.full():
            try:
                channel.frame_queue.get_nowait()
                channel.frame_queue.task_done()
                async with self._lock:
                    channel.frames_dropped += 1
            except asyncio.QueueEmpty:
                break

        await channel.frame_queue.put(frame)
        await self._mark_ready(channel.camera_id)
        return frame_id

    async def next_frame(self) -> tuple[str, CameraFrame]:
        """Block until some camera has a frame; return it with its camera id."""
        while True:
            camera_id = await self._ready_queue.get()
            self._ready_queue.task_done()
            async with self._lock:
                self._ready_pending.discard(camera_id)
            channel = self.channels.get(camera_id)
            if channel is None:
                continue
            try:
                frame = channel.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                # The frame was already claimed on a previous pass (a submit can
                # re-arm the signal between the pop and the drain); wait again.
                continue
            channel.frame_queue.task_done()
            return camera_id, frame

    async def _mark_ready(self, camera_id: str) -> None:
        async with self._lock:
            if camera_id in self._ready_pending:
                return
            self._ready_pending.add(camera_id)
        # At most one entry per camera is ever pending, and the queue is sized to
        # the channel count, so this can never block or raise QueueFull.
        self._ready_queue.put_nowait(camera_id)

    async def publish_detection(
        self,
        frame: CameraFrame,
        detection: dict[str, Any],
        processing_started_at: float | None = None,
        camera_id: str | None = None,
    ) -> None:
        channel = self.channel(camera_id)
        completed_at = time.time()
        inference_ms = float(detection.get("inference_ms") or 0.0)
        queue_latency_ms = (
            max(0.0, processing_started_at - frame.received_at) * 1000.0
            if processing_started_at is not None
            else 0.0
        )
        total_latency_ms = max(0.0, completed_at - frame.received_at) * 1000.0

        async with self._lock:
            channel.frames_processed += 1
            channel.latest_frame = frame
            channel.latest_detection = detection
            channel.last_inference_ms = inference_ms
            channel.last_queue_latency_ms = round(queue_latency_ms, 2)
            channel.last_total_latency_ms = round(total_latency_ms, 2)
            channel.last_processed_at = completed_at
            channel.total_inference_ms += inference_ms
            channel.total_queue_latency_ms += queue_latency_ms
            channel.total_total_latency_ms += total_latency_ms
            channel.last_error = detection.get("error")
            camera = channel.camera_client
            viewers = self._subscribed_viewers(channel.camera_id)

        camera_payload = {"type": "detection", "detection": detection}
        if camera is not None:
            ok = await self._safe_send_json(camera, camera_payload)
            if not ok:
                await self.clear_camera(camera, channel.camera_id)

        if not viewers:
            return

        for viewer in viewers:
            ok = await self._safe_send_viewer_frame(viewer, channel, frame, detection)
            if not ok:
                await self.remove_viewer(viewer)

    async def broadcast_alert(self, events: list[dict[str, Any]]) -> None:
        """Push fired-alert messages (out of band from frames) to every viewer."""
        if not events:
            return
        for event in events:
            async with self._lock:
                viewers = self._subscribed_viewers(event.get("camera_id"))
            for viewer in viewers:
                if not await self._safe_send_json(viewer, event):
                    await self.remove_viewer(viewer)

    async def broadcast_vlm(self, camera_id: str, payload: dict[str, Any]) -> None:
        """Publish a VLM channel result (out of band from frames) to viewers.

        Retains the payload as the camera's ``latest_vlm`` so a viewer joining
        later can be primed, then fans it out to everyone watching this camera.
        """
        channel = self.channels.get(camera_id)
        if channel is None:
            return
        async with self._lock:
            channel.latest_vlm = payload
            viewers = self._subscribed_viewers(camera_id)
        for viewer in viewers:
            if not await self._safe_send_json(viewer, payload):
                await self.remove_viewer(viewer)

    def _subscribed_viewers(self, camera_id: str | None) -> list[WebSocket]:
        """Viewers watching ``camera_id`` (or watching everything). Lock held."""
        return [
            viewer
            for viewer, subscription in self.viewer_clients.items()
            if subscription is None or camera_id is None or subscription == camera_id
        ]

    async def snapshot(self, detector_status: dict[str, Any] | None = None) -> dict[str, Any]:
        async with self._lock:
            uptime_sec = max(0.0, time.time() - self.started_at)
            cameras = {
                camera_id: channel.snapshot(uptime_sec)
                for camera_id, channel in self.channels.items()
            }
            aggregate = _aggregate(list(self.channels.values()), uptime_sec)
            return {
                "uptime_sec": round(uptime_sec, 1),
                **aggregate,
                "viewer_count": len(self.viewer_clients),
                "multi_camera": self.multi_camera,
                "camera_ids": list(self.camera_ids),
                "default_camera_id": self.default_camera_id,
                "cameras": cameras,
                "detector": detector_status or {},
            }

    async def send_latest_to_viewer(self, websocket: WebSocket) -> bool:
        """Prime a freshly-connected viewer with the newest frame per camera
        (plus the latest VLM payload) so both channels paint without waiting."""
        async with self._lock:
            subscription = self.viewer_clients.get(websocket)
            pending = [
                (channel, channel.latest_frame, channel.latest_detection)
                for channel in self.channels.values()
                if subscription is None or subscription == channel.camera_id
            ]
            vlm_pending = [
                channel.latest_vlm
                for channel in self.channels.values()
                if (subscription is None or subscription == channel.camera_id)
                and channel.latest_vlm is not None
            ]
        for channel, frame, detection in pending:
            if frame is None or detection is None:
                continue
            if not await self._safe_send_viewer_frame(websocket, channel, frame, detection):
                return False
        for payload in vlm_pending:
            if not await self._safe_send_json(websocket, payload):
                return False
        return True

    def _viewer_metadata(
        self, channel: StreamChannel, frame: CameraFrame, detection: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "type": "frame",
            "camera_id": channel.camera_id,
            "display_name": channel.display_name,
            "captured_at": frame.received_at,
            "transport": "binary",
            "content_type": "image/jpeg",
            "byte_length": len(frame.jpeg),
            "detection": detection,
        }

    async def _safe_send_viewer_frame(
        self,
        websocket: WebSocket,
        channel: StreamChannel,
        frame: CameraFrame,
        detection: dict[str, Any],
    ) -> bool:
        try:
            await websocket.send_json(self._viewer_metadata(channel, frame, detection))
            await websocket.send_bytes(frame.jpeg)
            return True
        except Exception:
            return False

    async def _safe_send_json(self, websocket: WebSocket, payload: dict[str, Any]) -> bool:
        try:
            await websocket.send_json(payload)
            return True
        except Exception:
            return False



def _aggregate(channels: list[StreamChannel], uptime_sec: float) -> dict[str, Any]:
    """Server-wide roll-up of the per-camera counters.

    Counters sum; "last frame" fields come from whichever camera processed most
    recently. With one channel every value reduces to that channel's own number,
    so the existing single-camera status payload is unchanged.
    """
    received = sum(channel.frames_received for channel in channels)
    processed = sum(channel.frames_processed for channel in channels)
    latest = max(channels, key=lambda channel: channel.last_processed_at)
    active = max(channels, key=lambda channel: channel.last_received_at)
    return {
        "camera_connected": any(channel.camera_client is not None for channel in channels),
        "camera_storage_mode": active.camera_storage_mode,
        "camera_recording": active.camera_recording,
        "frames_received": received,
        "frames_processed": processed,
        "frames_dropped": sum(channel.frames_dropped for channel in channels),
        "receive_fps": _rate(received, uptime_sec),
        "process_fps": _rate(processed, uptime_sec),
        "queue_depth": sum(channel.frame_queue.qsize() for channel in channels),
        "last_frame_bytes": active.last_frame_bytes,
        "last_inference_ms": latest.last_inference_ms,
        "last_queue_latency_ms": latest.last_queue_latency_ms,
        "last_total_latency_ms": latest.last_total_latency_ms,
        "avg_inference_ms": _average(
            sum(channel.total_inference_ms for channel in channels), processed
        ),
        "avg_queue_latency_ms": _average(
            sum(channel.total_queue_latency_ms for channel in channels), processed
        ),
        "avg_total_latency_ms": _average(
            sum(channel.total_total_latency_ms for channel in channels), processed
        ),
        "last_error": latest.last_error,
        "latest_frame_id": (
            latest.latest_detection.get("frame_id") if latest.latest_detection else None
        ),
    }


def _average(total: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return round(total / count, 2)


def _rate(count: int, uptime_sec: float) -> float:
    if count <= 0 or uptime_sec <= 0.0:
        return 0.0
    return round(count / uptime_sec, 2)
