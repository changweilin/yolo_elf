from __future__ import annotations

import asyncio
import base64
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


class StreamHub:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.frame_queue: asyncio.Queue[CameraFrame] = asyncio.Queue(maxsize=1)
        self.viewer_clients: set[WebSocket] = set()
        self.camera_client: WebSocket | None = None
        self.started_at = time.time()
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
        self._lock = asyncio.Lock()

    async def set_camera(self, websocket: WebSocket | None) -> None:
        async with self._lock:
            self.camera_client = websocket

    async def clear_camera(self, websocket: WebSocket) -> None:
        async with self._lock:
            if self.camera_client is websocket:
                self.camera_client = None

    async def add_viewer(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.viewer_clients.add(websocket)

    async def remove_viewer(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.viewer_clients.discard(websocket)

    async def submit_frame(self, jpeg: bytes) -> int:
        if len(jpeg) > self.settings.max_frame_bytes:
            raise ValueError(
                f"Frame is too large: {len(jpeg)} bytes > {self.settings.max_frame_bytes} bytes"
            )

        received_at = time.time()
        async with self._lock:
            self.frames_received += 1
            frame_id = self.frames_received
            self.last_frame_bytes = len(jpeg)
            self.last_received_at = received_at

        frame = CameraFrame(frame_id=frame_id, jpeg=jpeg, received_at=received_at)
        while self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.task_done()
                async with self._lock:
                    self.frames_dropped += 1
            except asyncio.QueueEmpty:
                break

        await self.frame_queue.put(frame)
        return frame_id

    async def publish_detection(
        self,
        frame: CameraFrame,
        detection: dict[str, Any],
        processing_started_at: float | None = None,
    ) -> None:
        completed_at = time.time()
        inference_ms = float(detection.get("inference_ms") or 0.0)
        queue_latency_ms = (
            max(0.0, processing_started_at - frame.received_at) * 1000.0
            if processing_started_at is not None
            else 0.0
        )
        total_latency_ms = max(0.0, completed_at - frame.received_at) * 1000.0

        async with self._lock:
            self.frames_processed += 1
            self.latest_frame = frame
            self.latest_detection = detection
            self.last_inference_ms = inference_ms
            self.last_queue_latency_ms = round(queue_latency_ms, 2)
            self.last_total_latency_ms = round(total_latency_ms, 2)
            self.last_processed_at = completed_at
            self.total_inference_ms += inference_ms
            self.total_queue_latency_ms += queue_latency_ms
            self.total_total_latency_ms += total_latency_ms
            self.last_error = detection.get("error")
            camera = self.camera_client
            viewers = list(self.viewer_clients)

        camera_payload = {"type": "detection", "detection": detection}
        if camera is not None:
            ok = await self._safe_send_json(camera, camera_payload)
            if not ok:
                await self.clear_camera(camera)

        if not viewers:
            return

        viewer_payload = self._viewer_payload(frame, detection)
        for viewer in viewers:
            ok = await self._safe_send_json(viewer, viewer_payload)
            if not ok:
                await self.remove_viewer(viewer)

    async def snapshot(self, detector_status: dict[str, Any] | None = None) -> dict[str, Any]:
        async with self._lock:
            uptime_sec = max(0.0, time.time() - self.started_at)
            processed_count = self.frames_processed
            return {
                "uptime_sec": round(uptime_sec, 1),
                "camera_connected": self.camera_client is not None,
                "viewer_count": len(self.viewer_clients),
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
                "detector": detector_status or {},
            }

    async def latest_viewer_payload(self) -> dict[str, Any] | None:
        async with self._lock:
            frame = self.latest_frame
            detection = self.latest_detection
        if frame is None or detection is None:
            return None
        return self._viewer_payload(frame, detection)

    def _viewer_payload(self, frame: CameraFrame, detection: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "frame",
            "captured_at": frame.received_at,
            "jpeg": "data:image/jpeg;base64," + base64.b64encode(frame.jpeg).decode("ascii"),
            "detection": detection,
        }

    async def _safe_send_json(self, websocket: WebSocket, payload: dict[str, Any]) -> bool:
        try:
            await websocket.send_json(payload)
            return True
        except Exception:
            return False


def _average(total: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return round(total / count, 2)


def _rate(count: int, uptime_sec: float) -> float:
    if count <= 0 or uptime_sec <= 0.0:
        return 0.0
    return round(count / uptime_sec, 2)
