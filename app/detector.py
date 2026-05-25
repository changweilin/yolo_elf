from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any

from app.config import Settings


class DetectionError(RuntimeError):
    """Raised when a frame cannot be decoded or inferred."""


@dataclass(frozen=True)
class DecodedImage:
    data: Any
    width: int
    height: int


def clamp_xyxy(xyxy: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def detection_error_payload(frame_id: int, message: str) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "width": 0,
        "height": 0,
        "inference_ms": 0.0,
        "boxes": [],
        "error": message,
    }


class YoloDetector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._names: dict[int, str] = {}
        self._device: str | int | None = None
        self._load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def status(self) -> dict[str, Any]:
        cuda_available: bool | None
        torch_version: str | None
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            torch_version = str(torch.__version__)
        except Exception:
            cuda_available = None
            torch_version = None

        return {
            "model": self.settings.yolo_model,
            "loaded": self.loaded,
            "device": self._device,
            "requested_device": self.settings.yolo_device,
            "conf_thresh": self.settings.conf_thresh,
            "img_size": self.settings.img_size,
            "cuda_available": cuda_available,
            "torch_version": torch_version,
            "last_load_error": self._load_error,
        }

    def detect(self, jpeg_bytes: bytes, frame_id: int) -> dict[str, Any]:
        decoded = self._decode_jpeg(jpeg_bytes)
        model = self._ensure_model()

        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "source": decoded.data,
            "imgsz": self.settings.img_size,
            "conf": self.settings.conf_thresh,
            "verbose": False,
        }
        if self._device is not None:
            kwargs["device"] = self._device

        try:
            results = model.predict(**kwargs)
        except Exception as exc:
            raise DetectionError(f"YOLO inference failed: {exc}") from exc

        inference_ms = (time.perf_counter() - started) * 1000.0
        boxes = self._extract_boxes(results[0], decoded.width, decoded.height)
        return {
            "frame_id": frame_id,
            "width": decoded.width,
            "height": decoded.height,
            "inference_ms": round(inference_ms, 2),
            "boxes": boxes,
        }

    def _decode_jpeg(self, jpeg_bytes: bytes) -> DecodedImage:
        try:
            import numpy as np
            from PIL import Image
        except Exception as exc:
            raise DetectionError(f"Image dependencies are not installed: {exc}") from exc

        try:
            with Image.open(io.BytesIO(jpeg_bytes)) as image:
                rgb_image = image.convert("RGB")
                width, height = rgb_image.size
                data = np.asarray(rgb_image)
        except Exception as exc:
            raise DetectionError(f"Invalid JPEG frame: {exc}") from exc

        return DecodedImage(data=data, width=width, height=height)

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            import torch
            from ultralytics import YOLO
        except Exception as exc:
            self._load_error = f"YOLO dependencies are not installed: {exc}"
            raise DetectionError(self._load_error) from exc

        requested_device = self.settings.yolo_device.strip().lower()
        if requested_device == "auto":
            self._device = 0 if torch.cuda.is_available() else "cpu"
        elif requested_device in {"", "none"}:
            self._device = None
        else:
            self._device = self.settings.yolo_device

        try:
            self._model = YOLO(self.settings.yolo_model)
            names = getattr(self._model, "names", {}) or {}
            if isinstance(names, dict):
                self._names = {int(key): str(value) for key, value in names.items()}
            else:
                self._names = {index: str(value) for index, value in enumerate(names)}
        except Exception as exc:
            self._load_error = f"Could not load YOLO model {self.settings.yolo_model!r}: {exc}"
            raise DetectionError(self._load_error) from exc

        self._load_error = None
        return self._model

    def _extract_boxes(self, result: Any, width: int, height: int) -> list[dict[str, Any]]:
        extracted: list[dict[str, Any]] = []
        result_boxes = getattr(result, "boxes", None)
        if result_boxes is None:
            return extracted

        for box in result_boxes:
            xyxy = box.xyxy[0].detach().cpu().tolist()
            class_id = int(box.cls[0].detach().cpu().item())
            confidence = float(box.conf[0].detach().cpu().item())
            extracted.append(
                {
                    "xyxy": clamp_xyxy(xyxy, width, height),
                    "class_id": class_id,
                    "label": self._names.get(class_id, str(class_id)),
                    "confidence": round(confidence, 4),
                }
            )
        return extracted
