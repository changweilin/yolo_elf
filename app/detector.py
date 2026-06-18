from __future__ import annotations

import io
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from app.config import DETECT_MODES, Settings


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


def device_supports_half(device: str | int | None) -> bool:
    if isinstance(device, int):
        return device >= 0
    if device is None:
        return False
    normalized = str(device).strip().lower()
    return normalized == "cuda" or normalized.startswith("cuda:") or normalized.isdigit()


class YoloDetector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._mode = settings.detect_mode
        self._models: dict[str, Any] = {}
        self._names_by_mode: dict[str, dict[int, str]] = {}
        self._names: dict[int, str] = {}
        self._device: str | int | None = None
        self._device_resolved = False
        self._half_enabled = False
        self._warmed_up = False
        self._warmup_ms = 0.0
        self._load_error: str | None = None
        self._last_warmup_error: str | None = None

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def loaded(self) -> bool:
        return self._models.get(self._mode) is not None

    def _model_name_for_mode(self, mode: str) -> str:
        if mode == "accurate":
            return self.settings.yolo_model_accurate
        return self.settings.yolo_model

    def models_by_mode(self) -> dict[str, str]:
        return {mode: self._model_name_for_mode(mode) for mode in DETECT_MODES}

    def set_mode(self, mode: Any) -> str:
        normalized = str(mode).strip().lower() if mode is not None else ""
        if normalized not in DETECT_MODES:
            raise ValueError(
                f"Detection mode must be one of {', '.join(DETECT_MODES)}, got {mode!r}"
            )
        self._mode = normalized
        return normalized

    def status(self) -> dict[str, Any]:
        cuda_available: bool | None
        cuda_device_count: int | None
        cuda_device_name: str | None
        cuda_version: str | None
        torch_version: str | None
        resolved_device: str | int | None
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
            cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
            cuda_version = str(torch.version.cuda)
            torch_version = str(torch.__version__)
            resolved_device = self._device if self._device is not None else self._resolve_device(torch)
        except Exception:
            cuda_available = None
            cuda_device_count = None
            cuda_device_name = None
            cuda_version = None
            torch_version = None
            resolved_device = self._device

        return {
            "model": self._model_name_for_mode(self._mode),
            "mode": self._mode,
            "available_modes": list(DETECT_MODES),
            "models": self.models_by_mode(),
            "loaded": self.loaded,
            "device": self._device,
            "resolved_device": resolved_device,
            "requested_device": self.settings.yolo_device,
            "half": self._half_enabled,
            "requested_half": self.settings.yolo_half,
            "warmup_enabled": self.settings.yolo_warmup,
            "warmup_runs": self.settings.yolo_warmup_runs,
            "warmed_up": self._warmed_up,
            "warmup_ms": self._warmup_ms,
            "conf_thresh": self.settings.conf_thresh,
            "img_size": self.settings.img_size,
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
            "cuda_device_name": cuda_device_name,
            "cuda_version": cuda_version,
            "torch_version": torch_version,
            "last_load_error": self._load_error,
            "last_warmup_error": self._last_warmup_error,
        }

    def detect(self, jpeg_bytes: bytes, frame_id: int) -> dict[str, Any]:
        decoded = self._decode_jpeg(jpeg_bytes)
        model = self._ensure_model()

        started = time.perf_counter()
        try:
            results = self._predict(model, decoded.data)
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

    def warmup(self) -> dict[str, Any]:
        if not self.settings.yolo_warmup:
            return {"enabled": False, "ok": True, "warmup_ms": 0.0}

        started = time.perf_counter()
        try:
            import numpy as np

            model = self._ensure_model()
            warmup_size = min(max(self.settings.img_size, 32), 1280)
            source = np.zeros((warmup_size, warmup_size, 3), dtype=np.uint8)
            for _ in range(self.settings.yolo_warmup_runs):
                self._predict(model, source)
            self._synchronize_device()
        except Exception as exc:
            self._last_warmup_error = f"YOLO warmup failed: {exc}"
            return {
                "enabled": True,
                "ok": False,
                "warmup_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "error": self._last_warmup_error,
            }

        self._warmed_up = True
        self._warmup_ms = round((time.perf_counter() - started) * 1000.0, 2)
        self._last_warmup_error = None
        return {"enabled": True, "ok": True, "warmup_ms": self._warmup_ms}

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
        mode = self._mode
        cached = self._models.get(mode)
        if cached is not None:
            self._names = self._names_by_mode.get(mode, {})
            return cached

        try:
            import torch
            from ultralytics import YOLO
        except Exception as exc:
            self._load_error = f"YOLO dependencies are not installed: {exc}"
            raise DetectionError(self._load_error) from exc

        if not self._device_resolved:
            self._device = self._resolve_device(torch)
            self._half_enabled = self.settings.yolo_half and device_supports_half(self._device)
            self._device_resolved = True

        model_name = self._model_name_for_mode(mode)
        try:
            model = YOLO(model_name)
            names = getattr(model, "names", {}) or {}
            if isinstance(names, dict):
                resolved_names = {int(key): str(value) for key, value in names.items()}
            else:
                resolved_names = {index: str(value) for index, value in enumerate(names)}
        except Exception as exc:
            self._load_error = f"Could not load YOLO model {model_name!r}: {exc}"
            raise DetectionError(self._load_error) from exc

        self._models[mode] = model
        self._names_by_mode[mode] = resolved_names
        self._names = resolved_names
        self._load_error = None
        return model

    def _resolve_device(self, torch_module: Any) -> str | int | None:
        requested_device = self.settings.yolo_device.strip().lower()
        if requested_device == "auto":
            return 0 if torch_module.cuda.is_available() else "cpu"
        if requested_device in {"", "none"}:
            return None
        return self.settings.yolo_device

    def _prediction_kwargs(self, source: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "source": source,
            "imgsz": self.settings.img_size,
            "conf": self.settings.conf_thresh,
            "verbose": False,
        }
        if self._device is not None:
            kwargs["device"] = self._device
        if self._half_enabled:
            kwargs["half"] = True
        return kwargs

    def _predict(self, model: Any, source: Any) -> Any:
        try:
            import torch

            context = torch.inference_mode()
        except Exception:
            context = nullcontext()

        with context:
            return model.predict(**self._prediction_kwargs(source))

    def _synchronize_device(self) -> None:
        if not device_supports_half(self._device):
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            return

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
