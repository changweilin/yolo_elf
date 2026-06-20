from __future__ import annotations

import io
import threading
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
        # Runtime-mutable detector config (seeded from settings, then editable via
        # update_config / the settings page without restarting the server).
        self._model_names: dict[str, str] = {
            "fast": settings.yolo_model,
            "accurate": settings.yolo_model_accurate,
        }
        self._classes: tuple[str, ...] = settings.yolo_classes
        self._conf_thresh: float = settings.conf_thresh
        self._img_size: int = settings.img_size
        # Optional second-stage classifier: when a model name is set, every
        # detection box is cropped and classified so each box gets a fine-grained
        # `species` label on top of its coarse detection `label`. Empty = off.
        self._classifier_name: str = settings.classifier_model
        self._classifier_min_conf: float = settings.classifier_min_conf
        self._classifier_model: Any = None
        self._classifier_names: dict[int, str] = {}
        self._classifier_lock = threading.Lock()
        self._classifier_error: str | None = None
        self._models: dict[str, Any] = {}
        # Guards model loading so a background preload (triggered by a mode
        # switch) and the detection worker never load the same weights twice.
        self._load_lock = threading.Lock()
        self._names_by_mode: dict[str, dict[int, str]] = {}
        self._open_vocab_applied: dict[str, bool] = {}
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
        return self._model_names.get(mode, self._model_names["fast"])

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

    def update_config(self, payload: Any) -> dict[str, Any]:
        """Apply a partial detector config at runtime. Returns the new status.

        Only the keys present in ``payload`` are changed. Swapping a model name
        drops that preset's cached weights so the new model loads on the next
        frame; changing classes is re-applied to any already-loaded open-vocab
        model. Raises ``ValueError`` for invalid values.
        """
        if not isinstance(payload, dict):
            raise ValueError("Detector config must be an object")

        if payload.get("mode") is not None:
            self.set_mode(payload["mode"])
        if payload.get("conf_thresh") is not None:
            self._conf_thresh = self._validate_conf_thresh(payload["conf_thresh"])
        if payload.get("img_size") is not None:
            self._img_size = self._validate_img_size(payload["img_size"])
        if payload.get("fast_model") is not None:
            self._set_model_name("fast", payload["fast_model"])
        if payload.get("accurate_model") is not None:
            self._set_model_name("accurate", payload["accurate_model"])
        if payload.get("classes") is not None:
            self._set_classes(payload["classes"])
        if payload.get("classifier_model") is not None:
            self._set_classifier_name(payload["classifier_model"])
        if payload.get("classifier_min_conf") is not None:
            self._classifier_min_conf = self._validate_classifier_min_conf(
                payload["classifier_min_conf"]
            )

        return self.status()

    @staticmethod
    def _validate_conf_thresh(value: Any) -> float:
        try:
            conf = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("conf_thresh must be a number") from exc
        if not 0.0 <= conf <= 1.0:
            raise ValueError("conf_thresh must be between 0.0 and 1.0")
        return conf

    @staticmethod
    def _validate_img_size(value: Any) -> int:
        try:
            size = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("img_size must be an integer") from exc
        if not 32 <= size <= 4096:
            raise ValueError("img_size must be between 32 and 4096")
        return size

    @staticmethod
    def _validate_classifier_min_conf(value: Any) -> float:
        try:
            conf = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("classifier_min_conf must be a number") from exc
        if not 0.0 <= conf <= 1.0:
            raise ValueError("classifier_min_conf must be between 0.0 and 1.0")
        return conf

    def _set_classifier_name(self, name: Any) -> None:
        # Unlike the detection presets, an empty name is valid here: it disables
        # the second-stage classifier. Swapping the name drops the cached weights
        # so the new model (or the disabled state) takes effect on the next frame.
        resolved = str(name).strip()
        if resolved == self._classifier_name:
            return
        self._classifier_name = resolved
        self._classifier_model = None
        self._classifier_names = {}
        self._classifier_error = None

    def _set_model_name(self, mode: str, name: Any) -> None:
        resolved = str(name).strip()
        if not resolved:
            raise ValueError(f"{mode} model name must not be empty")
        if resolved == self._model_names.get(mode):
            return
        self._model_names[mode] = resolved
        # Drop cached state so the new weights load on the next frame.
        self._models.pop(mode, None)
        self._names_by_mode.pop(mode, None)
        self._open_vocab_applied.pop(mode, None)

    def _set_classes(self, classes: Any) -> None:
        if isinstance(classes, str):
            parsed = tuple(item.strip() for item in classes.split(",") if item.strip())
        elif isinstance(classes, (list, tuple)):
            parsed = tuple(str(item).strip() for item in classes if str(item).strip())
        else:
            raise ValueError("classes must be a list or comma-separated string")
        self._classes = parsed
        self._reapply_open_vocabulary()

    def _reapply_open_vocabulary(self) -> None:
        for mode, model in self._models.items():
            applied = self._apply_open_vocabulary(model)
            self._open_vocab_applied[mode] = applied
            if not applied:
                continue
            names = getattr(model, "names", {}) or {}
            if isinstance(names, dict):
                resolved = {int(key): str(value) for key, value in names.items()}
            else:
                resolved = {index: str(value) for index, value in enumerate(names)}
            self._names_by_mode[mode] = resolved
            if mode == self._mode:
                self._names = resolved

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
            "configured_classes": list(self._classes),
            "open_vocabulary": self._open_vocab_applied.get(self._mode, False),
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
            "conf_thresh": self._conf_thresh,
            "img_size": self._img_size,
            "classifier_model": self._classifier_name,
            "classifier_enabled": bool(self._classifier_name),
            "classifier_loaded": self._classifier_model is not None,
            "classifier_min_conf": self._classifier_min_conf,
            "last_classifier_error": self._classifier_error,
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
        if boxes and self._classifier_name:
            self._classify_boxes(boxes, decoded.data, decoded.width, decoded.height)
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
            warmup_size = min(max(self._img_size, 32), 1280)
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

    def preload(self) -> dict[str, Any]:
        """Eagerly load the current mode's weights so ``status().loaded`` flips.

        Called in a background thread after a mode switch so the UI progress bar
        can poll ``/api/status`` and tell when the new model is ready, even when
        no frames are streaming. Load failures are recorded in ``last_load_error``
        and swallowed so the caller never has to handle an exception.
        """
        try:
            self._ensure_model()
        except Exception:  # noqa: BLE001 - reported via last_load_error
            pass
        # Warm the second-stage classifier too (no-op when disabled). It never
        # raises; load failures surface via `last_classifier_error` in status.
        self._ensure_classifier()
        return self.status()

    def _ensure_model(self) -> Any:
        mode = self._mode
        cached = self._models.get(mode)
        if cached is not None:
            self._names = self._names_by_mode.get(mode, {})
            return cached

        with self._load_lock:
            # Re-check under the lock: a concurrent loader may have finished
            # while we were waiting, so we never build the same model twice.
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
                applied = self._apply_open_vocabulary(model)
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
            self._open_vocab_applied[mode] = applied
            self._names = resolved_names
            self._load_error = None
            return model

    def _apply_open_vocabulary(self, model: Any) -> bool:
        """Set custom prompt classes on open-vocabulary models (YOLO-World/YOLOE).

        Returns True when the configured ``YOLO_CLASSES`` were applied. Closed-set
        models (plain COCO/Open Images detectors) lack ``set_classes`` and keep
        their built-in vocabulary unchanged.
        """
        classes = self._classes
        if not classes:
            return False
        set_classes = getattr(model, "set_classes", None)
        if not callable(set_classes):
            return False
        set_classes(list(classes))
        return True

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
            "imgsz": self._img_size,
            "conf": self._conf_thresh,
            "verbose": False,
        }
        if self._device is not None:
            kwargs["device"] = self._device
        if self._half_enabled:
            kwargs["half"] = True
        return kwargs

    def _inference_context(self) -> Any:
        try:
            import torch

            return torch.inference_mode()
        except Exception:
            return nullcontext()

    def _predict(self, model: Any, source: Any) -> Any:
        with self._inference_context():
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

    def _ensure_classifier(self) -> Any:
        """Lazily load the second-stage classifier; ``None`` when disabled/failed.

        Unlike ``_ensure_model`` this never raises: a missing dependency or a bad
        model name is recorded in ``_classifier_error`` and surfaced via status so
        detection keeps working without species labels.
        """
        if not self._classifier_name:
            return None
        cached = self._classifier_model
        if cached is not None:
            return cached

        with self._classifier_lock:
            if self._classifier_model is not None:
                return self._classifier_model

            name = self._classifier_name
            try:
                from ultralytics import YOLO

                model = YOLO(name)
                names = getattr(model, "names", {}) or {}
                if isinstance(names, dict):
                    resolved = {int(key): str(value) for key, value in names.items()}
                else:
                    resolved = {index: str(value) for index, value in enumerate(names)}
            except Exception as exc:  # noqa: BLE001 - reported via _classifier_error
                self._classifier_error = f"Could not load classifier {name!r}: {exc}"
                return None

            self._classifier_model = model
            self._classifier_names = resolved
            self._classifier_error = None
            return model

    def _classify_boxes(
        self, boxes: list[dict[str, Any]], image: Any, width: int, height: int
    ) -> None:
        """Attach a fine-grained ``species`` label to each detection box.

        Crops every box from the frame, runs them through the classifier as one
        batch, and writes ``species``/``species_confidence``/``species_class_id``
        for boxes whose top-1 confidence clears ``classifier_min_conf``. Failures
        are swallowed (recorded in ``_classifier_error``) so detection survives.
        """
        classifier = self._ensure_classifier()
        if classifier is None:
            return

        crops: list[Any] = []
        targets: list[dict[str, Any]] = []
        for box in boxes:
            crop = self._crop_box(image, box["xyxy"], width, height)
            if crop is None:
                continue
            crops.append(crop)
            targets.append(box)

        if not crops:
            return

        try:
            with self._inference_context():
                results = classifier.predict(**self._classifier_prediction_kwargs(crops))
        except Exception as exc:  # noqa: BLE001 - reported via _classifier_error
            self._classifier_error = f"Classifier inference failed: {exc}"
            return

        for box, result in zip(targets, results):
            species = self._top_species(result)
            if species is None:
                continue
            box["species"] = species["label"]
            box["species_confidence"] = species["confidence"]
            box["species_class_id"] = species["class_id"]
        self._classifier_error = None

    @staticmethod
    def _crop_box(image: Any, xyxy: list[float], width: int, height: int) -> Any:
        import math

        x1, y1, x2, y2 = xyxy
        left = max(0, int(math.floor(x1)))
        top = max(0, int(math.floor(y1)))
        right = min(width, int(math.ceil(x2)))
        bottom = min(height, int(math.ceil(y2)))
        if right - left < 1 or bottom - top < 1:
            return None
        return image[top:bottom, left:right]

    def _classifier_prediction_kwargs(self, source: Any) -> dict[str, Any]:
        # Classification models use their own training resolution (typically 224),
        # so we intentionally omit `imgsz`/`conf` here and only mirror the device
        # and half-precision settings resolved for the detector.
        kwargs: dict[str, Any] = {"source": source, "verbose": False}
        if self._device is not None:
            kwargs["device"] = self._device
        if self._half_enabled:
            kwargs["half"] = True
        return kwargs

    def _top_species(self, result: Any) -> dict[str, Any] | None:
        probs = getattr(result, "probs", None)
        if probs is None:
            return None
        try:
            class_id = int(probs.top1)
            confidence = float(probs.top1conf)
        except Exception:
            return None
        if confidence < self._classifier_min_conf:
            return None
        return {
            "class_id": class_id,
            "label": self._classifier_names.get(class_id, str(class_id)),
            "confidence": round(confidence, 4),
        }

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
