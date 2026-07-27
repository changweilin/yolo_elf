from __future__ import annotations

import io
import threading
import time
from typing import Any

from app.config import Settings
from app.detector import clamp_xyxy, device_supports_half


class VlmError(RuntimeError):
    """Raised when a frame cannot be decoded or the VLM cannot infer it."""


# Florence-2 emits neither a per-box confidence nor a track id. The box schema
# (and the viewer) still expect a numeric confidence, so every VLM box carries
# this sentinel; ``track_id`` stays ``None``. This is why VLM boxes never feed
# zones / alerts / history — those need real confidences and stable track ids.
VLM_BOX_CONFIDENCE = 1.0

# Florence task tokens this engine drives. <OD> uses the model's built-in
# vocabulary; <OPEN_VOCABULARY_DETECTION> grounds a caller-supplied phrase list.
TASK_OD = "<OD>"
TASK_OPEN_VOCAB = "<OPEN_VOCABULARY_DETECTION>"


def florence_od_to_boxes(parsed: Any, width: int, height: int) -> list[dict[str, Any]]:
    """Map a Florence-2 post-processed detection result to the app's box schema.

    ``parsed`` is what ``processor.post_process_generation`` returns for a
    detection task: a dict with parallel ``bboxes`` (pixel ``[x1,y1,x2,y2]``) and
    a labels list (``labels`` for ``<OD>``, ``bboxes_labels`` for open-vocab).
    Pure and defensive so it is unit-testable without loading the model.
    """
    if not isinstance(parsed, dict):
        return []
    bboxes = parsed.get("bboxes") or []
    labels = parsed.get("labels")
    if not labels:
        labels = parsed.get("bboxes_labels") or []

    boxes: list[dict[str, Any]] = []
    for index, bbox in enumerate(bboxes):
        try:
            xyxy = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        if len(xyxy) < 4:
            continue
        label = str(labels[index]) if index < len(labels) else str(index)
        boxes.append(
            {
                "xyxy": clamp_xyxy(xyxy[:4], width, height),
                # Florence has no stable class ids; the label string is the
                # identity. -1 keeps colorForClass()/schemas happy on the client.
                "class_id": -1,
                "label": label,
                "confidence": VLM_BOX_CONFIDENCE,
                "track_id": None,
            }
        )
    return boxes


def _caption_text(parsed: Any) -> str | None:
    """Pull the caption string out of a Florence-2 caption result.

    ``_generate`` already unwraps ``{task: value}`` to ``value`` (a string for
    caption tasks); the dict branch is a defensive fallback. Empty -> ``None``.
    """
    if isinstance(parsed, str):
        return parsed.strip() or None
    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _torch_device(device: str | int | None) -> str:
    return f"cuda:{device}" if isinstance(device, int) else str(device)


def _decode_jpeg(jpeg_bytes: bytes) -> tuple[Any, int, int]:
    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001 - surfaced as a VlmError to the worker
        raise VlmError(f"Image dependencies are not installed: {exc}") from exc
    try:
        with Image.open(io.BytesIO(jpeg_bytes)) as image:
            rgb = image.convert("RGB")
    except Exception as exc:
        raise VlmError(f"Invalid JPEG frame: {exc}") from exc
    return rgb, rgb.width, rgb.height


class VlmEngine:
    """Florence-2 sidecar producing open-vocabulary boxes (and, in Phase 2, a
    scene caption). Runs on its own slow cadence, never on the per-frame path, so
    it cannot stall YOLO. Device / half / prompt classes mirror the YOLO detector.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model_name = settings.vlm_model
        self._classes: tuple[str, ...] = settings.yolo_classes
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device: str | int | None = None
        self._half = False
        self._device_resolved = False
        self._load_lock = threading.Lock()
        self._load_error: str | None = None
        self._last_error: str | None = None
        self._last_inference_ms = 0.0
        self._last_box_count = 0
        self._last_caption: str | None = None

    @property
    def detect_task(self) -> str:
        if self.settings.vlm_detect_task:
            return self.settings.vlm_detect_task
        return TASK_OPEN_VOCAB if self._classes else TASK_OD

    @property
    def caption_task(self) -> str:
        return self.settings.vlm_caption_task

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.settings.vlm_enabled,
            "model": self._model_name,
            "loaded": self.loaded,
            "device": self._device,
            "half": self._half,
            "detect_task": self.detect_task,
            "caption_enabled": self.settings.vlm_caption,
            "caption_task": self.caption_task,
            "configured_classes": list(self._classes),
            "interval_sec": self.settings.vlm_interval_sec,
            "last_inference_ms": self._last_inference_ms,
            "last_boxes": self._last_box_count,
            "last_caption": self._last_caption,
            "last_load_error": self._load_error,
            "last_error": self._last_error,
        }

    def preload(self) -> dict[str, Any]:
        """Eagerly load the model so ``status().loaded`` flips before the first
        frame. Swallows load errors (surfaced via ``last_load_error``)."""
        try:
            self._ensure_model()
        except Exception:  # noqa: BLE001 - reported via last_load_error
            pass
        return self.status()

    def detect(self, jpeg_bytes: bytes) -> dict[str, Any]:
        """Run open-vocabulary detection on one frame; returns the box payload."""
        image, width, height = _decode_jpeg(jpeg_bytes)
        model = self._ensure_model()
        task = self.detect_task
        prompt = self._build_prompt(task)

        started = time.perf_counter()
        try:
            parsed = self._generate(model, prompt, task, image, width, height)
        except Exception as exc:
            self._last_error = f"VLM inference failed: {exc}"
            raise VlmError(self._last_error) from exc

        inference_ms = round((time.perf_counter() - started) * 1000.0, 2)
        boxes = florence_od_to_boxes(parsed, width, height)
        self._last_inference_ms = inference_ms
        self._last_box_count = len(boxes)
        self._last_error = None
        return {
            "width": width,
            "height": height,
            "inference_ms": inference_ms,
            "boxes": boxes,
        }

    def analyze(self, jpeg_bytes: bytes) -> dict[str, Any]:
        """Detect + (optionally) caption one frame in a single decode.

        The Phase 2 worker entry point: returns the same box payload as
        :meth:`detect` plus a ``description`` string (``None`` when captioning is
        off or fails). A caption failure is best-effort and never drops the
        boxes; a detection failure raises ``VlmError`` so the worker skips the tick.
        """
        image, width, height = _decode_jpeg(jpeg_bytes)
        model = self._ensure_model()
        detect_task = self.detect_task

        started = time.perf_counter()
        try:
            parsed = self._generate(
                model, self._build_prompt(detect_task), detect_task, image, width, height
            )
        except Exception as exc:
            self._last_error = f"VLM inference failed: {exc}"
            raise VlmError(self._last_error) from exc
        boxes = florence_od_to_boxes(parsed, width, height)
        self._last_error = None

        description: str | None = None
        if self.settings.vlm_caption:
            caption_task = self.caption_task
            try:
                caption_parsed = self._generate(
                    model, caption_task, caption_task, image, width, height
                )
                description = _caption_text(caption_parsed)
            except Exception as exc:  # noqa: BLE001 - caption is best-effort
                self._last_error = f"VLM caption failed: {exc}"

        inference_ms = round((time.perf_counter() - started) * 1000.0, 2)
        self._last_inference_ms = inference_ms
        self._last_box_count = len(boxes)
        self._last_caption = description
        return {
            "width": width,
            "height": height,
            "inference_ms": inference_ms,
            "boxes": boxes,
            "description": description,
        }

    def _build_prompt(self, task: str) -> str:
        # Open-vocab detection takes the phrase list appended after the token
        # (Florence grounds each phrase); other tasks take the bare token.
        if task == TASK_OPEN_VOCAB and self._classes:
            return f"{task} {'. '.join(self._classes)}"
        return task

    def _resolve_device(self, torch_module: Any) -> str | int | None:
        requested = self.settings.yolo_device.strip().lower()
        if requested == "auto":
            return 0 if torch_module.cuda.is_available() else "cpu"
        if requested in {"", "none"}:
            return None
        return self.settings.yolo_device

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model

            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoProcessor
            except Exception as exc:
                self._load_error = f"VLM dependencies are not installed: {exc}"
                raise VlmError(self._load_error) from exc

            if not self._device_resolved:
                self._device = self._resolve_device(torch)
                self._half = self.settings.yolo_half and device_supports_half(self._device)
                self._device_resolved = True

            try:
                dtype = torch.float16 if self._half else torch.float32
                model = AutoModelForCausalLM.from_pretrained(
                    self._model_name, trust_remote_code=True, torch_dtype=dtype
                )
                processor = AutoProcessor.from_pretrained(
                    self._model_name, trust_remote_code=True
                )
                if self._device is not None:
                    model = model.to(_torch_device(self._device))
                model.eval()
            except Exception as exc:
                self._load_error = f"Could not load VLM {self._model_name!r}: {exc}"
                raise VlmError(self._load_error) from exc

            self._torch = torch
            self._processor = processor
            self._model = model
            self._load_error = None
            return model

    def _to_device(self, inputs: Any) -> Any:
        if self._device is None:
            return inputs
        target = _torch_device(self._device)
        moved = inputs.to(target) if hasattr(inputs, "to") else inputs
        if self._half:
            pixel_values = moved["pixel_values"]
            moved["pixel_values"] = pixel_values.to(self._torch.float16)
        return moved

    def _generate(
        self, model: Any, prompt: str, task: str, image: Any, width: int, height: int
    ) -> Any:
        processor = self._processor
        torch = self._torch
        inputs = self._to_device(processor(text=prompt, images=image, return_tensors="pt"))
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )
        text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(text, task=task, image_size=(width, height))
        # post_process_generation wraps its result under the task key.
        return parsed.get(task, parsed) if isinstance(parsed, dict) else parsed
