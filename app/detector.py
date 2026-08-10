from __future__ import annotations

import io
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import (
    BOX_TASKS,
    DETECT_MODES,
    DETECT_TASKS,
    MAX_ACTIVE_TASKS,
    END2END_FLAGS,
    END2END_MODES,
    RASTER_TASKS,
    Settings,
    parse_detect_tasks,
)


class DetectionError(RuntimeError):
    """Raised when a frame cannot be decoded or inferred."""


# Vertices kept per segmentation contour. Raw contours run to several hundred
# points; at 10 fps across every instance that dominates the payload, and the
# overlay is a translucent shape where the difference is not visible.
_MASK_MAX_POINTS = 48


def _class_color(class_id: int) -> tuple[int, int, int]:
    """A stable, reasonably distinct RGB colour for a class index.

    Deterministic on purpose: the semantic overlay is only readable if a class
    keeps the same colour frame to frame and across restarts. The golden-ratio
    hue step spreads adjacent ids far apart in hue rather than clustering them.
    """
    import colorsys

    hue = (class_id * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return (int(red * 255), int(green * 255), int(blue * 255))


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


def _box_track_id(box: Any) -> int | None:
    """Return the tracker id for a single result box, or ``None`` when absent.

    ``boxes.id`` is ``None`` until the tracker confirms a track (and always when
    tracking is off), so this stays defensive against the whole tensor chain.
    """
    box_id = getattr(box, "id", None)
    if box_id is None:
        return None
    try:
        return int(box_id[0].detach().cpu().item())
    except Exception:
        return None


def detection_error_payload(frame_id: int, message: str) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "width": 0,
        "height": 0,
        "inference_ms": 0.0,
        "boxes": [],
        "error": message,
    }


def model_supports_end2end(model: Any) -> bool:
    """Whether a freshly loaded model carries a one-to-one (NMS-free) head.

    Ultralytics hangs the flag off the wrapped ``nn.Module`` (``YOLO().model``),
    not the wrapper, and only YOLO26/YOLOv10 checkpoints set it — so this stays
    defensive and reports False for every older generation (and for exported
    ``.engine``/``.onnx``, where ``.model`` is still just the path string).

    Read this at load time only: predicting with an explicit ``end2end`` kwarg
    *overwrites* the attribute, so after one forced-off frame it would report
    False for a model that plainly has the head.
    """
    return bool(getattr(getattr(model, "model", None), "end2end", False))


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
        # `_tasks` is the running set; `_task` is its first entry, kept as the
        # single-task view every existing caller (and payload) still reads.
        self._tasks: tuple[str, ...] = settings.detect_tasks
        self._task = self._tasks[0]
        # Runtime-mutable detector config (seeded from settings, then editable via
        # update_config / the settings page without restarting the server).
        self._model_names: dict[str, str] = {
            "fast": settings.yolo_model,
            "accurate": settings.yolo_model_accurate,
        }
        # One model per non-detect task. Kept separate from `_model_names` because
        # the fast/accurate split only exists for "detect" — the other heads have
        # a single checkpoint each.
        self._task_model_names: dict[str, str] = dict(settings.task_model_map)
        self._classes: tuple[str, ...] = settings.yolo_classes
        self._conf_thresh: float = settings.conf_thresh
        self._img_size: int = settings.img_size
        # NMS-free head selection ("auto"/"on"/"off") and the per-frame box cap.
        # Both are plain predict kwargs, so unlike the model name they take effect
        # on the very next frame without dropping any cached weights.
        self._end2end: str = settings.yolo_end2end
        self._max_det: int = settings.yolo_max_det
        # Multi-object tracking: when on, detect() runs model.track(persist=True)
        # so each box carries a stable `track_id` across frames. Fixed at launch
        # (per-model tracker state makes a mid-stream toggle a footgun).
        self._track_enabled: bool = settings.yolo_track
        self._tracker: str = settings.yolo_tracker
        # Optional second-stage classifier: when a model name is set, every
        # detection box is cropped and classified so each box gets a fine-grained
        # `species` label on top of its coarse detection `label`. Empty = off.
        self._classifier_name: str = settings.classifier_model
        self._classifier_min_conf: float = settings.classifier_min_conf
        self._classifier_max_boxes: int = settings.classifier_max_boxes
        self._classifier_model: Any = None
        self._classifier_names: dict[int, str] = {}
        self._classifier_lock = threading.Lock()
        self._classifier_error: str | None = None
        # Every per-weights cache below is keyed by *preset key* (see
        # `_preset_key`), not by mode: switching task swaps the checkpoint just as
        # much as switching fast/accurate does, and each needs its own slot.
        self._models: dict[str, Any] = {}
        # Extra model instances, one per non-primary camera, keyed by
        # (preset_key, tracker_key). Ultralytics keeps its tracker state inside
        # the model object, so two streams sharing one instance would interleave
        # their track ids; a dedicated instance per stream is the only reliable
        # isolation. Costs one extra copy of the weights per extra camera, and
        # is only ever populated when tracking is on and >1 camera streams.
        self._tracker_models: dict[tuple[str, str], Any] = {}
        # The artifact actually loaded per preset (a `.pt`, or an auto-exported
        # `.engine`/`.onnx`) — may differ from the configured model name.
        self._loaded_sources: dict[str, str] = {}
        self._export_error: str | None = None
        # Guards model loading so a background preload (triggered by a mode
        # switch) and the detection worker never load the same weights twice.
        self._load_lock = threading.Lock()
        self._names_by_preset: dict[str, dict[int, str]] = {}
        # Each preset's NMS-free capability, sampled once at load time (see
        # model_supports_end2end for why it cannot be read back later).
        self._end2end_native: dict[str, bool] = {}
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
    def task(self) -> str:
        return self._task

    @property
    def tasks(self) -> tuple[str, ...]:
        return self._tasks

    @property
    def loaded(self) -> bool:
        """True once *every* active head has its weights in memory."""
        return all(self._models.get(self._preset_key(task)) is not None for task in self._tasks)

    def _preset_key(self, task: str | None = None, mode: str | None = None) -> str:
        """Cache key identifying one set of weights.

        "detect" keeps its fast/accurate split so both presets stay cached across
        a mode toggle (the original behaviour); every other task has exactly one
        checkpoint and so keys on the bare task name.
        """
        resolved_task = self._task if task is None else task
        if resolved_task != "detect":
            return resolved_task
        return f"detect:{self._mode if mode is None else mode}"

    def _model_name_for_mode(self, mode: str) -> str:
        return self._model_names.get(mode, self._model_names["fast"])

    def _model_name_for_preset(self, task: str | None = None) -> str:
        resolved = self._task if task is None else task
        if resolved == "detect":
            return self._model_name_for_mode(self._mode)
        return self._task_model_names.get(resolved, "")

    def models_by_mode(self) -> dict[str, str]:
        return {mode: self._model_name_for_mode(mode) for mode in DETECT_MODES}

    def models_by_task(self) -> dict[str, str]:
        """Every task's configured model, with "detect" showing the active preset."""
        resolved = {"detect": self._model_name_for_mode(self._mode)}
        resolved.update({task: self._task_model_names.get(task, "") for task in DETECT_TASKS
                         if task != "detect"})
        return resolved

    def set_mode(self, mode: Any) -> str:
        normalized = str(mode).strip().lower() if mode is not None else ""
        if normalized not in DETECT_MODES:
            raise ValueError(
                f"Detection mode must be one of {', '.join(DETECT_MODES)}, got {mode!r}"
            )
        self._mode = normalized
        return normalized

    def set_task(self, task: Any) -> str:
        """Switch to exactly one head. Weights load lazily on the next frame."""
        normalized = str(task).strip().lower() if task is not None else ""
        if normalized not in DETECT_TASKS:
            raise ValueError(
                f"Detection task must be one of {', '.join(DETECT_TASKS)}, got {task!r}"
            )
        self._tasks = (normalized,)
        self._task = normalized
        return normalized

    def set_tasks(self, tasks: Any) -> tuple[str, ...]:
        """Set every head that runs on a frame, in draw order.

        The first entry stays the "primary" task reported as ``task``: it owns
        the fast/accurate axis and is what a single-task client sees.
        """
        resolved = parse_detect_tasks(tasks, self._task)
        self._tasks = resolved
        self._task = resolved[0]
        return resolved

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
        if payload.get("task") is not None:
            self.set_task(payload["task"])
        if payload.get("tasks") is not None:
            self.set_tasks(payload["tasks"])
        if payload.get("task_models") is not None:
            self._set_task_models(payload["task_models"])
        if payload.get("conf_thresh") is not None:
            self._conf_thresh = self._validate_conf_thresh(payload["conf_thresh"])
        if payload.get("img_size") is not None:
            self._img_size = self._validate_img_size(payload["img_size"])
        if payload.get("end2end") is not None:
            self._set_end2end(payload["end2end"])
        if payload.get("max_det") is not None:
            self._max_det = self._validate_max_det(payload["max_det"])
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
        if payload.get("classifier_max_boxes") is not None:
            self._classifier_max_boxes = self._validate_classifier_max_boxes(
                payload["classifier_max_boxes"]
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
    def _validate_end2end(value: Any) -> str:
        normalized = str(value).strip().lower() if value is not None else ""
        if normalized not in END2END_MODES:
            raise ValueError(f"end2end must be one of {', '.join(END2END_MODES)}")
        return normalized

    @staticmethod
    def _validate_max_det(value: Any) -> int:
        try:
            limit = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_det must be an integer") from exc
        if not 1 <= limit <= 1000:
            raise ValueError("max_det must be between 1 and 1000")
        return limit

    @staticmethod
    def _validate_classifier_min_conf(value: Any) -> float:
        try:
            conf = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("classifier_min_conf must be a number") from exc
        if not 0.0 <= conf <= 1.0:
            raise ValueError("classifier_min_conf must be between 0.0 and 1.0")
        return conf

    @staticmethod
    def _validate_classifier_max_boxes(value: Any) -> int:
        try:
            limit = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("classifier_max_boxes must be an integer") from exc
        if not 1 <= limit <= 100:
            raise ValueError("classifier_max_boxes must be between 1 and 100")
        return limit

    def _set_end2end(self, mode: Any) -> None:
        """Switch the requested head, dropping cached weights only when needed.

        For a ``.pt`` this is a per-call predict kwarg, so the next frame just
        picks it up. An exported ``.engine``/``.onnx`` bakes the head in instead,
        and :meth:`_export_target` gives each head its own filename — so with
        export enabled the cache has to go, or the old artifact would keep serving
        the previous head while status claimed otherwise. Reloading is expensive
        (a first-time TensorRT export especially), which is why it is scoped to
        the case that actually needs it.
        """
        resolved = self._validate_end2end(mode)
        if resolved == self._end2end:
            return
        self._end2end = resolved
        if not self.settings.yolo_export:
            return
        self._models.clear()
        self._names_by_preset.clear()
        self._end2end_native.clear()
        self._open_vocab_applied.clear()
        self._loaded_sources.clear()
        self._tracker_models.clear()

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
        self._drop_preset(self._preset_key("detect", mode))

    def _set_task_model_name(self, task: str, name: Any) -> None:
        resolved = str(name).strip()
        if not resolved:
            raise ValueError(f"{task} model name must not be empty")
        if resolved == self._task_model_names.get(task):
            return
        self._task_model_names[task] = resolved
        self._drop_preset(self._preset_key(task))

    def _set_task_models(self, payload: Any) -> None:
        """Apply a partial ``{task: model_name}`` map. Unknown tasks are rejected."""
        if not isinstance(payload, dict):
            raise ValueError("task_models must be an object")
        for task, name in payload.items():
            if name is None:
                continue
            if task not in DETECT_TASKS or task == "detect":
                raise ValueError(
                    "task_models keys must be one of "
                    f"{', '.join(t for t in DETECT_TASKS if t != 'detect')}, got {task!r}"
                )
            self._set_task_model_name(task, name)

    def _drop_preset(self, preset: str) -> None:
        """Forget one preset's weights so the next frame reloads them."""
        self._models.pop(preset, None)
        self._names_by_preset.pop(preset, None)
        self._end2end_native.pop(preset, None)
        self._open_vocab_applied.pop(preset, None)
        self._loaded_sources.pop(preset, None)
        for key in [key for key in self._tracker_models if key[0] == preset]:
            self._tracker_models.pop(key, None)

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
        # Per-camera tracker instances share the configured vocabulary; they are
        # not part of `_names_by_preset` bookkeeping since the names are identical.
        for model in self._tracker_models.values():
            self._apply_open_vocabulary(model)
        active = self._preset_key()
        for preset, model in self._models.items():
            applied = self._apply_open_vocabulary(model)
            self._open_vocab_applied[preset] = applied
            if not applied:
                continue
            self._names_by_preset[preset] = self._resolve_names(model)
            if preset == active:
                self._names = self._names_by_preset[preset]

    @staticmethod
    def _resolve_names(model: Any) -> dict[int, str]:
        names = getattr(model, "names", {}) or {}
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        return {index: str(value) for index, value in enumerate(names)}

    def _end2end_capable(self) -> bool | None:
        """``None`` until the current preset's weights are loaded, then a bool."""
        preset = self._preset_key()
        if self._models.get(preset) is None:
            return None
        return self._end2end_native.get(preset, False)

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

        preset = self._preset_key()
        return {
            "model": self._model_name_for_preset(),
            "mode": self._mode,
            "available_modes": list(DETECT_MODES),
            "models": self.models_by_mode(),
            # The task axis. `emits_boxes` is what viewers and the docs key off:
            # false means zones / alerts / history see an empty frame by design.
            "task": self._task,
            "tasks": list(self._tasks),
            "available_tasks": list(DETECT_TASKS),
            "box_tasks": list(BOX_TASKS),
            "raster_tasks": list(RASTER_TASKS),
            "max_active_tasks": MAX_ACTIVE_TASKS,
            "task_models": self.models_by_task(),
            "emits_boxes": any(task in BOX_TASKS for task in self._tasks),
            "emits_raster": any(task in RASTER_TASKS for task in self._tasks),
            "configured_classes": list(self._classes),
            "open_vocabulary": self._open_vocab_applied.get(preset, False),
            "loaded": self.loaded,
            "export_format": self.settings.yolo_export,
            "loaded_source": self._loaded_sources.get(preset),
            "last_export_error": self._export_error,
            "device": self._device,
            "resolved_device": resolved_device,
            "requested_device": self.settings.yolo_device,
            # Effective for the active task, which is not always the requested
            # value: open-vocabulary forces FP32 (see _half_for_task).
            "half": self._half_for_task(),
            "requested_half": self.settings.yolo_half,
            "warmup_enabled": self.settings.yolo_warmup,
            "warmup_runs": self.settings.yolo_warmup_runs,
            "warmed_up": self._warmed_up,
            "warmup_ms": self._warmup_ms,
            "conf_thresh": self._conf_thresh,
            "img_size": self._img_size,
            # Requested mode vs. what the weights can actually do: an "on" here
            # with `end2end_capable` False means the model has no one-to-one head
            # and Ultralytics silently ignored the request.
            "end2end": self._end2end,
            "end2end_capable": self._end2end_capable(),
            "max_det": self._max_det,
            "track_enabled": self._track_enabled,
            "tracker": self._tracker,
            # Extra model instances held for non-primary cameras (see
            # `_ensure_tracker_model`); each one costs another copy of the
            # weights, so this is the number to watch when adding cameras.
            "tracker_streams": len(self._tracker_models),
            "classifier_model": self._classifier_name,
            "classifier_enabled": bool(self._classifier_name),
            "classifier_loaded": self._classifier_model is not None,
            "classifier_min_conf": self._classifier_min_conf,
            "classifier_max_boxes": self._classifier_max_boxes,
            "last_classifier_error": self._classifier_error,
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
            "cuda_device_name": cuda_device_name,
            "cuda_version": cuda_version,
            "torch_version": torch_version,
            "last_load_error": self._load_error,
            "last_warmup_error": self._last_warmup_error,
        }

    def detect(
        self, jpeg_bytes: bytes, frame_id: int, tracker_key: str | None = None
    ) -> dict[str, Any]:
        """Run detection on one frame.

        ``tracker_key`` selects which stream's tracker state to advance. ``None``
        (the single-camera path) uses the primary model exactly as before; any
        other key gets its own model instance so track ids stay per-stream.
        """
        decoded = self._decode_jpeg(jpeg_bytes)
        # Snapshot the running set: a runtime task switch racing the worker must
        # not make us read this frame's result with another head's extractor
        # (e.g. looking for `.obb` on a pose result).
        tasks = self._tasks
        payload: dict[str, Any] = {
            "frame_id": frame_id,
            "width": decoded.width,
            "height": decoded.height,
            "inference_ms": 0.0,
            "task": tasks[0],
            "tasks": list(tasks),
            "boxes": [],
        }

        # One pass per head, sequentially on this worker thread — the GPU is
        # serialized anyway, so the frame's cost is the sum and `inference_ms`
        # reports exactly that. `task_ms` breaks it down so the viewer can show
        # which head is expensive.
        boxes: list[dict[str, Any]] = []
        task_ms: dict[str, float] = {}
        total_ms = 0.0
        for task in tasks:
            model = self._ensure_model(task)
            names = self._names_by_preset.get(self._preset_key(task), {})
            if tracker_key is not None and self._track_enabled and task in BOX_TASKS:
                model = self._ensure_tracker_model(tracker_key, task)

            started = time.perf_counter()
            try:
                results = self._infer(model, decoded.data, task)
            except Exception as exc:
                raise DetectionError(f"YOLO inference failed for task {task!r}: {exc}") from exc
            elapsed = (time.perf_counter() - started) * 1000.0
            task_ms[task] = round(elapsed, 2)
            total_ms += elapsed

            result = results[0]
            if task in RASTER_TASKS:
                # No boxes at all for these heads — the raster *is* the result.
                # parse_detect_tasks admits at most one, so this never overwrites.
                raster = self._extract_raster(result, task, names)
                if raster is not None:
                    payload["raster"] = raster
                continue

            head_boxes = self._extract_boxes(result, decoded.width, decoded.height, names, task)
            # Track ids are only unique within one head's tracker, so tag each
            # box with its origin: two heads both reporting `track_id` 1 are
            # different objects, and the viewer keys its labels off this.
            for box in head_boxes:
                box["task"] = task
            boxes.extend(head_boxes)

        if boxes and self._classifier_name:
            self._classify_boxes(boxes, decoded.data, decoded.width, decoded.height)
        payload["boxes"] = boxes
        payload["inference_ms"] = round(total_ms, 2)
        if len(tasks) > 1:
            payload["task_ms"] = task_ms
        return payload

    def warmup(self) -> dict[str, Any]:
        if not self.settings.yolo_warmup:
            return {"enabled": False, "ok": True, "warmup_ms": 0.0}

        started = time.perf_counter()
        try:
            import numpy as np

            warmup_size = min(max(self._img_size, 32), 1280)
            source = np.zeros((warmup_size, warmup_size, 3), dtype=np.uint8)
            # Every active head, or the first frame still pays the lazy cost for
            # whichever ones warmup skipped.
            for task in self._tasks:
                model = self._ensure_model(task)
                for _ in range(self.settings.yolo_warmup_runs):
                    self._predict(model, source, task)
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
        for task in self._tasks:
            try:
                self._ensure_model(task)
            except Exception:  # noqa: BLE001 - reported via last_load_error
                pass
        # Warm the second-stage classifier too (no-op when disabled). It never
        # raises; load failures surface via `last_classifier_error` in status.
        self._ensure_classifier()
        return self.status()

    def _ensure_model(self, task: str | None = None) -> Any:
        preset = self._preset_key(task)
        cached = self._models.get(preset)
        if cached is not None:
            self._names = self._names_by_preset.get(preset, {})
            return cached

        with self._load_lock:
            # Re-check under the lock: a concurrent loader may have finished
            # while we were waiting, so we never build the same model twice.
            cached = self._models.get(preset)
            if cached is not None:
                self._names = self._names_by_preset.get(preset, {})
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

            model_name = self._model_name_for_preset(task)
            if not model_name:
                self._load_error = f"No model configured for task {task or self._task!r}"
                raise DetectionError(self._load_error)
            load_name = self._resolve_model_source(model_name)
            try:
                # YOLO() dispatches on the filename, so a "yoloe-*" name yields a
                # YOLOE instance and "-seg"/"-pose"/"-obb"/"-sem"/"-depth" pick the
                # right head — no explicit task argument needed.
                model = YOLO(load_name)
                # Sample before the first predict: an explicit end2end kwarg
                # overwrites the attribute we are reading here.
                native_end2end = model_supports_end2end(model)
                applied = self._apply_open_vocabulary(model)
                resolved_names = self._resolve_names(model)
            except Exception as exc:
                self._load_error = f"Could not load YOLO model {load_name!r}: {exc}"
                raise DetectionError(self._load_error) from exc

            self._models[preset] = model
            self._loaded_sources[preset] = load_name
            self._names_by_preset[preset] = resolved_names
            self._end2end_native[preset] = native_end2end
            self._open_vocab_applied[preset] = applied
            self._names = resolved_names
            self._load_error = None
            return model

    def _ensure_tracker_model(self, tracker_key: str, task: str | None = None) -> Any:
        """Lazily build the dedicated model instance backing one extra stream.

        Loads the same artifact the primary model resolved to, so an exported
        ``.engine``/``.onnx`` is reused rather than re-exported. Shares
        ``_load_lock`` with :meth:`_ensure_model` since both compete for the same
        weights on disk / device memory.
        """
        preset = self._preset_key(task)
        cache_key = (preset, tracker_key)
        cached = self._tracker_models.get(cache_key)
        if cached is not None:
            return cached

        with self._load_lock:
            cached = self._tracker_models.get(cache_key)
            if cached is not None:
                return cached

            try:
                from ultralytics import YOLO
            except Exception as exc:
                self._load_error = f"YOLO dependencies are not installed: {exc}"
                raise DetectionError(self._load_error) from exc

            load_name = self._loaded_sources.get(preset, self._model_name_for_preset(task))
            try:
                model = YOLO(load_name)
                self._apply_open_vocabulary(model)
            except Exception as exc:
                self._load_error = (
                    f"Could not load YOLO model {load_name!r} for stream {tracker_key!r}: {exc}"
                )
                raise DetectionError(self._load_error) from exc

            self._tracker_models[cache_key] = model
            self._load_error = None
            return model

    @staticmethod
    def _export_target(model_name: str, export_format: str, end2end: str = "auto") -> Path | None:
        """Pure: the artifact path an export would produce, or ``None`` to load as-is.

        ``None`` means no export is needed — either the feature is off, or the
        configured name is already an exported artifact (never re-export those, so
        pointing ``YOLO_MODEL`` straight at a hand-built ``.engine`` works too).

        An exported artifact bakes its detection head in at export time, and the
        ``end2end`` predict kwarg cannot override it afterwards. So a forced head
        gets its own filename: flipping ``YOLO_END2END`` triggers a fresh export
        instead of silently reusing a cache built with the other head. ``auto``
        keeps the plain ``<stem>.<ext>`` name, i.e. the original layout.
        """
        if export_format not in {"engine", "onnx"}:
            return None
        if not model_name.lower().endswith(".pt"):
            return None
        suffix = ".engine" if export_format == "engine" else ".onnx"
        source = Path(model_name)
        stem = source.stem if end2end == "auto" else f"{source.stem}-e2e{end2end}"
        return source.with_name(stem + suffix)

    def _resolve_model_source(self, model_name: str) -> str:
        """Resolve the artifact to load, auto-exporting a ``.pt`` when configured.

        A cached export on disk is reused; otherwise the ``.pt`` is exported once.
        Export needs the real toolchain (TensorRT engines require a GPU and bind to
        the device/version), so it is exercised at load time, not in CI — the pure
        path decision in :meth:`_export_target` is what the unit tests cover. A
        failed export is recorded and falls back to the original ``.pt`` so
        detection keeps working (unaccelerated) instead of going dark.
        """
        target = self._export_target(model_name, self.settings.yolo_export, self._end2end)
        if target is None:
            return model_name
        if target.exists():
            self._export_error = None
            return str(target)
        try:
            from ultralytics import YOLO

            export_kwargs: dict[str, Any] = {
                "format": self.settings.yolo_export,
                "half": self._half_enabled,
                "imgsz": self._img_size,
                "device": self._device,
            }
            # Same "auto" contract as _prediction_kwargs: omit the key entirely so
            # the checkpoint's own head is exported untouched.
            end2end = END2END_FLAGS[self._end2end]
            if end2end is not None:
                export_kwargs["end2end"] = end2end
            exported = YOLO(model_name).export(**export_kwargs)
            self._export_error = None
            return str(exported)
        except Exception as exc:  # noqa: BLE001 - surfaced via status; falls back to .pt
            self._export_error = (
                f"Export of {model_name!r} to {self.settings.yolo_export} failed: {exc}"
            )
            return model_name

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

    def _half_for_task(self, task: str | None = None) -> bool:
        """FP16 is off for open-vocabulary regardless of ``YOLO_HALF``.

        YOLOE fuses float32 text embeddings into its head, so a half-precision
        backbone raises "mat1 and mat2 must have the same dtype" on the first
        frame. Every other task keeps the configured setting.
        """
        if not self._half_enabled:
            return False
        return (self._task if task is None else task) != "openvocab"

    def _prediction_kwargs(self, source: Any, task: str | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "source": source,
            "imgsz": self._img_size,
            "conf": self._conf_thresh,
            "max_det": self._max_det,
            "verbose": False,
        }
        # "auto" deliberately omits the argument rather than passing a default:
        # `end2end` only exists from ultralytics 8.4, and an unknown predict key
        # is a hard error there, so omitting it is what keeps 8.3 working.
        end2end = END2END_FLAGS[self._end2end]
        if end2end is not None:
            kwargs["end2end"] = end2end
        if self._device is not None:
            kwargs["device"] = self._device
        if self._half_for_task(task):
            kwargs["half"] = True
        return kwargs

    def _inference_context(self) -> Any:
        try:
            import torch

            return torch.inference_mode()
        except Exception:
            return nullcontext()

    def _predict(self, model: Any, source: Any, task: str | None = None) -> Any:
        with self._inference_context():
            return model.predict(**self._prediction_kwargs(source, task))

    def _infer(self, model: Any, source: Any, task: str | None = None) -> Any:
        """Run detection for a live frame, tracking when enabled.

        ``model.track(persist=True)`` keeps the tracker state across calls so
        boxes get a stable ``track_id``; with tracking off it degrades to the
        stateless ``predict`` path. Warmup deliberately stays on ``_predict`` so
        it never seeds the tracker with blank frames.

        Semantic and depth results carry no boxes, and Ultralytics raises rather
        than tracking them, so those tasks always take the ``predict`` path.
        """
        resolved_task = self._task if task is None else task
        if not self._track_enabled or resolved_task not in BOX_TASKS:
            return self._predict(model, source, resolved_task)
        with self._inference_context():
            return model.track(
                persist=True,
                tracker=self._tracker,
                **self._prediction_kwargs(source, resolved_task),
            )

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

        # Throttle: classify only the N largest boxes (by area). On a crowded
        # frame the smaller boxes keep their detection label without a species,
        # which bounds the per-frame classifier cost.
        candidates = boxes
        if len(candidates) > self._classifier_max_boxes:
            candidates = sorted(boxes, key=self._box_area, reverse=True)[
                : self._classifier_max_boxes
            ]

        crops: list[Any] = []
        targets: list[dict[str, Any]] = []
        for box in candidates:
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
    def _box_area(box: dict[str, Any]) -> float:
        x1, y1, x2, y2 = box["xyxy"]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

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

    def _extract_raster(
        self, result: Any, task: str, names: dict[int, str]
    ) -> dict[str, Any] | None:
        """Encode a semantic class map or depth map as a small PNG for the viewer.

        These heads produce one value per pixel, which is far too much to ship raw
        every frame. The map is downscaled to ``raster_max_size`` on its long edge
        and PNG-encoded; the viewer stretches it back over the frame, so this is a
        deliberate crispness-for-bandwidth trade. Returns ``None`` (rather than
        raising) when the result carries no map, so a mid-switch frame degrades to
        "nothing to draw" instead of tearing down the worker.
        """
        try:
            import numpy as np
        except Exception:
            return None

        source = getattr(result, "semantic_mask" if task == "semantic" else "depth", None)
        if source is None:
            return None
        try:
            data = np.asarray(source.data.detach().cpu() if hasattr(source, "data") else source)
        except Exception:
            return None
        data = np.squeeze(data)
        if data.ndim != 2 or data.size == 0:
            return None

        if task == "depth":
            return self._encode_depth(data, np)
        return self._encode_semantic(data, names, np)

    def _encode_semantic(self, data: Any, names: dict[int, str], np: Any) -> dict[str, Any]:
        """Colourise a class-index map with a stable per-class palette."""
        classes = data.astype("int32")
        present = [int(value) for value in np.unique(classes)]
        # A fixed hash-based palette means a class keeps its colour across frames
        # and across restarts, which is what makes the overlay readable.
        palette = np.zeros((int(classes.max()) + 1, 3), dtype="uint8")
        for class_id in range(palette.shape[0]):
            palette[class_id] = _class_color(class_id)
        rgb = palette[np.clip(classes, 0, palette.shape[0] - 1)]
        return {
            "kind": "semantic",
            "png": self._png_data_url(rgb, np),
            "legend": [
                {"class_id": class_id, "label": names.get(class_id, str(class_id)),
                 "color": list(_class_color(class_id))}
                for class_id in present
            ],
        }

    def _encode_depth(self, data: Any, np: Any) -> dict[str, Any]:
        """Normalise metres to a 0-255 grayscale ramp, reporting the real range.

        The ramp is per-frame, so the image alone says nothing absolute — the
        min/max in metres travel with it so the viewer can label the scale.
        """
        depth = data.astype("float32")
        finite = np.isfinite(depth)
        if not bool(finite.any()):
            return {"kind": "depth", "png": "", "min_m": None, "max_m": None}
        valid = depth[finite]
        low = float(valid.min())
        high = float(valid.max())
        span = high - low
        normalized = (
            np.zeros_like(depth) if span <= 0 else np.clip((depth - low) / span, 0.0, 1.0)
        )
        # Near = bright reads more naturally as "close to the camera".
        gray = ((1.0 - normalized) * 255.0).astype("uint8")
        gray[~finite] = 0
        # Posterise to 32 levels before encoding. A smooth 8-bit gradient is
        # nearly incompressible (~17 KB/frame at 256px, i.e. 170 KB/s at 10 fps);
        # collapsing to 32 bands cuts that several-fold and is indistinguishable
        # in a translucent overlay. The real range travels as min_m/max_m anyway,
        # so nothing quantitative is lost here that the legend does not restore.
        gray = (gray >> 3) << 3
        return {
            "kind": "depth",
            "png": self._png_data_url(gray, np),
            "min_m": round(low, 3),
            "max_m": round(high, 3),
        }

    def _png_data_url(self, array: Any, np: Any) -> str:
        import base64

        from PIL import Image

        image = Image.fromarray(array)
        long_edge = max(image.size)
        limit = self.settings.raster_max_size
        if long_edge > limit:
            scale = limit / float(long_edge)
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                # NEAREST for class maps keeps invented in-between class ids out of
                # the palette lookup; depth is smooth enough that it costs nothing.
                Image.NEAREST,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def _extract_boxes(
        self,
        result: Any,
        width: int,
        height: int,
        names: dict[int, str] | None = None,
        task: str | None = None,
    ) -> list[dict[str, Any]]:
        # `names` is passed in rather than read off `self._names` so a config
        # swap racing the worker can never label boxes from the wrong vocabulary.
        resolved_names = self._names if names is None else names
        resolved_task = self._task if task is None else task
        if resolved_task == "obb":
            return self._extract_obb_boxes(result, width, height, resolved_names)

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
                    "label": resolved_names.get(class_id, str(class_id)),
                    "confidence": round(confidence, 4),
                    # None until the tracker confirms this box (or when tracking
                    # is off / the tracker hasn't assigned an id this frame).
                    "track_id": _box_track_id(box),
                }
            )
        # Segment and pose hang their extra geometry off the *same* result index
        # as the boxes, so these attach per-box fields rather than replacing any.
        self._attach_masks(extracted, result, width, height)
        self._attach_keypoints(extracted, result, width, height)
        return extracted

    def _extract_obb_boxes(
        self, result: Any, width: int, height: int, names: dict[int, str]
    ) -> list[dict[str, Any]]:
        """Build boxes from an OBB result.

        Rotated detections live on ``result.obb``, not ``result.boxes``. Each one
        still emits an axis-aligned ``xyxy`` (Ultralytics computes it for us) so
        zones, alerts, history and the recording sidecar keep working unchanged;
        the rotated quad rides alongside as ``obb`` for the overlay to draw.
        """
        extracted: list[dict[str, Any]] = []
        obb = getattr(result, "obb", None)
        if obb is None:
            return extracted

        try:
            quads = obb.xyxyxyxy.detach().cpu().tolist()
            aligned = obb.xyxy.detach().cpu().tolist()
            classes = obb.cls.detach().cpu().tolist()
            confidences = obb.conf.detach().cpu().tolist()
        except Exception:
            return extracted

        ids = getattr(obb, "id", None)
        track_ids: list[int | None]
        if ids is None:
            track_ids = [None] * len(aligned)
        else:
            try:
                track_ids = [int(value) for value in ids.detach().cpu().tolist()]
            except Exception:
                track_ids = [None] * len(aligned)

        for index, box_xyxy in enumerate(aligned):
            class_id = int(classes[index])
            # xyxyxyxy comes through as 4 [x, y] pairs; flatten to a plain list so
            # the payload stays JSON-friendly and the canvas can walk it directly.
            # Deliberately *not* clamped to the frame: a rotated box that runs off
            # the edge has corners outside it, and clamping them individually would
            # shear the rectangle into a different quadrilateral. The canvas clips
            # it for free. The axis-aligned `xyxy` above is clamped as always,
            # because that is what zones and alerts do arithmetic on.
            quad = [round(float(value), 1) for point in quads[index] for value in point]
            extracted.append(
                {
                    "xyxy": clamp_xyxy(box_xyxy, width, height),
                    "class_id": class_id,
                    "label": names.get(class_id, str(class_id)),
                    "confidence": round(float(confidences[index]), 4),
                    "track_id": track_ids[index] if index < len(track_ids) else None,
                    "obb": quad,
                }
            )
        return extracted

    def _attach_masks(
        self, boxes: list[dict[str, Any]], result: Any, width: int, height: int
    ) -> None:
        """Attach each instance-segmentation polygon to its box as ``mask``.

        ``masks.xy`` is already in source-image pixels, one contour per box in box
        order. Contours routinely run to hundreds of vertices, which would dwarf
        the rest of the payload at 10 fps, so each is subsampled to at most
        ``_MASK_MAX_POINTS`` and rounded — the overlay is a translucent shape, not
        a measurement, so the lost precision is invisible.
        """
        masks = getattr(result, "masks", None)
        if masks is None:
            return
        polygons = getattr(masks, "xy", None)
        if polygons is None:
            return

        for box, polygon in zip(boxes, polygons):
            simplified = self._simplify_polygon(polygon, width, height)
            if simplified:
                box["mask"] = simplified

    @staticmethod
    def _simplify_polygon(polygon: Any, width: int, height: int) -> list[list[float]]:
        try:
            points = [[float(x), float(y)] for x, y in polygon]
        except Exception:
            return []
        if len(points) < 3:
            return []
        if len(points) > _MASK_MAX_POINTS:
            # Even stride keeps the contour's shape; taking a prefix would clip it.
            stride = len(points) / _MASK_MAX_POINTS
            points = [points[min(len(points) - 1, int(i * stride))] for i in range(_MASK_MAX_POINTS)]
        return [
            [
                round(max(0.0, min(float(width), x)), 1),
                round(max(0.0, min(float(height), y)), 1),
            ]
            for x, y in points
        ]

    @staticmethod
    def _attach_keypoints(
        boxes: list[dict[str, Any]], result: Any, width: int, height: int
    ) -> None:
        """Attach pose keypoints to each box as ``[[x, y, conf], ...]``.

        ``keypoints.xy`` is in source pixels and ordered to match the boxes.
        Confidence is optional (``has_visible`` is False for some checkpoints), in
        which case every point is reported at 1.0 so the overlay still draws it.
        """
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None:
            return
        try:
            coordinates = keypoints.xy.detach().cpu().tolist()
        except Exception:
            return

        confidence_data = getattr(keypoints, "conf", None)
        try:
            confidences = (
                None if confidence_data is None else confidence_data.detach().cpu().tolist()
            )
        except Exception:
            confidences = None

        for index, box in enumerate(boxes):
            if index >= len(coordinates):
                break
            points = coordinates[index]
            scores = confidences[index] if confidences is not None and index < len(confidences) else None
            box["keypoints"] = [
                [
                    round(max(0.0, min(float(width), float(point[0]))), 1),
                    round(max(0.0, min(float(height), float(point[1]))), 1),
                    round(float(scores[joint]) if scores is not None else 1.0, 3),
                ]
                for joint, point in enumerate(points)
            ]
