from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
ULTRALYTICS_DIR = ROOT_DIR / ".ultralytics"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_DIR))
ULTRALYTICS_DIR.mkdir(exist_ok=True)

# Detection presets the runtime can switch between. "fast" favours speed and
# uses `yolo_model`; "accurate" swaps in the larger `yolo_model_accurate`. This
# axis only applies to the "detect" task — the other heads have one model each.
DETECT_MODES = ("fast", "accurate")

# Switchable model heads. "detect" is the historical path and stays the default,
# so an unset YOLO_TASK behaves exactly like every previous release.
#
# "openvocab" is not an Ultralytics task — it is the YOLOE-26 segmentation model
# driven by YOLO_CLASSES text prompts. It gets its own entry because from the
# viewer's point of view it is another channel to switch to, and because it needs
# a different checkpoint family than plain "segment".
DETECT_TASKS = ("detect", "segment", "pose", "obb", "openvocab", "semantic", "depth")

# Tasks whose results carry boxes. Only these can be tracked (Ultralytics rejects
# `mode=track` for anything else) and only these feed zones / alerts / history —
# every one of those is defined in terms of a box with a `track_id`.
BOX_TASKS = ("detect", "segment", "pose", "obb", "openvocab")

# Tasks that emit one full-frame raster instead of boxes. They ride a separate
# payload field, never the box list, so the downstream box consumers simply see
# an empty frame rather than being fed something they cannot interpret.
RASTER_TASKS = ("semantic", "depth")

# Env var holding each task's model name. "detect" is absent: it keeps the
# original two-preset YOLO_MODEL / YOLO_MODEL_ACCURATE pair.
TASK_MODEL_ENV = {
    "segment": ("YOLO_MODEL_SEGMENT", "yolo26s-seg.pt"),
    "pose": ("YOLO_MODEL_POSE", "yolo26s-pose.pt"),
    "obb": ("YOLO_MODEL_OBB", "yolo26s-obb.pt"),
    # YOLOE-26 is a segmentation checkpoint; the "-pf" variants are prompt-free.
    "openvocab": ("YOLO_MODEL_OPENVOCAB", "yoloe-26s-seg.pt"),
    "semantic": ("YOLO_MODEL_SEMANTIC", "yolo26s-sem.pt"),
    "depth": ("YOLO_MODEL_DEPTH", "yolo26s-depth.pt"),
}

# NMS-free (end-to-end) head selection for models that ship one — YOLO26 and
# YOLOv10. "auto" leaves the checkpoint's own default alone, which is the only
# value that behaves identically on every model generation; "on"/"off" force the
# one-to-one / one-to-many head. Ignored by models without an end2end head.
END2END_MODES = ("auto", "on", "off")

# What each END2END_MODES entry means to Ultralytics' `end2end` predict arg:
# None = don't pass it at all (leave the checkpoint alone).
END2END_FLAGS: dict[str, bool | None] = {"auto": None, "on": True, "off": False}

# The implicit single-camera identity. With CAMERAS unset the registry holds
# exactly one channel under this id, so every camera_id-aware call site keeps
# behaving like the original single-stream server.
DEFAULT_CAMERA_ID = "default"
DEFAULT_CAMERA_NAME = "Camera"
CAMERA_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {raw!r}")
    return value


def _bounded_int_env(name: str, default: int, min_value: int, max_value: int) -> int:
    value = _int_env(name, default)
    if value < min_value or value > max_value:
        raise ValueError(f"{name} must be between {min_value} and {max_value}, got {value}")
    return value


def _bounded_float_env(name: str, default: float, min_value: float, max_value: float) -> float:
    value = _float_env(name, default)
    if value < min_value or value > max_value:
        raise ValueError(f"{name} must be between {min_value} and {max_value}, got {value}")
    return value


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _choice_env(name: str, default: str, choices: tuple[str, ...]) -> str:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}, got {raw!r}")
    return value


def _list_env(name: str) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def parse_cameras(raw: str, max_cameras: int) -> tuple[tuple[str, str], ...]:
    """Parse the ``CAMERAS`` allowlist into ``(camera_id, display_name)`` pairs.

    Format is a comma-separated list of ``id`` or ``id:顯示名`` entries, e.g.
    ``front:前門,back:後院``. An empty value yields the single implicit default
    camera, which is what keeps the single-stream behaviour untouched. The
    allowlist exists so a recorder cannot invent arbitrary ids (which would let
    a stray client spawn unbounded channels) and so the viewer grid has a stable
    layout even while a camera is offline.
    """
    entries = [item.strip() for item in (raw or "").split(",") if item.strip()]
    if not entries:
        return ((DEFAULT_CAMERA_ID, DEFAULT_CAMERA_NAME),)

    if len(entries) > max_cameras:
        raise ValueError(
            f"CAMERAS lists {len(entries)} cameras but MAX_CAMERAS is {max_cameras}"
        )

    cameras: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entry in entries:
        camera_id, _, display_name = entry.partition(":")
        camera_id = camera_id.strip()
        display_name = display_name.strip() or camera_id
        if not CAMERA_ID_PATTERN.match(camera_id):
            raise ValueError(
                f"CAMERAS id {camera_id!r} must be 1-32 chars of letters, digits, '-' or '_'"
            )
        if camera_id in seen:
            raise ValueError(f"CAMERAS contains duplicate id {camera_id!r}")
        seen.add(camera_id)
        cameras.append((camera_id, display_name))
    return tuple(cameras)


def _path_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    detect_mode: str
    detect_task: str
    yolo_model: str
    yolo_model_accurate: str
    # One model name per non-detect task, as (task, model) pairs. A tuple rather
    # than a dict so `replace(settings, ...)` copies stay genuinely independent —
    # a shared mutable dict would alias across every copy.
    task_models: tuple[tuple[str, str], ...]
    # Upper bound on the long edge of an encoded semantic/depth raster.
    raster_max_size: int
    yolo_classes: tuple[str, ...]
    yolo_device: str
    yolo_half: bool
    yolo_track: bool
    yolo_tracker: str
    yolo_end2end: str
    yolo_max_det: int
    yolo_export: str
    yolo_warmup: bool
    yolo_warmup_runs: int
    # Optional additive VLM (Florence-2) channel. Runs alongside — never replaces
    # — the YOLO pipeline: a periodic worker samples each camera's latest frame
    # and produces open-vocabulary boxes (Phase 1) and a scene caption (Phase 2).
    # Device / half / prompt classes are shared with the YOLO detector.
    vlm_enabled: bool
    vlm_model: str
    vlm_interval_sec: float
    vlm_detect_task: str
    vlm_caption: bool
    vlm_caption_task: str
    conf_thresh: float
    img_size: int
    classifier_model: str
    classifier_min_conf: float
    classifier_max_boxes: int
    frame_fps: int
    capture_width: int
    capture_height: int
    jpeg_quality: float
    max_frame_bytes: int
    recording_enabled: bool
    recording_keep_local_copy: bool
    recording_storage_dir: Path
    recording_max_bytes: int
    remote_storage_url: str
    remote_storage_token: str
    remote_storage_include_frame: bool
    remote_storage_recording_url: str
    remote_storage_queue_size: int
    remote_storage_timeout: float
    remote_storage_retries: int
    alert_rules_json: str
    alert_webhook_url: str
    alert_webhook_token: str
    alert_cooldown_sec: float
    alert_webhook_timeout: float
    alert_webhook_retries: int
    zones_json: str
    event_log_enabled: bool
    event_db_path: Path
    event_expiry_sec: float
    metrics_enabled: bool
    auth_token: str
    auth_session_ttl: int
    cameras: tuple[tuple[str, str], ...]
    max_cameras: int
    static_dir: Path

    @property
    def task_model_map(self) -> dict[str, str]:
        return dict(self.task_models)

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(camera_id for camera_id, _ in self.cameras)

    @property
    def default_camera_id(self) -> str:
        """The camera every camera_id-less call site implicitly addresses."""
        return self.cameras[0][0]

    @property
    def multi_camera(self) -> bool:
        return len(self.cameras) > 1


def get_settings() -> Settings:
    max_cameras = _bounded_int_env("MAX_CAMERAS", 4, 1, 16)
    return Settings(
        host=os.getenv("HOST", "0.0.0.0"),
        port=_bounded_int_env("PORT", 8766, 1, 65535),
        detect_mode=_choice_env("DETECT_MODE", "fast", DETECT_MODES),
        # Which head runs. "detect" keeps the original box-only pipeline; the
        # other entries swap in a task-specific checkpoint and add their own
        # payload fields. Switchable at runtime from the viewer's icon row.
        detect_task=_choice_env("DETECT_TASK", "detect", DETECT_TASKS),
        yolo_model=os.getenv("YOLO_MODEL", "yolo26s.pt"),
        yolo_model_accurate=os.getenv("YOLO_MODEL_ACCURATE", "yolo26x.pt"),
        task_models=tuple(
            (task, os.getenv(env_name, default).strip() or default)
            for task, (env_name, default) in TASK_MODEL_ENV.items()
        ),
        # Semantic/depth rasters are downscaled to this long edge before being
        # PNG-encoded onto the payload. 256 keeps a frame in the low single-digit
        # KB; the overlay is stretched back over the frame client-side, so this
        # trades overlay crispness against bandwidth on every single frame.
        raster_max_size=_bounded_int_env("RASTER_MAX_SIZE", 256, 32, 1024),
        yolo_classes=_list_env("YOLO_CLASSES"),
        yolo_device=os.getenv("YOLO_DEVICE", "auto"),
        yolo_half=_bool_env("YOLO_HALF", True),
        # Multi-object tracking: assign a stable `track_id` to each box across
        # frames via Ultralytics' built-in tracker. Off falls back to stateless
        # per-frame detection. YOLO_TRACKER picks the tracker config; ultralytics
        # 8.4 ships six (bytetrack / botsort / tracktrack / fasttrack / ocsort /
        # deepocsort). Left free-form so a custom tracker .yaml still resolves.
        yolo_track=_bool_env("YOLO_TRACK", True),
        yolo_tracker=os.getenv("YOLO_TRACKER", "bytetrack.yaml").strip() or "bytetrack.yaml",
        # YOLO26 / YOLOv10 ship a one-to-one head that emits already-deduplicated
        # boxes, so no NMS runs and the `iou` threshold stops mattering. "auto"
        # keeps whatever the checkpoint was built with — the only setting that is
        # bit-for-bit the old behaviour on a YOLOv8/v11 model, which has no such
        # head and ignores the argument entirely.
        yolo_end2end=_choice_env("YOLO_END2END", "auto", END2END_MODES),
        # Hard cap on boxes per frame. 300 mirrors the Ultralytics default, and is
        # also the floor the end2end head is pinned to internally, so lowering it
        # only truncates the (already sorted) output — it never speeds up the head.
        yolo_max_det=_bounded_int_env("YOLO_MAX_DET", 300, 1, 1000),
        # Inference acceleration. Empty = load the model name as-is (a `.pt`, or a
        # pre-exported `.engine`/`.onnx` you point YOLO_MODEL at). "engine"/"onnx"
        # auto-export a `.pt` on first load and load the product instead. The
        # export artifact is cached on disk; a failed export falls back to the .pt.
        yolo_export=_choice_env("YOLO_EXPORT", "", ("", "engine", "onnx")),
        yolo_warmup=_bool_env("YOLO_WARMUP", False),
        yolo_warmup_runs=_bounded_int_env("YOLO_WARMUP_RUNS", 1, 1, 10),
        # Additive VLM channel (off by default). When on, a Florence-2 engine runs
        # a slow periodic pass in parallel with YOLO. It carries no confidence and
        # no tracking (Florence emits neither), so it never feeds zones/alerts/
        # history — it is a separate viewer channel. Device/half come from
        # YOLO_DEVICE/YOLO_HALF; the open-vocab prompt comes from YOLO_CLASSES.
        # VLM_DETECT_TASK overrides the Florence task token; empty auto-selects
        # <OPEN_VOCABULARY_DETECTION> when YOLO_CLASSES is set, else <OD>.
        vlm_enabled=_bool_env("VLM_ENABLED", False),
        vlm_model=os.getenv("VLM_MODEL", "microsoft/Florence-2-base").strip()
        or "microsoft/Florence-2-base",
        vlm_interval_sec=_bounded_float_env("VLM_INTERVAL_SEC", 3.0, 0.5, 120.0),
        vlm_detect_task=os.getenv("VLM_DETECT_TASK", "").strip(),
        # Phase 2 scene caption. When on, each VLM tick also runs a caption task
        # and ships the text on the same `vlm` payload for the viewer HUD.
        vlm_caption=_bool_env("VLM_CAPTION", True),
        vlm_caption_task=os.getenv("VLM_CAPTION_TASK", "").strip() or "<MORE_DETAILED_CAPTION>",
        conf_thresh=_bounded_float_env("CONF_THRESH", 0.2, 0.0, 1.0),
        img_size=_bounded_int_env("IMG_SIZE", 1280, 32, 4096),
        # Optional second-stage classifier (e.g. yolov8x-cls.pt, ImageNet 1000)
        # that names the species inside each detection box. Empty = disabled.
        classifier_model=os.getenv("CLASSIFIER_MODEL", "").strip(),
        classifier_min_conf=_bounded_float_env("CLASSIFIER_MIN_CONF", 0.0, 0.0, 1.0),
        # Throttle: classify only the N largest detection boxes per frame so a
        # crowded frame can't stall the pipeline. Boxes outside the top-N keep
        # their detection label without a species.
        classifier_max_boxes=_bounded_int_env("CLASSIFIER_MAX_BOXES", 5, 1, 100),
        frame_fps=_bounded_int_env("FRAME_FPS", 10, 1, 60),
        capture_width=_bounded_int_env("CAPTURE_WIDTH", 1920, 64, 4096),
        capture_height=_bounded_int_env("CAPTURE_HEIGHT", 1080, 64, 4096),
        jpeg_quality=_bounded_float_env("JPEG_QUALITY", 0.9, 0.3, 0.95),
        max_frame_bytes=_bounded_int_env(
            "MAX_FRAME_BYTES", 5 * 1024 * 1024, 64 * 1024, 50 * 1024 * 1024
        ),
        recording_enabled=_bool_env("RECORDING_ENABLED", True),
        recording_keep_local_copy=_bool_env("RECORDING_KEEP_LOCAL_COPY", True),
        recording_storage_dir=_path_env("RECORDING_STORAGE_DIR", ROOT_DIR / "recordings"),
        recording_max_bytes=_bounded_int_env(
            "RECORDING_MAX_BYTES",
            250 * 1024 * 1024,
            1 * 1024 * 1024,
            2 * 1024 * 1024 * 1024,
        ),
        remote_storage_url=os.getenv("REMOTE_STORAGE_URL", "").strip(),
        remote_storage_token=os.getenv("REMOTE_STORAGE_TOKEN", "").strip(),
        remote_storage_include_frame=_bool_env("REMOTE_STORAGE_INCLUDE_FRAME", False),
        remote_storage_recording_url=os.getenv("REMOTE_STORAGE_RECORDING_URL", "").strip(),
        remote_storage_queue_size=_bounded_int_env("REMOTE_STORAGE_QUEUE_SIZE", 100, 1, 10000),
        remote_storage_timeout=_bounded_float_env("REMOTE_STORAGE_TIMEOUT", 5.0, 0.1, 60.0),
        remote_storage_retries=_bounded_int_env("REMOTE_STORAGE_RETRIES", 2, 0, 5),
        # Alert rules (JSON array) that fire when a detection matches; parsed and
        # validated by app/alerts.py. Empty = alerts off. The optional webhook is
        # a dedicated outbound channel (kept separate from remote storage, and
        # env-only so runtime rule edits can't repoint it — SSRF guard).
        alert_rules_json=os.getenv("ALERT_RULES", "").strip(),
        alert_webhook_url=os.getenv("ALERT_WEBHOOK_URL", "").strip(),
        alert_webhook_token=os.getenv("ALERT_WEBHOOK_TOKEN", "").strip(),
        alert_cooldown_sec=_bounded_float_env("ALERT_COOLDOWN_SEC", 15.0, 0.0, 3600.0),
        alert_webhook_timeout=_bounded_float_env("ALERT_WEBHOOK_TIMEOUT", 5.0, 0.1, 60.0),
        alert_webhook_retries=_bounded_int_env("ALERT_WEBHOOK_RETRIES", 2, 0, 5),
        # ROI polygons (JSON array) in normalized 0..1 frame coordinates; parsed
        # by app/zones.py. Boxes get a `zones` label and alert rules can be
        # scoped to a zone. Empty = zones off. Editable via POST /api/zones.
        zones_json=os.getenv("ZONES", "").strip(),
        # Detection event history: aggregate tracked objects into per-sighting
        # rows (first/last seen, dwell, zones) and persist to a local SQLite file
        # for the /history timeline. A sighting is finalized once its track has
        # been unseen for EVENT_EXPIRY_SEC.
        event_log_enabled=_bool_env("EVENT_LOG_ENABLED", True),
        event_db_path=_path_env("EVENT_DB_PATH", ROOT_DIR / "events.db"),
        event_expiry_sec=_bounded_float_env("EVENT_EXPIRY_SEC", 5.0, 0.5, 3600.0),
        # Prometheus /metrics endpoint. On by default; set 0 to return 404 there.
        metrics_enabled=_bool_env("METRICS_ENABLED", True),
        # Optional access control. Empty AUTH_TOKEN = auth disabled (current
        # behaviour). When set, pages/REST/WS require a signed session cookie
        # (issued after the token is entered on /login) or a Bearer token. The
        # cookie-signing key is derived from AUTH_TOKEN, so rotating the token
        # invalidates every existing session.
        auth_token=os.getenv("AUTH_TOKEN", "").strip(),
        auth_session_ttl=_bounded_int_env(
            "AUTH_SESSION_TTL", 7 * 24 * 3600, 60, 30 * 24 * 3600
        ),
        # Multi-camera allowlist. Empty = one implicit `default` channel, i.e.
        # the original single-recorder behaviour. Detection parameters stay
        # global; zones / alerts / history carry a camera_id because those are
        # tied to a physical location and would be meaningless if merged.
        cameras=parse_cameras(os.getenv("CAMERAS", ""), max_cameras),
        max_cameras=max_cameras,
        static_dir=ROOT_DIR / "static",
    )
