from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
ULTRALYTICS_DIR = ROOT_DIR / ".ultralytics"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_DIR))
ULTRALYTICS_DIR.mkdir(exist_ok=True)


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


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    yolo_model: str
    yolo_device: str
    yolo_half: bool
    yolo_warmup: bool
    yolo_warmup_runs: int
    conf_thresh: float
    img_size: int
    frame_fps: int
    capture_width: int
    capture_height: int
    jpeg_quality: float
    max_frame_bytes: int
    remote_storage_url: str
    remote_storage_token: str
    remote_storage_include_frame: bool
    remote_storage_queue_size: int
    remote_storage_timeout: float
    remote_storage_retries: int
    static_dir: Path


def get_settings() -> Settings:
    return Settings(
        host=os.getenv("HOST", "0.0.0.0"),
        port=_bounded_int_env("PORT", 8766, 1, 65535),
        yolo_model=os.getenv("YOLO_MODEL", "yolov8n.pt"),
        yolo_device=os.getenv("YOLO_DEVICE", "auto"),
        yolo_half=_bool_env("YOLO_HALF", False),
        yolo_warmup=_bool_env("YOLO_WARMUP", False),
        yolo_warmup_runs=_bounded_int_env("YOLO_WARMUP_RUNS", 1, 1, 10),
        conf_thresh=_bounded_float_env("CONF_THRESH", 0.35, 0.0, 1.0),
        img_size=_bounded_int_env("IMG_SIZE", 640, 32, 4096),
        frame_fps=_bounded_int_env("FRAME_FPS", 10, 1, 60),
        capture_width=_bounded_int_env("CAPTURE_WIDTH", 960, 64, 4096),
        capture_height=_bounded_int_env("CAPTURE_HEIGHT", 540, 64, 4096),
        jpeg_quality=_bounded_float_env("JPEG_QUALITY", 0.65, 0.3, 0.95),
        max_frame_bytes=_bounded_int_env(
            "MAX_FRAME_BYTES", 5 * 1024 * 1024, 64 * 1024, 50 * 1024 * 1024
        ),
        remote_storage_url=os.getenv("REMOTE_STORAGE_URL", "").strip(),
        remote_storage_token=os.getenv("REMOTE_STORAGE_TOKEN", "").strip(),
        remote_storage_include_frame=_bool_env("REMOTE_STORAGE_INCLUDE_FRAME", False),
        remote_storage_queue_size=_bounded_int_env("REMOTE_STORAGE_QUEUE_SIZE", 100, 1, 10000),
        remote_storage_timeout=_bounded_float_env("REMOTE_STORAGE_TIMEOUT", 5.0, 0.1, 60.0),
        remote_storage_retries=_bounded_int_env("REMOTE_STORAGE_RETRIES", 2, 0, 5),
        static_dir=ROOT_DIR / "static",
    )
