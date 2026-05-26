from __future__ import annotations

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
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    yolo_model: str
    yolo_device: str
    conf_thresh: float
    img_size: int
    frame_fps: int
    capture_width: int
    capture_height: int
    jpeg_quality: float
    max_frame_bytes: int
    static_dir: Path


def get_settings() -> Settings:
    return Settings(
        host=os.getenv("HOST", "0.0.0.0"),
        port=_int_env("PORT", 8766),
        yolo_model=os.getenv("YOLO_MODEL", "yolov8n.pt"),
        yolo_device=os.getenv("YOLO_DEVICE", "auto"),
        conf_thresh=_float_env("CONF_THRESH", 0.35),
        img_size=_int_env("IMG_SIZE", 640),
        frame_fps=_int_env("FRAME_FPS", 10),
        capture_width=_int_env("CAPTURE_WIDTH", 960),
        capture_height=_int_env("CAPTURE_HEIGHT", 540),
        jpeg_quality=_float_env("JPEG_QUALITY", 0.65),
        max_frame_bytes=_int_env("MAX_FRAME_BYTES", 5 * 1024 * 1024),
        static_dir=ROOT_DIR / "static",
    )
