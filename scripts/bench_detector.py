from __future__ import annotations

import argparse
import io
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.detector import DetectionError, YoloDetector


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((percent / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def make_jpeg(width: int, height: int, quality: float) -> bytes:
    image = Image.new("RGB", (width, height), (18, 22, 28))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (width * 0.15, height * 0.18, width * 0.72, height * 0.78),
        outline=(82, 190, 255),
        width=max(2, width // 160),
    )
    draw.ellipse(
        (width * 0.62, height * 0.24, width * 0.86, height * 0.58),
        outline=(72, 213, 151),
        width=max(2, width // 180),
    )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(max(1, min(95, quality * 100))))
    return buffer.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO Elf detector latency.")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--quality", type=float, default=0.65)
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--img-size", type=int, default=0)
    parser.add_argument("--conf", type=float, default=-1.0)
    parser.add_argument("--half", action="store_true")
    return parser.parse_args()


def apply_overrides(args: argparse.Namespace) -> Any:
    settings = get_settings()
    overrides: dict[str, Any] = {
        "capture_width": args.width,
        "capture_height": args.height,
        "jpeg_quality": args.quality,
        "yolo_half": args.half or settings.yolo_half,
    }
    if args.model:
        overrides["yolo_model"] = args.model
    if args.device:
        overrides["yolo_device"] = args.device
    if args.img_size > 0:
        overrides["img_size"] = args.img_size
    if args.conf >= 0.0:
        overrides["conf_thresh"] = args.conf
    return replace(settings, **overrides)


def main() -> int:
    args = parse_args()
    if args.frames <= 0:
        raise SystemExit("--frames must be greater than 0")
    if args.warmup < 0:
        raise SystemExit("--warmup must be 0 or greater")

    settings = apply_overrides(args)
    detector = YoloDetector(settings)
    frame = make_jpeg(settings.capture_width, settings.capture_height, settings.jpeg_quality)

    try:
        for index in range(args.warmup):
            detector.detect(frame, -(index + 1))

        latencies: list[float] = []
        boxes = 0
        started = time.perf_counter()
        for frame_id in range(1, args.frames + 1):
            frame_started = time.perf_counter()
            result = detector.detect(frame, frame_id)
            latencies.append((time.perf_counter() - frame_started) * 1000.0)
            boxes = len(result.get("boxes") or [])
        total_sec = time.perf_counter() - started
    except DetectionError as exc:
        print(f"Benchmark failed: {exc}")
        return 1

    status = detector.status()
    print("YOLO Elf benchmark")
    print(f"model: {status['model']}")
    print(f"device: {status['device']} half: {status['half']}")
    print(f"frames: {args.frames} warmup: {args.warmup} jpeg_bytes: {len(frame)} boxes_last: {boxes}")
    print(f"avg_ms: {statistics.mean(latencies):.2f}")
    print(f"p50_ms: {percentile(latencies, 50):.2f}")
    print(f"p95_ms: {percentile(latencies, 95):.2f}")
    print(f"min_ms: {min(latencies):.2f}")
    print(f"max_ms: {max(latencies):.2f}")
    print(f"fps: {args.frames / total_sec:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
