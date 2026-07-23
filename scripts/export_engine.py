"""Pre-export a YOLO .pt model to a TensorRT .engine or ONNX .onnx artifact.

The server can auto-export on first load (`YOLO_EXPORT=engine|onnx`), but doing it
here up front avoids the one-time export cost on startup and lets you verify the
toolchain separately. TensorRT engines require a CUDA GPU and bind to the exact
device + TensorRT/CUDA versions, so export them on the machine that will serve.

Examples (from the repo root, in the project venv):
    python scripts/export_engine.py yolov8s.pt --format engine --half
    python scripts/export_engine.py yolov8x.pt --format onnx --imgsz 1280

Then point the server at the product:
    $env:YOLO_MODEL = "yolov8s.engine"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Source .pt weights (e.g. yolov8s.pt)")
    parser.add_argument(
        "--format",
        choices=("engine", "onnx"),
        default="engine",
        help="Export format (default: engine / TensorRT)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280, help="Inference image size (default: 1280)"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export FP16 (GPU only; ignored on CPU by Ultralytics)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Export device, e.g. 0 or cuda:0 (default: let Ultralytics choose)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.model.lower().endswith(".pt"):
        print(f"Refusing to export a non-.pt source: {args.model!r}", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except Exception as exc:  # noqa: BLE001 - surface a clear setup hint
        print(f"Ultralytics is not available: {exc}", file=sys.stderr)
        return 1

    export_kwargs = {"format": args.format, "imgsz": args.imgsz, "half": args.half}
    if args.device is not None:
        export_kwargs["device"] = args.device

    print(f"Exporting {args.model} -> {args.format} (imgsz={args.imgsz}, half={args.half})…")
    try:
        produced = YOLO(args.model).export(**export_kwargs)
    except Exception as exc:  # noqa: BLE001 - the toolchain error is the useful output
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {produced}")
    print(f"Point the server at it, e.g.  $env:YOLO_MODEL = \"{Path(produced).name}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
