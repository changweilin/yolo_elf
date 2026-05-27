---
name: yolo-elf-spatial-metrics
description: Analyze and verify YOLO Elf coordinate geometry, bounding boxes, canvas scaling, frame dimensions, FPS, queue latency, inference timing, and benchmark math. Use when changing `clamp_xyxy`, overlay drawing, `fitContain`, detection extraction, stream metrics, adaptive FPS, or performance tuning.
---

# YOLO Elf Spatial Metrics

## Focus

Protect numerical correctness from model output to canvas overlay. Coordinate math crosses Python inference and JavaScript rendering, so verify both sides when boxes, dimensions, or performance metrics change.

## File Map

- Detection geometry: `app/detector.py`.
- Stream metrics: `app/stream_state.py`.
- Overlay rendering: `static/phone.js`, `static/viewer.js`.
- Benchmarks: `scripts/bench_detector.py`, `scripts/bench.ps1`.
- Tests: `tests/test_detector.py`, `tests/test_app.py`.

## Geometry Rules

- YOLO boxes use `xyxy`: `[x1, y1, x2, y2]` in source image pixels.
- `clamp_xyxy` must keep values inside `[0,width]` and `[0,height]`, swapping inverted endpoints.
- `fitContain(stageWidth, stageHeight, sourceWidth, sourceHeight)` must match CSS `object-fit: contain`.
- Canvas drawing must account for `devicePixelRatio` through `setTransform`.
- Label backgrounds should not depend on source image coordinates after scaling.

## Metrics Rules

- `inference_ms` measures detector prediction time, not queue wait.
- `last_queue_latency_ms` measures wait from frame receive to processing start.
- `last_total_latency_ms` measures receive-to-publish completion.
- Adaptive FPS should lower send rate when inference or socket buffering grows; it must not exceed user-requested FPS.

## Workflow

1. Write down the source coordinate space and destination coordinate space before editing.
2. Add targeted tests for boundary boxes, inverted coordinates, zero/empty detections, and latency averages.
3. For performance claims, use benchmark output instead of intuition.
4. Keep CPU-only verification available; GPU-specific improvements must degrade gracefully.

## Validation

Run `npm.cmd run test`. For performance-sensitive changes, also run `.\scripts\bench.ps1 -Frames 20 -Warmup 3 -Device cpu -ImgSize 960 -Quality 0.85` when dependencies are available.
