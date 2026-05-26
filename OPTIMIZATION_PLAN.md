# YOLO Elf Optimization Progress

| Priority | Work item | Status | Acceptance check |
| --- | --- | --- | --- |
| P0 | Add a durable optimization progress table | Done | This file exists and is updated as work lands |
| P0 | Add JavaScript syntax smoke tests | Done | `scripts/run-tests.ps1` checks browser/client scripts |
| P0 | Validate runtime settings ranges | Done | Invalid env values fail fast with clear messages |
| P0 | Expand `/api/status` stream metrics | Done | Status includes FPS, queue depth, frame size, and latency |
| P1 | Keep README ports and operations docs aligned | Done | README examples use the active default port |
| P1 | Reduce viewer frame transport overhead | Done | Base64 work is avoided when no viewer is connected |
| P2 | Add detector warmup and precision controls | Done | Startup/warmup behavior and precision are configurable |
| P2 | Add a repeatable benchmark script | Done | A script reports average and tail latency for sample frames |

## Current Execution Batch

Started: 2026-05-26

1. Stabilize test coverage for Python plus static JavaScript.
2. Add configuration guardrails for common tuning values.
3. Expose stream throughput and latency counters in `/api/status`.
4. Update the README examples to match the current `8766` default.

Verification: `npm.cmd run test` passes with 15 tests. Pytest reports a cache-write permission warning for `.pytest_cache`, but the test suite succeeds.

## Second Execution Batch

Completed: 2026-05-26

1. Added `YOLO_HALF`, `YOLO_WARMUP`, and `YOLO_WARMUP_RUNS` settings.
2. Added detector status fields for half precision and warmup state.
3. Added `scripts/bench.ps1` and `scripts/bench_detector.py` for repeatable latency checks.
4. Added tests for boolean/range config parsing and half-precision device selection.

Verification: `npm.cmd run test` passes with 19 tests. A minimal benchmark command also runs successfully on CPU:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bench.ps1 -Frames 1 -Warmup 0 -Width 64 -Height 64 -ImgSize 32 -Device cpu
```
