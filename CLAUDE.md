# CLAUDE.md — YOLO Elf

> Macro principles. Per-role detail lives in `.claude/skills/yolo-elf-*/SKILL.md`
> (division of labour: `docs/ai-agent-matrix.md`). This file wins on conflict.
> Written in English for agents; product copy is not — see rule 15.

## What this is

FastAPI + WebSocket + framework-free static frontend, real-time detection. The
phone (recorder) pushes JPEG frames to `/ws/camera`; a **single** GPU worker runs
YOLO inference and broadcasts frames + boxes to every `/ws/viewer`. Inference is
fully local; frames do not leave the host by default.

Feature work is complete — tracking, alerts, ROI zones, SQLite history, auth,
metrics, ONNX/TensorRT export, multi-camera, VLM channel, recordings, and the
YOLO26 multi-head task channels (`detect`/`segment`/`pose`/`obb`/`openvocab`/
`semantic`/`depth`, up to `MAX_ACTIVE_TASKS`, one raster head at a time). What
remains is maintenance and incremental improvement.

## Module map

- `app/` — one module per concern: `config` (settings), `main` (routes + worker),
  `stream_state` (per-camera channels, frame queue), `detector` (inference),
  `zones` / `alerts` / `events` (regions, rules, SQLite history), plus `auth`,
  `metrics`, `recordings`, `remote_storage`, `vlm`.
- `static/` — `phone` (recorder), `viewer` (display only), `history` / `settings`
  / `login` pages, shared `app.css` / `theme.js` / `mode-switch.js`.
- `scripts/` — PowerShell run/test/bench scripts + Node static build
  (`build-static.mjs` produces the GitHub Pages demo).
- `tests/` — `test_<feature>.py` unit tests + `test_app.py` over a real ASGI
  lifespan.
- Docs — `README.md` (features, settings table, API table, structure),
  `TUNING.md` (operations and tuning).

## Core principles

**Architecture**
1. A new feature is its own `app/<feature>.py`: pure functions + an
   Engine/Registry class that fail-loud validates in `__init__(settings)`.
2. GPU access is serialized by a **single** detection worker — never spawn a
   thread per stream or per request to contend for the GPU. Low latency comes
   from a single-slot queue that keeps only the newest frame and drops the rest,
   not from queueing.
3. Back every performance claim with `scripts/bench_detector.py`, never
   intuition. GPU-only optimizations must degrade gracefully on CPU.

**Configuration**
4. All settings go through the existing `app/config.py` helpers (`_bool_env`,
   `_bounded_int_env`, `_bounded_float_env`, `_list_env`, `_choice_env`) and land
   as `Settings` fields. Numbers must be bounded; errors must be specific.
5. Defaults are the safest option and the closest to prior behavior. Anything
   privacy-sensitive (remote upload, frames leaving the host) is opt-in.

**Privacy and security**
6. "Frames stay on the host" is a product promise: every path that ships a frame
   off-box must be explicitly enabled.
7. Outbound endpoints (webhook, remote storage URL) are **env-var only** — never
   settable through a runtime API (SSRF guard). Tokens never go in URLs or logs.

**Compatibility**
8. Add a dimension, don't swap behavior: new parameters are optional and absent
   means the old path, bit-for-bit (e.g. empty `CAMERAS` = single camera, unset
   `DETECT_TASKS` = the previous single-head pipeline).
9. Evolve the SQLite schema with `ALTER TABLE ADD COLUMN`; tolerate NULL in old
   rows.
10. External contracts are frozen: WebSocket `payload.type` values, the
    JSON-then-binary frame order, DOM ids, routes, env var names. Renaming
    breaks them — add rather than rename.

**Geometry and numbers**
11. Boxes are `xyxy` in **source-image pixels**. Crossing the Python/JS boundary,
    write down the source and target coordinate space before touching anything.
    `clamp_xyxy` clips, `fitContain` mirrors CSS `object-fit: contain`, canvas is
    corrected by `devicePixelRatio`; zones use 0–1 normalized coordinates.

**Testing**
12. Every PR: `npm test` green. It runs `scripts/check.mjs` — ruff, pytest,
    py_compile, and `node --check` over every `static/*.js` and `scripts/*.mjs`
    — the same set CI runs on Windows and Linux. Exercise UI changes in a real
    browser: headless tabs pause rAF, so canvas geometry needs manual checking.
13. App-level tests run a real ASGI lifespan. Test isolation is central:
    `tests/conftest.py` clears every settings env var before each test, and
    `test_settings_env_covers_every_setting` fails when `app/config.py` grows a
    variable that list is missing. Add new settings there, not to a local list.
    Shared payload builders live in `tests/helpers.py`.
14. The suite fakes `ultralytics` and never imports torch, so CI installs only
    `requirements-ci.txt`. A green run proves nothing about the installed
    Ultralytics version — for that, benchmark against the repo-root `.venv`.

**Docs and demo**
15. Shipping a feature means code + tests + README (settings/API tables) +
    TUNING (how to operate it) + a matching static-demo surface, in one pass.
    Doc tables must match `app/config.py` exactly.
16. Product copy is zh-TW first, README bilingual, commit messages zh-TW. Keep UI
    labels short so the mobile topbar does not overflow.

## Commands

| Purpose | Command |
| --- | --- |
| All checks | `npm test` (Windows: `npm.cmd run test`) |
| Lint only | `npm run lint` |
| Static demo build | `npm run build` |
| Local dev | `npm run dev:local` → `/phone`, `/viewer` |
| Benchmark | `.\scripts\bench.ps1 -Frames 20 -Warmup 3 -Device cpu -ImgSize 960 -Quality 0.85` |

CI (`.github/workflows/ci.yml`) runs the checks and the static build on both
windows-latest and ubuntu-latest, resolves the full `requirements.txt` without
installing it, and deploys GitHub Pages from `main`.
