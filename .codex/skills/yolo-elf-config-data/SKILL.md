---
name: yolo-elf-config-data
description: Manage YOLO Elf runtime parameters, settings schema, status payloads, remote storage contracts, and future preset/profile data. Use when adding or changing environment variables, `/api/status` fields, remote upload payloads, benchmark parameters, README configuration tables, or tests for configuration validation.
---

# YOLO Elf Config Data

## Focus

Keep configuration, runtime status, and remote-storage data contracts synchronized across backend code, frontend consumers, docs, scripts, and tests. The project currently has no database; treat "data management" as settings/schema management unless a real persistence layer is added.

## File Map

- Settings source of truth: `app/config.py`.
- Runtime status and WebSocket contracts: `app/main.py`, `app/stream_state.py`.
- Remote data upload: `app/remote_storage.py`.
- Consumers: `static/phone.js`, `static/viewer.js`.
- Docs/scripts: `README.md`, `TUNING.md`, `scripts/run.ps1`, `scripts/bench.ps1`.
- Tests: `tests/test_config.py`, `tests/test_app.py`, `tests/test_remote_storage.py`.

## Workflow

1. Start from `Settings` in `app/config.py`; add bounded parsing and clear errors for every new env var.
2. Propagate new fields to status payloads, WebSocket config messages, remote payloads, docs, and scripts only when those surfaces need them.
3. Preserve safe defaults. Do not enable remote uploads or frame inclusion by default.
4. Add or update tests for invalid values, defaults, and serialized payload shape.
5. If introducing profiles/presets, define a schema and migration path before adding storage.

## Contract Rules

- Environment variable names use uppercase snake case.
- Numeric runtime inputs must be bounded with `_bounded_int_env` or `_bounded_float_env`.
- Boolean env parsing must use `_bool_env`.
- Remote frame bytes are privacy-sensitive; keep `REMOTE_STORAGE_INCLUDE_FRAME=0` as the default.
- Frontend code should tolerate missing status fields with nullish fallback.

## Validation

Run `npm.cmd run test`. For parameter behavior, test representative env overrides through `scripts/run.ps1` or `scripts/bench.ps1` when the changed setting affects runtime capture/detection.
