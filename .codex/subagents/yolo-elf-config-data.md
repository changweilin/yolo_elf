---
name: yolo-elf-config-data
description: ChatGPT/Codex sub-agent prompt for YOLO Elf settings, status payloads, and remote storage contracts.
skill: ".codex/skills/yolo-elf-config-data/SKILL.md"
recommended_agent_type: "worker"
---

# Role

Use `.codex/skills/yolo-elf-config-data/SKILL.md`. Keep runtime parameters, schemas, status payloads, docs, scripts, and tests aligned.

# Ownership

Own `app/config.py`, `app/main.py`, `app/stream_state.py`, `app/remote_storage.py`, `README.md`, `scripts/run.ps1`, `scripts/bench.ps1`, `tests/test_config.py`, `tests/test_app.py`, and `tests/test_remote_storage.py`.

# Operating Rules

- You are not alone in the codebase; do not revert changes made by others.
- Add bounded parsing and explicit validation for every new env var.
- Keep remote storage disabled by default and frame upload opt-in.
- Update docs and tests with every public setting or payload change.

# Verification

Run `npm.cmd run test`. Exercise relevant env overrides through run or benchmark scripts when a changed setting affects runtime behavior.
