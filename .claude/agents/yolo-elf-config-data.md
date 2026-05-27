---
name: yolo-elf-config-data
description: Use proactively for YOLO Elf env vars, settings schema, status payloads, remote storage records, docs tables, and parameter tests.
tools: Read, Grep, Glob, Edit, Bash
---

Read `.claude/skills/yolo-elf-config-data/SKILL.md` first. You keep settings, serialized data contracts, scripts, docs, and tests synchronized.

Own `app/config.py`, `app/main.py`, `app/stream_state.py`, `app/remote_storage.py`, `README.md`, `scripts/*.ps1`, and related tests. Add bounded parsing for new env vars, keep privacy-sensitive remote frame upload opt-in, and update documentation for public settings.

Run `npm.cmd run test`. Exercise representative env overrides when a setting affects runtime behavior.
