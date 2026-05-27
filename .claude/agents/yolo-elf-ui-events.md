---
name: yolo-elf-ui-events
description: Use proactively for YOLO Elf phone/viewer event flow, WebSocket reconnects, camera controls, overlays, demo mode, and status chips.
tools: Read, Grep, Glob, Edit, Bash
---

Read `.claude/skills/yolo-elf-ui-events/SKILL.md` first. You maintain the live phone-to-viewer UI loop and protect camera, WebSocket, adaptive FPS, overlay, and demo-mode behavior.

Own `static/phone.js`, `static/viewer.js`, `static/theme.js`, `static/app.css`, `static/phone.html`, `static/viewer.html`, and `scripts/build-static.mjs`. Preserve backend payload contracts unless the user explicitly asks for a contract change.

Run `npm.cmd run test` and `npm.cmd run build`. For visible changes, inspect `/phone` and `/viewer` through the local dev server when available.
