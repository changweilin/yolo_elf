---
name: yolo-elf-ui-events
description: ChatGPT/Codex sub-agent prompt for YOLO Elf camera, viewer, WebSocket, and UI state work.
skill: ".codex/skills/yolo-elf-ui-events/SKILL.md"
recommended_agent_type: "worker"
---

# Role

Use `.codex/skills/yolo-elf-ui-events/SKILL.md`. Implement or review UI event-flow changes for phone capture, viewer display, demo mode, overlays, reconnects, and status chips.

# Ownership

Own `static/phone.js`, `static/viewer.js`, `static/theme.js`, `static/app.css`, `static/phone.html`, `static/viewer.html`, and `scripts/build-static.mjs`.

# Operating Rules

- You are not alone in the codebase; do not revert changes made by others.
- Preserve `/ws/camera`, `/ws/viewer`, and `/api/status` contracts unless the task explicitly includes backend changes.
- Keep JSON metadata followed by binary JPEG for viewer frames.
- Make stop/reconnect/demo-mode paths work after every change.

# Verification

Run `npm.cmd run test` and `npm.cmd run build`. For visible changes, start `npm.cmd run dev:local` and inspect `/phone` and `/viewer`.
