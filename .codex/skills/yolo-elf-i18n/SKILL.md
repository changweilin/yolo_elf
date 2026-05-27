---
name: yolo-elf-i18n
description: Translate, localize, and repair multilingual copy for the YOLO Elf FastAPI/WebSocket detector app. Use when editing UI strings in static HTML/JS, README/TUNING/optimization docs, zh-TW/en copy, camera permission messaging, demo-mode wording, or labels that must stay consistent with app behavior.
---

# YOLO Elf I18N

## Focus

Localize YOLO Elf without changing runtime behavior. Treat text as product UI, not literal translation: camera, viewer, model status, privacy, WebSocket, and detection wording must stay short enough for the existing HUD/chip layout.

## File Map

- UI copy: `static/phone.html`, `static/viewer.html`, `static/phone.js`, `static/viewer.js`, `static/theme.js`.
- Styling constraints: `static/app.css`.
- Documentation: `README.md`, `TUNING.md`, `OPTIMIZATION_PLAN.md`.
- Static demo build: `scripts/build-static.mjs`.

## Workflow

1. Identify every user-facing string touched by the request before editing.
2. Preserve DOM ids, data attributes, WebSocket payload types, environment variable names, and route names exactly.
3. Prefer Traditional Chinese (`zh-TW`) when localizing Chinese. Use concise technical English when the app surface stays English.
4. Keep chip/status labels compact; long labels can overflow the topbar on phone screens.
5. If repairing garbled text, infer from adjacent commands and repo behavior, then flag uncertain terms in the final answer.
6. Verify docs tables still match `app/config.py` whenever parameter descriptions are translated.

## Terms

- Phone page: mobile camera capture surface.
- Viewer page: browser display surface for JPEG frames and detection boxes.
- Detection box: bounding box, not "checkbox".
- Demo mode: privacy-safe static GitHub Pages mode with frozen camera/WebSocket/upload behavior.
- Remote storage: optional metadata upload controlled by `REMOTE_STORAGE_*`.

## Validation

Run `npm.cmd run test` after UI/docs copy changes when possible. Run `npm.cmd run build` when static demo copy or build output paths are affected.
