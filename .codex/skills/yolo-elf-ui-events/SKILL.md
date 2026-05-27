---
name: yolo-elf-ui-events
description: Trace, modify, and verify UI event logic for the YOLO Elf phone and viewer experiences. Use when working on camera startup/stop, WebSocket reconnects, adaptive FPS, demo mode, overlay redraws, theme behavior, HUD state chips, or static build interactions in static/*.js/html/css.
---

# YOLO Elf UI Events

## Focus

Maintain the live phone-to-viewer interaction loop. UI changes must respect camera permissions, WebSocket state, binary frame transport, adaptive pacing, and canvas overlay rendering.

## File Map

- Phone capture: `static/phone.js`, `static/phone.html`.
- Viewer display: `static/viewer.js`, `static/viewer.html`.
- Shared visual system: `static/app.css`, `static/theme.js`.
- Static demo packaging: `scripts/build-static.mjs`.
- Backend contracts: `/ws/camera`, `/ws/viewer`, `/api/status` in `app/main.py`.

## Event Flow

1. Phone opens camera with `getUserMedia`, paints video into `captureCanvas`, JPEG-encodes with `toBlob`, and sends binary bytes to `/ws/camera`.
2. Backend sends config and detection JSON to the phone; phone updates overlay and adaptive status.
3. Viewer receives frame metadata JSON followed by binary JPEG bytes from `/ws/viewer`.
4. Viewer pairs pending metadata with the next binary frame, renders the image, then draws boxes on the overlay canvas.
5. Demo mode disables live camera/WebSocket/upload behavior and renders synthetic detection boxes.

## Guardrails

- Do not change DOM ids or `data-start-camera` without updating JS selectors and tests.
- Keep `payload.type` values stable: `config`, `detection`, `error`, `status`, `frame`.
- Preserve JSON-then-binary ordering for viewer frames.
- Keep `requestAnimationFrame(drawOverlay)` loops non-blocking.
- Avoid layout changes that make HUD chips overlap on mobile.

## Workflow

1. Map the requested behavior to the exact event handlers and state fields.
2. Check whether the backend WebSocket contract also needs a change.
3. Make the smallest UI state change that preserves reconnect, stop, and demo-mode behavior.
4. Update tests when DOM text, routes, or WebSocket payload shape changes.

## Validation

Run `npm.cmd run test` for JS syntax and Python flow tests. Run `npm.cmd run build` for demo/static route changes. For visual behavior, run `npm.cmd run dev:local` and inspect `/phone` and `/viewer`.
