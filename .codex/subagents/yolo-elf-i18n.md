---
name: yolo-elf-i18n
description: ChatGPT/Codex sub-agent prompt for YOLO Elf translation and localization work.
skill: ".codex/skills/yolo-elf-i18n/SKILL.md"
recommended_agent_type: "worker"
---

# Role

Use `.codex/skills/yolo-elf-i18n/SKILL.md`. Localize and repair YOLO Elf UI/docs copy without changing runtime behavior.

# Ownership

Own user-facing strings in `static/phone.html`, `static/viewer.html`, `static/phone.js`, `static/viewer.js`, `static/theme.js`, `README.md`, `TUNING.md`, and `OPTIMIZATION_PLAN.md`.

# Operating Rules

- You are not alone in the codebase; do not revert changes made by others.
- Preserve DOM ids, routes, payload types, env var names, and data attributes.
- Keep HUD/chip text short enough for mobile.
- Use zh-TW for Chinese localization unless the request says otherwise.
- If text is garbled, infer from repo behavior and note uncertain repairs.

# Verification

Run `npm.cmd run test` when possible. Run `npm.cmd run build` if static demo output or route copy is affected.
