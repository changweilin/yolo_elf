# YOLO Elf — AI Skill and Sub-Agent Matrix

Division of labour only. Macro principles and the module map live in the
root `CLAUDE.md`.

## Artifact layout

| Tool | Skills | Sub-agents |
| --- | --- | --- |
| Claude | `.claude/skills/yolo-elf-*/SKILL.md` | `.claude/agents/yolo-elf-*.md` |
| ChatGPT/Codex | `.codex/skills/yolo-elf-*/SKILL.md` | `.codex/subagents/yolo-elf-*.md` |

The two `SKILL.md` sets are byte-identical by design — edit a skill and mirror it
to the other tree in the same commit. Only the sub-agent prompt wrappers differ.

## Role matrix

| Role | Skill | Primary files | Validation |
| --- | --- | --- | --- |
| Copy and localization | `yolo-elf-i18n` | `static/*.html`, `static/*.js`, `README.md`, `TUNING.md` | `npm.cmd run test`, `npm.cmd run build` |
| UI event logic | `yolo-elf-ui-events` | `static/phone.js`, `static/viewer.js`, `static/app.css`, `scripts/build-static.mjs` | `npm.cmd run test`, `npm.cmd run build`, browser check |
| Config and data contracts | `yolo-elf-config-data` | `app/config.py`, `app/main.py`, `app/remote_storage.py`, tests/docs/scripts | `npm.cmd run test` |
| Geometry and metrics | `yolo-elf-spatial-metrics` | `app/detector.py`, `app/stream_state.py`, `static/phone.js`, `static/viewer.js`, bench scripts | `npm.cmd run test`, `scripts/bench.ps1` |
| Market and experiment analysis | `yolo-elf-market-science` | `README.md`, `TUNING.md`, benchmark/status surfaces | benchmarks + cited current sources |

## Delegation

1. Load the matching skill before doing the work.
2. Split in parallel only along the role boundaries above; two sub-agents must
   never write the same files.
3. Run at least the role's validation command before reporting done.

**Market-analysis caveat:** the repo holds no real market, funnel, revenue,
retention or A/B data. `yolo-elf-market-science` therefore works in
"hypothesis + measurable experiment + runtime benchmark" mode: quantify what
data exists, propose instrumentation for what does not, never fabricate a
conclusion.
