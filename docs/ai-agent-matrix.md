# YOLO Elf AI Skill And Sub-Agent Matrix

本專案已依職能建立 ChatGPT/Codex 與 Claude 兩套對應版本。

## Artifact Layout

- ChatGPT/Codex skills: `.codex/skills/yolo-elf-*`
- ChatGPT/Codex sub-agent prompts: `.codex/subagents/yolo-elf-*.md`
- Claude skills: `.claude/skills/yolo-elf-*`
- Claude sub-agents: `.claude/agents/yolo-elf-*.md`

## Project Diagnosis

YOLO Elf 是 FastAPI + WebSocket + 靜態前端的即時偵測 app。

- `static/phone.js`：手機相機、JPEG capture、WebSocket 傳 frame。
- `app/main.py`：FastAPI routes、WebSocket 端點、背景 detection worker。
- `app/stream_state.py`：frame queue、viewer/camera client、丟幀與延遲統計。
- `app/detector.py`：JPEG decode、Ultralytics YOLO inference、box extraction。
- `static/viewer.js`：接收 metadata + binary JPEG，繪製 detection overlay。
- `app/config.py`：所有 runtime/env 參數。
- `app/remote_storage.py`：選配遠端 metadata/frame upload。

## Role Matrix

| Function | Skill | Primary Files | Main Validation |
| --- | --- | --- | --- |
| 多國語言翻譯與文案 | `yolo-elf-i18n` | `static/*.html`, `static/*.js`, `README.md`, `TUNING.md` | `npm.cmd run test`, `npm.cmd run build` |
| UI 邏輯事件 | `yolo-elf-ui-events` | `static/phone.js`, `static/viewer.js`, `static/app.css`, `scripts/build-static.mjs` | `npm.cmd run test`, `npm.cmd run build`, browser check |
| 參數與資料契約管理 | `yolo-elf-config-data` | `app/config.py`, `app/main.py`, `app/remote_storage.py`, tests/docs/scripts | `npm.cmd run test` |
| 空間運算與數值分析 | `yolo-elf-spatial-metrics` | `app/detector.py`, `app/stream_state.py`, `static/phone.js`, `static/viewer.js`, benchmark scripts | `npm.cmd run test`, `scripts/bench.ps1` |
| 市場科學化與實驗分析 | `yolo-elf-market-science` | `README.md`, `TUNING.md`, `OPTIMIZATION_PLAN.md`, benchmark/status surfaces | benchmarks plus cited current sources |

## Market Analysis Caveat

目前 repo 沒有真實市場、funnel、付費、留存或 A/B data。`yolo-elf-market-science` 因此採「市場假設 + 可測實驗 + runtime benchmark」模式：有資料就量化，沒資料就提出 instrumentation，不假裝已有結論。

## Delegation Pattern

1. 先用對應 skill 讀取職能規範。
2. 若任務可平行拆分，依 `.codex/subagents` 或 `.claude/agents` 委派。
3. 不同 sub-agent 應避免同時寫同一組檔案。
4. 完成後至少跑 matrix 中的主驗證命令。
