# CLAUDE.md — YOLO Elf 專案記憶

> 這份是宏觀原則。各職能的細部規範在 `.claude/skills/yolo-elf-*/SKILL.md`（分工矩陣見 `docs/ai-agent-matrix.md`）。原則與細則衝突時，以本檔為準。

## 專案是什麼

FastAPI + WebSocket + 靜態前端的即時物件偵測 app：手機（錄影端）把 JPEG 影格推到 `/ws/camera`，伺服器以單一 GPU worker 跑 YOLO 推論，把畫面與偵測框廣播給所有 `/ws/viewer`。推論完全在本機，畫面預設不離開主機。

roadmap 上的功能（追蹤、告警、zones、歷史、驗證、指標、export、多相機、VLM 通道）皆已完成；後續是維護與增量改進。

## 模組地圖

- `app/` — 每個功能一個模組（`config` 設定、`main` 路由與 worker、`stream_state` 串流狀態、`detector` 推論、`zones`/`alerts`/`events` 區域/告警/SQLite 歷史、`auth`/`metrics`/`recordings`/`remote_storage`/`vlm`）。
- `static/` — 無框架前端（`phone` 錄影端、`viewer` 檢視端、`history`/`settings`/`login` 頁、共用 `app.css`/`theme.js`）。
- `scripts/` — PowerShell 啟動/測試/benchmark 腳本 + Node 靜態建置（`build-static.mjs` 產 GitHub Pages demo）。
- `tests/` — `test_<feature>.py` 單元測試 + `test_app.py` 真實 ASGI lifespan 測試。
- 文件 — `README.md`（功能列、設定表、API 表、專案結構）、`TUNING.md`（操作與調校）。

## 核心原則

**架構**
1. 新功能 = 獨立的 `app/<feature>.py`：純函式 + Engine/Registry 類別，`__init__(settings)` 內 fail-loud 驗證。
2. GPU 存取由**單一** detection worker 序列化——不要每路/每請求開 thread 搶 GPU。低延遲靠「單槽佇列只留最新影格、丟舊的」，不靠排隊。
3. 效能主張一律用 benchmark（`scripts/bench_detector.py`）佐證，不憑直覺；GPU 專屬優化必須能在 CPU 上優雅退化。

**設定**
4. 所有設定走 `app/config.py` 的既有 helper（`_bool_env`/`_bounded_int_env`/`_bounded_float_env`/`_list_env`/`_choice_env`），欄位進 `Settings`；數值必須有界、錯誤訊息要明確。
5. 預設值 = 最安全、最接近舊行為的那個。隱私敏感功能（遠端上傳、影格外傳）一律 opt-in。

**隱私與安全**
6. 畫面不離機是產品承諾；任何把影格送出主機的路徑都必須顯式啟用。
7. 對外端點（webhook、遠端儲存 URL）**只能由環境變數設定**，不開放 runtime API 修改（SSRF 防護）。權杖不進網址、不進 log。

**相容性**
8. 改動用「加維度」而非「換行為」：新參數可選、不帶 = 舊行為；未設定新功能時行為與過去逐位元相同（範例：`CAMERAS` 留空 = 單相機）。
9. SQLite schema 演進用 `ALTER TABLE ADD COLUMN` 平滑升級，舊資料容忍 NULL。
10. 對外契約保持穩定：WebSocket `payload.type` 值、JSON-then-binary 影格順序、DOM id、路由、環境變數名——改名即破壞，寧可新增不要改名。

**幾何與數值**
11. 偵測框是 `xyxy` 來源影像像素座標；跨 Python/JS 邊界時先寫下來源與目的座標空間再動手。`clamp_xyxy` 收邊、`fitContain` 對齊 CSS `object-fit: contain`、canvas 經 `devicePixelRatio` 校正；zones 用 0–1 正規化座標。

**測試與驗證**
12. 每個 PR：`pytest` 全綠 + 改動的 JS 過 `node --check`。Windows 上用 `npm.cmd run test`（=pytest + py_compile + node --check）；UI 變更盡量瀏覽器實跑（無頭分頁會暫停 rAF，canvas 幾何需手動驗證）。
13. app 層測試走真實 ASGI lifespan；autouse fixture 要清乾淨相關環境變數。

**文件與展示**
14. 功能落地 = 程式 + 測試 + README（設定表/API 表）+ TUNING（操作說明）+ 靜態 demo 有對應展示，一次到位；文件表格必須與 `app/config.py` 一致。
15. 文案以 zh-TW 為主、README 雙語；commit 訊息用 zh-TW。UI 標籤保持精簡，避免手機 topbar 溢出。

## 常用命令

| 目的 | 命令 |
| --- | --- |
| 測試（Windows） | `npm.cmd run test` |
| 測試（跨平台） | `python -m pytest -q` + `node --check <改動的 js>` |
| 靜態 demo 建置 | `npm run build` |
| 本機開發 | `npm run dev:local` → `/phone`、`/viewer` |
| Benchmark | `.\scripts\bench.ps1 -Frames 20 -Warmup 3 -Device cpu -ImgSize 960 -Quality 0.85` |

CI（`.github/workflows/ci.yml`）在 windows-latest 跑 `npm test` + 靜態建置，main 分支另行部署 GitHub Pages。
