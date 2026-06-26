# YOLO Elf

> 將手機相機畫面即時串流到本機 YOLO 偵測器，並在瀏覽器中檢視偵測框。
> Stream your phone camera to a local YOLO detector and view detection boxes live in any browser.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-0B23A9)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](./LICENSE)

---

## 簡介 | Description

**YOLO Elf** 是一套輕量的 FastAPI 網頁應用：任何具備相機的裝置（手機、平板、桌機網路攝影機）都能擔任「錄影端 / Recorder」，把畫面編碼成 JPEG 後透過 WebSocket 推送到本機伺服器；伺服器以單一共用的偵測管線執行 YOLO 推論，並把畫面與偵測框即時回傳給所有「檢視端 / Viewer」。推論完全在你自己的機器上進行，畫面不會離開本機，除非你主動啟用遠端儲存。

**YOLO Elf** is a small FastAPI web app for streaming phone camera frames to a local YOLO detector and viewing detection boxes in a browser. Any device with a camera can take the **recorder** role and push JPEG frames over a WebSocket; the server runs one shared detection pipeline and fans the frames plus boxes back to every **viewer**. Inference stays on your own machine — frames never leave the host unless you explicitly enable remote storage.

---

## 核心功能特性 | Features

- **即時手機串流 / Live phone streaming** — 瀏覽器擷取相機畫面、編碼為 JPEG，並透過 `/ws/camera` WebSocket 推送，伺服器即時回傳偵測結果。
- **單管線、零延遲積壓 / Single low-latency pipeline** — 只保留最新一張畫面的單槽佇列（single-slot queue），舊畫面會被丟棄，確保偵測永遠跟得上即時輸入。
- **快速 / 精準雙模式 / Fast & Accurate presets** — 可在小型快速模型與大型高精度模型間即時切換（Viewer 的「快速 / 精準」切換或 `POST /api/detector/mode`），無需重啟。
- **執行階段調參 / Runtime tuning** — Settings 頁面可即時修改模型、類別、信心門檻與影像尺寸，立即生效。
- **開放詞彙偵測 / Open-vocabulary detection** — 支援 YOLO-World / YOLOE 模型，以文字提示（如 `person,backpack,fire extinguisher`）自訂偵測類別。
- **第二階段分類器 / Second-stage classifier** — 選用的圖鑑模式：裁切每個偵測框並分類，為物件標註物種 / 細分類別。
- **多檢視端廣播 / Unlimited viewers** — 一次只有一個錄影端，但檢視端數量不限，全部接收相同畫面。
- **錄影與中繼資料 / Recording & metadata** — 透過瀏覽器 `MediaRecorder` 錄影，可存本機、遠端或兩者，並附帶逐格偵測 `.detections.json` sidecar。
- **遠端存取 / Remote access** — 內建 Tailscale Serve 輔助指令，讓外網手機透過 HTTPS 擔任錄影端。
- **靜態展示版 / Static demo** — 一鍵建置隱私安全的 GitHub Pages 展示頁（停用相機、串流與上傳）。
- **GPU / CPU 自動偵測 / Auto device** — 自動解析 CUDA / CPU 裝置，支援 FP16 半精度推論。

---

## 系統需求 | Prerequisites

| 項目 / Item | 需求 / Requirement |
| --- | --- |
| 作業系統 / OS | Windows（隨附 PowerShell 輔助腳本 / bundled PowerShell helpers）。核心 FastAPI 應用本身跨平台。 |
| Python | 3.10 以上 / 3.10 or newer |
| Node.js | 選用，用於 npm scripts 與靜態建置 / optional, for npm scripts & static build |
| GPU | 選用，支援 CUDA 可大幅加速推論 / optional CUDA GPU for fast inference |
| Tailscale | 選用，供外網手機遠端存取 / optional, for remote phone access |

主要 Python 相依套件 / Key Python dependencies（見 `requirements.txt`）：`fastapi`、`uvicorn[standard]`、`numpy`、`pillow`、`ultralytics`、`httpx`、`websockets`、`pytest`。

---

## 安裝步驟 | Installation

### 1. 取得原始碼 / Clone the repository

```powershell
git clone <repository-url> yolo_elf
cd yolo_elf
```

### 2. 建立環境並安裝相依套件 / Set up the environment

`scripts/setup.ps1` 會建立 `.venv` 虛擬環境、升級 pip 並安裝 `requirements.txt`：

`scripts/setup.ps1` creates a `.venv`, upgrades pip, and installs `requirements.txt`:

```powershell
# CPU 版本 / CPU-only
.\scripts\setup.ps1

# 安裝 CUDA 版 PyTorch（GPU 加速）/ install CUDA PyTorch wheels for GPU
.\scripts\setup.ps1 -Cuda
```

> 💡 若 Python 不在 PATH 中，可指定路徑：`.\scripts\setup.ps1 -Python C:\path\to\python.exe`
> If Python is not on your PATH, pass `-Python C:\path\to\python.exe`.

安裝完成後，腳本會印出 `torch` 與 `ultralytics` 版本以確認環境正常。
On success the script prints the installed `torch` and `ultralytics` versions to confirm the setup.

---

## 快速上手 | Quick Start / Usage

### 啟動伺服器 / Run the server

```powershell
.\scripts\run.ps1
```

或透過 npm scripts（`package.json` 已包裝好 PowerShell 輔助指令）/ or via the npm scripts:

```powershell
npm run dev   # 自動重載，監聽 0.0.0.0:8766 / auto-reload on 0.0.0.0:8766
npm start     # 不重載 / no reload
```

### 開啟頁面 / Open the pages

每個頁面頂部都有 **Recorder / Viewer / Settings** 切換，任何裝置都能在三者間切換：
Each page header has a **Recorder / Viewer / Settings** switch so any device can flip between them:

| 頁面 / Page | 網址 / URL | 用途 / Purpose |
| --- | --- | --- |
| 錄影端 / Recorder | `http://127.0.0.1:8766/recorder`（別名 `/phone`） | 開啟相機、擷取畫面、錄影 / camera capture & recording |
| 檢視端 / Viewer | `http://127.0.0.1:8766/viewer` | 即時畫面 + 偵測框 / live frames + detection boxes |
| 設定 / Settings | `http://127.0.0.1:8766/settings` | 執行階段調整模型 / 類別 / 門檻 / live detector config |

> 一次只有一個錄影端：在新裝置上取得錄影端角色，會把相機交接過去；檢視端數量不限。
> Only one recorder streams at a time — taking the recorder role hands the camera over from the previous device. Viewers are unlimited.

### 啟動時帶入參數 / Bake values in at launch

Settings 頁面可在執行階段即時調整偵測器（重啟後會回到環境變數 / 預設值）。若想在啟動時就固定數值，可傳入 `run.ps1` 參數或設定[環境變數](#設定--configuration)：

The Settings page edits the detector live (runtime edits reset on restart). To bake values in at launch, pass `run.ps1` parameters or set the [environment variables](#設定--configuration):

```powershell
.\scripts\run.ps1 -DetectMode accurate -ConfThresh 0.3 -ImgSize 1280 `
    -FastModel yolov8s.pt -AccurateModel yolov8x.pt `
    -Classes "person,backpack,fire extinguisher"
```

### 遠端存取（Tailscale）/ Remote access

瀏覽器僅允許在 `https://` 或 `localhost` 來源存取相機，因此外網手機需要 HTTPS 網址才能擔任錄影端。內建輔助指令會把本機伺服器放到 Tailscale Serve 的 HTTPS 端點後方：

Browsers only allow camera access on `https://` or `localhost` origins, so a phone on another network needs an HTTPS URL. The bundled helper puts the server behind a Tailscale Serve HTTPS endpoint:

```powershell
npm run tailscale   # tailscale serve --bg --https=8766 8766
```

在手機開啟印出的 `https://<machine>.<tailnet>.ts.net/` 網址並取得錄影端角色即可。推論仍在桌機執行；Tailscale 只負責傳輸 JPEG 畫面與偵測框。
Open the printed `https://<machine>.<tailnet>.ts.net/` URL on the phone and take the recorder role. Inference still runs on the desktop; Tailscale only carries frames and boxes.

### 執行測試 / Run the tests

```powershell
.\scripts\run-tests.ps1
```

測試腳本會執行 Python 測試、檢查 benchmark 腳本的 Python 語法，並對瀏覽器 / 建置 JavaScript 執行 `node --check`。
The test script runs the Python tests, checks the benchmark script's syntax, and runs `node --check` on the browser/build JavaScript.

### npm Scripts

| 指令 / Command | 動作 / Action |
| --- | --- |
| `npm run dev` | 自動重載，監聽 `0.0.0.0:8766` / run with auto-reload. |
| `npm start` | 不重載執行 / run without reload. |
| `npm run start:bg` | 在背景以分離模式啟動伺服器 / start detached in the background. |
| `npm run build` | 建置靜態 GitHub Pages 展示頁到 `dist/` / build the static demo. |
| `npm run tailscale` | 透過 Tailscale Serve 以 HTTPS 對外公開 / expose over HTTPS. |
| `npm run bench` | 執行偵測器 benchmark / run the detector benchmark. |
| `npm test` | 執行測試套件 / run the test suite. |

---

## 系統架構 | Architecture

伺服器執行單一、由所有連線客戶端共用的偵測管線：
The server runs a single detection pipeline shared by every connected client:

```
Recorder (browser)                Server (FastAPI)                  Viewers (browser)
 ┌────────────┐   JPEG / WS   ┌───────────────────────────┐   JPEG + boxes / WS  ┌──────────┐
 │  camera →  │ ────────────▶ │  stream hub (single-slot) │ ───────────────────▶ │ overlay  │
 │  encode    │  /ws/camera   │        ↓ newest frame     │     /ws/viewer       │ canvas   │
 └────────────┘               │  detection worker (YOLO)  │                      └──────────┘
        ▲  boxes only         │   + optional classifier   │
        └─────────────────────│        ↓ result           │── optional ──▶ remote storage
                              └───────────────────────────┘
```

- **擷取 / Capture** — 錄影端開啟相機、在瀏覽器中把畫面編碼成 JPEG，並透過 `/ws/camera` WebSocket 推送。
- **串流中樞 / Stream hub**（`app/stream_state.py`）— 追蹤唯一的活躍錄影端、檢視端集合，以及單槽畫面佇列。佇列只保留最新一張：若新畫面在舊畫面仍等待時抵達，舊畫面會被丟棄，使偵測永不落後即時輸入。
- **偵測工作者 / Detection worker**（`app/main.py`）— 背景任務取出最新畫面，以 `asyncio.to_thread` 在事件迴圈之外執行 YOLO 推論，選擇性執行第二階段分類器，再發佈結果。
- **扇出 / Fan-out** — 每筆結果回傳給錄影端（僅偵測框）與所有檢視端（JPEG 畫面 + 偵測框），並排入選用的遠端儲存上傳佇列。
- **偵測器 / Detector**（`app/detector.py`）— 依 preset 載入並快取 YOLO 權重、解析 CUDA / CPU 裝置、套用開放詞彙提示，並執行選用的「裁切後分類」第二階段。

---

## 設定 | Configuration

可透過環境變數調整行為。常用變數如下：
Behaviour is driven by environment variables. The most common ones:

| 變數 / Name | 預設 / Default | 說明 / Description |
| --- | --- | --- |
| `DETECT_MODE` | `fast` | 啟動時的偵測 preset：`fast`（用 `YOLO_MODEL`）或 `accurate`（用 `YOLO_MODEL_ACCURATE`）。可在執行階段由 Viewer 的「快速 / 精準」切換或 `POST /api/detector/mode` 變更。 |
| `YOLO_MODEL` | `yolov8s.pt` | **快速** preset 使用的模型，偏向速度。 |
| `YOLO_MODEL_ACCURATE` | `yolov8x.pt` | **精準** preset 使用的模型；越大越準但越慢，首次使用自動下載。可試 `yolo11x.pt`（最新最高精度）或 `yolov8x-oiv7.pt`（Open Images V7，600 類）。 |
| `YOLO_CLASSES` | _(空 / empty)_ | 開放詞彙模型（YOLO-World / YOLOE）的逗號分隔提示類別。留空維持模型內建詞彙；需搭配 `-world`/`-worldv2` 模型，封閉集偵測器會忽略。範例：`person,backpack,fire extinguisher`。 |
| `YOLO_DEVICE` | `auto` | `auto`、`cpu`、`0` 或其他 Ultralytics 裝置目標。 |
| `YOLO_HALF` | `1` | 對支援的 CUDA 裝置啟用 FP16（CPU 忽略）。 |
| `YOLO_WARMUP` | `0` | 啟動時預熱偵測器。 |
| `CONF_THRESH` | `0.2` | 偵測信心門檻。越低召回越高、誤判越多。 |
| `IMG_SIZE` | `1280` | 偵測影像尺寸。越大對小 / 遠物件越有利但越慢。 |
| `CLASSIFIER_MODEL` | _(空 / empty)_ | 選用的第二階段分類器（圖鑑模式），為每個偵測框內的物件命名物種。留空則僅偵測。可試 `yolov8x-cls.pt`（ImageNet 1000 類），首次使用自動下載。 |
| `CLASSIFIER_MIN_CONF` | `0.0` | 附加物種標籤所需的最低 top-1 信心。調高以抑制低信心猜測。僅在設定 `CLASSIFIER_MODEL` 時生效。 |
| `CLASSIFIER_MAX_BOXES` | `5` | 節流：每格最多分類這麼多框（取面積最大者）。在擁擠畫面限制分類成本；其餘框保留偵測標籤但無物種。僅在設定 `CLASSIFIER_MODEL` 時生效。 |
| `FRAME_FPS` | `10` | 手機擷取的請求 FPS。 |
| `CAPTURE_WIDTH` | `1920` | 擷取寬度上限。畫面保持相機原始長寬比，不放大或拉伸。 |
| `CAPTURE_HEIGHT` | `1080` | 擷取高度上限。畫面保持相機原始長寬比，不放大或拉伸。 |
| `JPEG_QUALITY` | `0.9` | WebSocket 傳送的 JPEG 品質。 |
| `MAX_FRAME_BYTES` | `5242880` | 可接受的最大畫面大小。 |
| `RECORDING_ENABLED` | `1` | 啟用上傳到伺服器的瀏覽器錄影。 |
| `RECORDING_KEEP_LOCAL_COPY` | `1` | 即使在 `remote` 模式也保留桌機副本。設 `0` 為僅上傳（不存本機檔）。 |
| `RECORDING_STORAGE_DIR` | `recordings` | 儲存上傳錄影的目錄。 |
| `RECORDING_MAX_BYTES` | `262144000` | 可接受的最大錄影上傳大小。 |
| `REMOTE_STORAGE_URL` | _(空 / empty)_ | 選用的偵測中繼資料上傳端點。未設定則停用遠端儲存。 |
| `REMOTE_STORAGE_TOKEN` | _(空 / empty)_ | 遠端上傳的選用 bearer token。 |
| `REMOTE_STORAGE_INCLUDE_FRAME` | `0` | 偵測上傳是否包含 JPEG 畫面位元組。 |
| `REMOTE_STORAGE_RECORDING_URL` | _(空 / empty)_ | 選用的 multipart 錄影上傳端點。 |
| `REMOTE_STORAGE_QUEUE_SIZE` | `100` | 背景遠端上傳佇列大小。 |
| `REMOTE_STORAGE_TIMEOUT` | `5.0` | 遠端上傳逾時（秒）。 |
| `REMOTE_STORAGE_RETRIES` | `2` | 每次遠端上傳的重試次數。 |

> 遠端儲存預設停用，只有設定 `REMOTE_STORAGE_URL` 才會啟用；啟用後伺服器於背景上傳偵測中繼資料，僅在 `REMOTE_STORAGE_INCLUDE_FRAME=1` 時包含畫面。
> Remote storage is disabled unless `REMOTE_STORAGE_URL` is set; frames are included only when `REMOTE_STORAGE_INCLUDE_FRAME=1`.

完整的 GPU / 精度調校、preset 切換、開放詞彙模型與第二階段分類器說明，請見 **`TUNING.md`**。
See **`TUNING.md`** for in-depth GPU/accuracy tuning, preset switching, open-vocabulary models, and the second-stage classifier.

---

## HTTP & WebSocket API

**頁面 / Pages**

| 路由 / Route | 用途 / Purpose |
| --- | --- |
| `GET /` | 重新導向至 `/phone` / redirects to `/phone`. |
| `GET /phone`, `GET /recorder` | 擷取 + 錄影頁（同一頁的裝置中立別名）/ capture + recording page. |
| `GET /viewer` | 即時畫面 + 偵測框 / live frames + detection boxes. |
| `GET /settings` | 執行階段偵測器設定 / runtime detector configuration. |

**JSON API**

| 路由 / Route | 用途 / Purpose |
| --- | --- |
| `GET /health` | 存活探測 / liveness probe (`{"status": "ok"}`). |
| `GET /api/status` | 完整執行階段快照：串流統計、偵測器狀態、錄影、遠端儲存。 |
| `POST /api/detector/mode` | 切換 preset。Body：`{"mode": "fast"}` 或 `{"mode": "accurate"}`。 |
| `GET /api/detector/config` | 目前偵測器設定 / current detector configuration. |
| `POST /api/detector/config` | 執行階段更新設定（部分更新，只變更傳入的鍵）/ partial runtime update. |
| `POST /api/recordings` | 上傳錄影，依 `X-Yolo-Elf-Storage-Mode` 標頭路由儲存 / upload a recording. |
| `POST /api/recordings/{id}/metadata` | 為錄影附加逐格偵測 sidecar / attach detection sidecar. |
| `GET /api/recordings/{id}` | 下載已儲存的錄影 / download a recording. |
| `GET /api/recordings/{id}/metadata` | 下載錄影的 `.detections.json` sidecar / download the sidecar. |

**WebSockets**

| 路由 / Route | 流向 / Flow | 內容 / Payload |
| --- | --- | --- |
| `/ws/camera` | recorder → server | 二進位 JPEG 畫面；文字 `client_state` 訊息回報儲存模式與錄影狀態。伺服器連線時回覆 `config` 訊息。 |
| `/ws/viewer` | server → viewer | 每格 JSON 中繼資料後接二進位 JPEG，並可依請求附上 `status` 快照。 |

---

## 專案架構說明 | Project Structure

| 路徑 / Path | 內容 / Contents |
| --- | --- |
| `app/main.py` | FastAPI 應用、路由、WebSocket 處理器、偵測工作者。 |
| `app/detector.py` | YOLO 載入、推論、框擷取、第二階段分類器。 |
| `app/stream_state.py` | 串流中樞：錄影端 / 檢視端追蹤、畫面佇列、指標。 |
| `app/recordings.py` | 錄影上傳、中繼資料 sidecar、本機儲存。 |
| `app/remote_storage.py` | 偵測中繼資料與錄影的背景上傳佇列。 |
| `app/config.py` | 環境變數驅動的 `Settings` 與驗證。 |
| `static/` | 瀏覽器頁面與資產（recorder、viewer、settings）。 |
| `scripts/` | PowerShell / Node 輔助腳本：setup、run、bench、tailscale、靜態建置。 |
| `tests/` | Pytest 測試套件。 |
| `TUNING.md` | GPU / 精度調校、preset 切換、開放詞彙與分類器完整說明。 |

---

## 靜態網頁展示 | Static Web Demo

GitHub Actions 工作流程會建置隱私安全的靜態 GitHub Pages 展示頁：
The GitHub Actions workflow builds a static, privacy-safe demo for GitHub Pages:

```powershell
npm run build:github-pages
```

建置會把以下內容寫入 `dist/`：`index.html`（靜態 viewer 展示）、`viewer/index.html`、`phone/index.html`、以及共用的 `static/` 資產。

在此靜態展示中，隱私敏感的即時功能皆停用：相機存取、WebSocket 串流、錄影與遠端上傳皆關閉，偵測框改由合成的展示畫面繪製。推送至預設分支或手動 `workflow_dispatch` 時，工作流程會把 `dist/` 上傳到 GitHub Pages。

In this static demo, privacy-sensitive live features are frozen: camera access, WebSocket streaming, recording, and remote uploads are all disabled, and boxes are drawn from a synthetic demo frame. On pushes to the default branch (or manual `workflow_dispatch`), the workflow uploads `dist/` to GitHub Pages.

---

## 授權條款 | License

本專案採用 **GNU 通用公共授權條款第 3 版（GPLv3）** 釋出。
This project is released under the **GNU General Public License v3.0 (GPLv3)**.

你可以自由使用、研究、修改與散布本軟體，但任何衍生作品在散布時必須同樣以 GPLv3 授權並提供原始碼。本軟體不附帶任何擔保。完整條款請見專案根目錄的 [`LICENSE`](./LICENSE) 檔案。

You are free to use, study, modify, and distribute this software, provided that derivative works are also distributed under the GPLv3 and accompanied by their source code. This software comes with no warranty. See the [`LICENSE`](./LICENSE) file in the repository root for the full terms.
