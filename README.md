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
- **YOLO26 與 NMS-free 推論 / YOLO26 & NMS-free inference** — 預設模型為 YOLO26（`yolo26s.pt` / `yolo26x.pt`），其一對一頭部直接輸出去重後的框、不跑 NMS，因此沒有 `iou` 門檻要調、匯出的圖也不含後處理節點。`YOLO_END2END=auto|on|off` 可強制切換頭部（`auto` 沿用權重預設；YOLOv8／v11 等無此頭部的模型會忽略此設定），`YOLO_MAX_DET` 調整每幀框數上限。**注意：在本專案的 PyTorch 推論路徑上，NMS-free 本身不會變快**（實測見 `TUNING.md`）；速度紅利主要出現在匯出後的 ONNX／邊緣執行環境。
- **七種任務通道 / Seven task heads** — Viewer 右側面板依輸出性質分成「物件疊加」與「整張畫面」兩個分頁，勾選即時切換偵測頭，換頭即換權重、不必重啟。同一分頁內的任務可以**同時勾選**（最多 4 個，每多一個就在同一張影格多跑一次推論，延遲近似倍增）；「整張畫面」的兩種逐像素圖層互斥，勾一個會取消另一個。可選的任務：**物件偵測**（方框）、**實例分割**（輪廓多邊形）、**姿態估計**（COCO-17 骨架）、**旋轉框 OBB**（可傾斜方框）、**開放詞彙**（YOLOE-26，用 `YOLO_CLASSES` 文字提示決定要找什麼）、**語意分割**與**單目深度**（逐像素圖層，疊在畫面上，深度附公尺範圍）。前五種仍輸出方框，因此追蹤 / 區域 / 告警 / 歷史照常運作；語意與深度沒有方框，那些功能會看到空白幀（設計如此）。預設 `detect`，行為與過去完全相同。
- **多物件追蹤 / Multi-object tracking** — 內建六種追蹤器（ByteTrack / BoT-SORT / TrackTrack / FastTrack / OC-SORT / Deep OC-SORT），跨影格為每個物件維持穩定 `track_id`，標籤以 `#id` 顯示、錄影中繼資料一併記錄；以 `YOLO_TRACKER` 選擇，可 `YOLO_TRACK=0` 關閉。
- **偵測歷史 / Detection history** — 以 `track_id` 為單位把每個物件聚合成一筆「出現紀錄」（首/末出現、停留秒數、經過的區域、最高信心），寫入本機 SQLite，`/history` 頁面可依類別 / 區域 / 時間範圍查詢。需 `YOLO_TRACK`（預設開），可 `EVENT_LOG_ENABLED=0` 關閉。
- **開放詞彙偵測 / Open-vocabulary detection** — 支援 YOLO-World / YOLOE 模型，以文字提示（如 `person,backpack,fire extinguisher`）自訂偵測類別。
- **VLM 語意通道 / VLM semantic channel** — 選用的 Florence-2 通道（`VLM_ENABLED=1`），與 YOLO **並存不取代**：慢速定時產生開放詞彙偵測框與場景描述，Viewer 以「YOLO / VLM」分頁切換。VLM 通道無信心分數與追蹤，故不進 zones / alerts / history；YOLO 通道維持全功能。詳見 `TUNING.md`。
- **ROI 區域 / Region-of-interest zones** — 在 Viewer 直接框選多邊形區域，偵測框會標記所屬區域並即時顯示佔用數；告警規則可限定「只在某區域內」觸發。座標正規化（0–1），跟著畫面自動縮放，可經 `POST /api/zones` 即時增修。
- **規則告警 / Rule-based alerts** — 依規則（類別、數量門檻、信心、區域）在偵測命中時觸發，帶冷卻時間去抖動；即時推送到 Viewer（含選用的瀏覽器通知）並可送出 webhook 串接外部系統。規則可經 `POST /api/alerts` 即時增修。
- **第二階段分類器 / Second-stage classifier** — 選用的圖鑑模式：裁切每個偵測框並分類，為物件標註物種 / 細分類別。
- **多相機 / Multi-camera** — 以 `CAMERAS` 定義允許清單（如 `front:前門,back:後院`），每路各自擁有畫面佇列、追蹤器狀態、ROI 區域、告警冷卻與歷史紀錄；單一 GPU worker 以公平佇列輪流處理各路，Viewer 以格狀同時監看、點一格放大。留空＝單相機，行為與過去完全相同。
- **多檢視端廣播 / Unlimited viewers** — 每路相機一次只有一個錄影端，但檢視端數量不限；檢視端可看全部或只訂閱其中一路（`/ws/viewer?camera_id=`）。
- **檢視端過濾與存圖 / Viewer filters & snapshot** — Viewer 可切換顯示哪些類別、拉一條「最低信心」滑桿（純前端過濾，不動後端偵測），並一鍵「存圖」把當前含框畫面以原始解析度輸出成 PNG。純前端，無需設定或 API。
- **錄影與中繼資料 / Recording & metadata** — 透過瀏覽器 `MediaRecorder` 錄影，可存本機、遠端或兩者，並附帶逐格偵測 `.detections.json` sidecar。
- **存取控制 / Access control** — 選用的共享權杖驗證：設定 `AUTH_TOKEN` 後，頁面 / REST / WebSocket 都需驗證。瀏覽器在 `/login` 輸入權杖換取 HttpOnly 簽章 cookie（連 WS 一起認證，權杖不進網址或 log）；程式端可用 `Authorization: Bearer`。留空＝不啟用（維持現狀）。
- **Prometheus 指標 / Prometheus metrics** — `GET /metrics` 以文字格式輸出既有指標（fps、延遲、丟幀、觀看人數、告警觸發數、活躍 sighting…），可直接接 Prometheus / Grafana。純讀取，可 `METRICS_ENABLED=0` 關閉。
- **遠端存取 / Remote access** — 內建 Tailscale Serve 輔助指令，讓外網手機透過 HTTPS 擔任錄影端。
- **靜態展示版 / Static demo** — 一鍵建置隱私安全的 GitHub Pages 展示頁（停用相機、串流與上傳）。
- **GPU / CPU 自動偵測 / Auto device** — 自動解析 CUDA / CPU 裝置，支援 FP16 半精度推論。
- **推論加速 / Inference acceleration** — `YOLO_MODEL` 可直接指向預先 export 的 `.engine`（TensorRT）/ `.onnx`；或設 `YOLO_EXPORT=engine|onnx` 讓伺服器首次載入時自動 export `.pt` 並改載入產物（產物會快取；export 失敗自動退回 `.pt`）。另附 `scripts/export_engine.py` 可離線預先 export。

---

## 系統需求 | Prerequisites

| 項目 / Item | 需求 / Requirement |
| --- | --- |
| 作業系統 / OS | Windows（隨附 PowerShell 輔助腳本 / bundled PowerShell helpers）。核心 FastAPI 應用本身跨平台。 |
| Python | 3.10 以上 / 3.10 or newer |
| Node.js | 選用，用於 npm scripts 與靜態建置 / optional, for npm scripts & static build |
| GPU | 選用，支援 CUDA 可大幅加速推論 / optional CUDA GPU for fast inference |
| Tailscale | 選用，供外網手機遠端存取 / optional, for remote phone access |

主要 Python 相依套件 / Key Python dependencies（見 `requirements.txt`）：`fastapi`、`uvicorn[standard]`、`numpy`、`pillow`、`ultralytics`（**需 8.4 以上**，YOLO26 權重、`end2end` 參數與新追蹤器都自 8.4.0 起提供）、`httpx`、`websockets`、`pytest`。

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
| 歷史 / History | `http://127.0.0.1:8766/history` | 偵測出現紀錄的時間軸與查詢 / sighting timeline & search |

> 每路相機一次只有一個錄影端：在新裝置上取得同一路的錄影端角色，會把相機交接過去；檢視端數量不限。
> Only one recorder streams per camera — taking that camera's recorder role hands it over from the previous device. Viewers are unlimited.

**多相機 / Multiple cameras**：設定 `CAMERAS` 後，每台裝置以 `?camera_id=` 指定自己是哪一路，Viewer 則自動變成格狀版面（點一格放大、再點回到全景）。

```powershell
$env:CAMERAS = "front:前門,back:後院"
.\scripts\run.ps1
```

| 用途 / Purpose | 網址 / URL |
| --- | --- |
| 前門的錄影端 / front-door recorder | `http://127.0.0.1:8766/recorder?camera_id=front` |
| 後院的錄影端 / back-yard recorder | `http://127.0.0.1:8766/recorder?camera_id=back` |
| 全部相機（格狀）/ all cameras | `http://127.0.0.1:8766/viewer` |
| 只看一路 / a single camera | `http://127.0.0.1:8766/viewer?camera_id=back` |

### 啟動時帶入參數 / Bake values in at launch

Settings 頁面可在執行階段即時調整偵測器（重啟後會回到環境變數 / 預設值）。若想在啟動時就固定數值，可傳入 `run.ps1` 參數或設定[環境變數](#設定--configuration)：

The Settings page edits the detector live (runtime edits reset on restart). To bake values in at launch, pass `run.ps1` parameters or set the [environment variables](#設定--configuration):

```powershell
.\scripts\run.ps1 -DetectMode accurate -ConfThresh 0.3 -ImgSize 1280 `
    -FastModel yolo26s.pt -AccurateModel yolo26x.pt `
    -End2End auto -MaxDet 300 -Tracker bytetrack.yaml `
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
Recorders (browser)                 Server (FastAPI)                  Viewers (browser)
 ┌────────────┐   JPEG / WS   ┌────────────────────────────────┐  JPEG + boxes / WS  ┌──────────┐
 │ camera A → │ ────────────▶ │ channel A (single-slot) ┐      │ ──────────────────▶ │ pane A   │
 │ camera B → │  /ws/camera   │ channel B (single-slot) ┴─▶ ready│    /ws/viewer     │ pane B   │
 └────────────┘  ?camera_id=  │        ↓ newest frame per camera│                    └──────────┘
        ▲  boxes only         │  detection worker (YOLO, 1×GPU) │
        └─────────────────────│   + optional classifier         │── optional ──▶ remote storage
                              └────────────────────────────────┘
```

- **擷取 / Capture** — 錄影端開啟相機、在瀏覽器中把畫面編碼成 JPEG，並透過 `/ws/camera?camera_id=<id>` WebSocket 推送。省略 `camera_id` 即為預設相機。
- **串流登錄 / Stream registry**（`app/stream_state.py`）— 依 `CAMERAS` 為每路建立一個 `StreamChannel`：各自的錄影端、單槽畫面佇列與計數器。佇列只保留最新一張：若新畫面在舊畫面仍等待時抵達，舊畫面會被丟棄，使偵測永不落後即時輸入。檢視端可訂閱全部或單一相機。
- **排程 / Scheduling** — 各路不去搶 GPU：提交端把「哪一路有新畫面」推進一個共用的就緒佇列，單一 worker 依序取用。這天生就是輪流（round-robin），且忙不過來時自然退化成較低的有效 FPS，而不是排隊爆掉。
- **偵測工作者 / Detection worker**（`app/main.py`）— 背景任務取出某一路的最新畫面，以 `asyncio.to_thread` 在事件迴圈之外執行 YOLO 推論，選擇性執行第二階段分類器，再發佈結果。GPU 存取序列化於此單一 worker。
- **扇出 / Fan-out** — 每筆結果回傳給該路錄影端（僅偵測框）與訂閱該路的檢視端（JPEG 畫面 + 偵測框），並排入選用的遠端儲存上傳佇列。
- **偵測器 / Detector**（`app/detector.py`）— 依 preset 載入並快取 YOLO 權重、解析 CUDA / CPU 裝置、套用開放詞彙提示，並執行選用的「裁切後分類」第二階段。追蹤器狀態存在 model 物件內，因此第一路沿用主 model，其餘每路各持一份實例，`track_id` 不會跨路混淆（記憶體成本＝額外相機數 × 權重）。

---

## 設定 | Configuration

可透過環境變數調整行為。常用變數如下：
Behaviour is driven by environment variables. The most common ones:

| 變數 / Name | 預設 / Default | 說明 / Description |
| --- | --- | --- |
| `CAMERAS` | _(空 / empty)_ | 多相機允許清單，逗號分隔的 `id` 或 `id:顯示名`。範例：`front:前門,back:後院`。留空＝單一隱含的 `default` 相機（行為與過去完全相同）。錄影端只能宣告清單內的 id，避免任意 id 灌爆伺服器，也讓 Viewer 版面在相機離線時保持穩定。偵測參數（preset / 信心 / 類別）全域共用；ROI 區域、告警與歷史則各自獨立。 |
| `MAX_CAMERAS` | `4` | `CAMERAS` 的數量上限（1–16）。超過即啟動失敗。先用 `scripts/bench_detector.py` 量測單卡吞吐再往上調。 |
| `DETECT_MODE` | `fast` | 啟動時的偵測 preset：`fast`（用 `YOLO_MODEL`）或 `accurate`（用 `YOLO_MODEL_ACCURATE`）。可在執行階段由 Viewer 的「快速 / 精準」切換或 `POST /api/detector/mode` 變更。 |
| `DETECT_TASK` | `detect` | 啟動時的偵測頭：`detect`、`segment`、`pose`、`obb`、`openvocab`、`semantic`、`depth`。可在 Viewer 的任務分頁或 `POST /api/detector/task` 即時切換。前五種輸出方框（追蹤／區域／告警／歷史照常）；`semantic`／`depth` 只輸出逐像素圖層，方框為空。 |
| `DETECT_TASKS` | _(空 / empty)_ | 同時執行的偵測頭清單，逗號分隔，例如 `detect,pose`。留空＝只跑 `DETECT_TASK` 一個頭（行為與過去完全相同）。每多一個頭就多跑一次推論，延遲近似線性增加；上限 4 個。`semantic` 與 `depth` 都會重畫整張畫面，只能擇一，同時填寫即啟動失敗。方框會合併成一份清單，每個框帶 `task` 欄位標明來源（不同頭的 `track_id` 各自編號，不可互相比對）。 |
| `YOLO_MODEL_SEGMENT` | `yolo26s-seg.pt` | `segment` 任務的模型（實例分割）。 |
| `YOLO_MODEL_POSE` | `yolo26s-pose.pt` | `pose` 任務的模型（COCO-17 關鍵點）。 |
| `YOLO_MODEL_OBB` | `yolo26s-obb.pt` | `obb` 任務的模型（旋轉框，DOTA 類別，多為空拍視角）。 |
| `YOLO_MODEL_OPENVOCAB` | `yoloe-26s-seg.pt` | `openvocab` 任務的模型（YOLOE-26）。搭配 `YOLO_CLASSES` 文字提示；`-pf` 變體免提示。此任務**強制 FP32**（YOLOE 的文字嵌入與半精度骨幹會型別衝突）。 |
| `YOLO_MODEL_SEMANTIC` | `yolo26s-sem.pt` | `semantic` 任務的模型（逐像素類別圖）。 |
| `YOLO_MODEL_DEPTH` | `yolo26s-depth.pt` | `depth` 任務的模型（單目深度，公尺）。需 `ultralytics>=8.4.104`。 |
| `RASTER_MAX_SIZE` | `256` | `semantic`／`depth` 圖層編碼成 PNG 前縮到的長邊上限（32–1024）。預設下每幀約 4–7 KB（10 fps 約 46–66 KB/s）；調大更清晰但每幀都要付頻寬。 |
| `YOLO_MODEL` | `yolo26s.pt` | **快速** preset 使用的模型，偏向速度。`DETECT_TASK=detect` 時才會用到。 |
| `YOLO_MODEL_ACCURATE` | `yolo26x.pt` | **精準** preset 使用的模型；越大越準但越慢，首次使用自動下載。可試 `yolov8x-oiv7.pt`（Open Images V7，600 類）或任何自訓 `best.pt`；舊的 `yolov8*.pt` / `yolo11*.pt` 仍完全相容。 |
| `YOLO_CLASSES` | _(空 / empty)_ | 開放詞彙模型（YOLO-World / YOLOE）的逗號分隔提示類別。留空維持模型內建詞彙；需搭配 `-world`/`-worldv2` 模型，封閉集偵測器會忽略。範例：`person,backpack,fire extinguisher`。 |
| `YOLO_DEVICE` | `auto` | `auto`、`cpu`、`0` 或其他 Ultralytics 裝置目標。 |
| `YOLO_HALF` | `1` | 對支援的 CUDA 裝置啟用 FP16（CPU 忽略）。 |
| `YOLO_TRACK` | `1` | 多物件追蹤：跨影格為每個框指派穩定的 `track_id`，Viewer／Recorder 標籤會以 `#id` 前綴顯示，錄影 sidecar 亦記錄。設 `0` 退回逐格獨立偵測。 |
| `YOLO_TRACKER` | `bytetrack.yaml` | 追蹤器設定。Ultralytics 8.4 內建六種：`bytetrack.yaml`（輕量、預設）、`fasttrack.yaml`（最省）、`ocsort.yaml` / `tracktrack.yaml`（遮擋復原較好）、`botsort.yaml` / `deepocsort.yaml`（加入 ReID，最準也最貴）。也可填自訂 `.yaml` 路徑。 |
| `YOLO_END2END` | `auto` | NMS-free（一對一頭部）開關，僅 YOLO26 / YOLOv10 權重具備。`auto` 沿用權重內建設定（在無此頭部的舊模型上完全不影響行為）；`on` 強制端到端輸出、跳過 NMS（`iou` 門檻隨之失效）；`off` 強制走一對多頭部 + NMS，便於 A/B 比較。`/api/status` 的 `detector.end2end_capable` 顯示目前權重是否支援。 |
| `YOLO_MAX_DET` | `300` | 每幀保留的最多偵測框數（1–1000）。與 Ultralytics 預設一致；端到端頭部內部固定至少 300 個候選，故調低只是截斷輸出，不會加速推論。 |
| `YOLO_EXPORT` | _(空 / empty)_ | 首次載入時把 `.pt` 自動 export 成加速格式並改載入產物：`engine`（TensorRT，需 GPU，綁定裝置 + 版本）或 `onnx`。產物快取於 `.pt` 同目錄；export 失敗會退回原 `.pt`（狀態顯示 `last_export_error`）。留空＝直接載入模型名稱（可為預先 export 的 `.engine`/`.onnx`）。 |
| `YOLO_WARMUP` | `0` | 啟動時預熱偵測器。 |
| `VLM_ENABLED` | `0` | 開啟**加法式 VLM 通道**（Florence-2）。與 YOLO 並存、不取代：一個慢速定時 worker 取各相機最新幀，產生開放詞彙偵測框（Phase 1）與場景描述（Phase 2），以獨立 `vlm` 訊息推給 Viewer，Viewer 以「YOLO / VLM」分頁切換。VLM 框沒有信心分數也沒有 `track_id`，故**不進 zones / alerts / history**。device/half 沿用 `YOLO_DEVICE`/`YOLO_HALF`，提示類別沿用 `YOLO_CLASSES`。留空＝關閉。需 `transformers`、`timm`、`einops`（見 `requirements.txt`）。 |
| `VLM_MODEL` | `microsoft/Florence-2-base` | VLM 通道使用的 Florence-2 模型（HF repo id 或本機路徑）。首次載入以 `trust_remote_code` 自 Hugging Face 下載。`Florence-2-large` 較準但較慢。 |
| `VLM_INTERVAL_SEC` | `3.0` | VLM 通道每一輪掃描各相機的間隔秒數（0.5–120）。Florence 一張圖需數百 ms～數秒，故刻意低頻；有 GPU 可調低，純 CPU 建議調高。 |
| `VLM_DETECT_TASK` | _(空 / empty)_ | 覆寫 Florence 偵測 task token。留空＝自動：設了 `YOLO_CLASSES` 用 `<OPEN_VOCABULARY_DETECTION>`（依提示接地），否則 `<OD>`（內建詞彙）。 |
| `VLM_CAPTION` | `1` | VLM 通道每輪同時產生**場景描述**（Phase 2），文字隨 `vlm` 訊息附上，顯示在 Viewer VLM 分頁底部的 HUD 疊字。設 `0` 只出框、不描述（省一次 generate、較快）。 |
| `VLM_CAPTION_TASK` | `<MORE_DETAILED_CAPTION>` | 描述用的 Florence task token。可改 `<CAPTION>`（簡短）或 `<DETAILED_CAPTION>`（中等）。 |
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
| `ALERT_RULES` | _(空 / empty)_ | 規則告警設定，JSON 陣列。每條規則：`name`（必填）、`classes`（逗號字串或陣列，留空＝任意類別）、`min_count`（預設 1）、`min_confidence`（0–1，預設 0）、`cooldown_sec`（預設 `ALERT_COOLDOWN_SEC`）、`zone`（選填）、`camera_id`（選填，留空＝套用所有相機）。留空＝停用。範例：`[{"name":"有人","classes":["person"],"min_count":1,"cooldown_sec":30}]`。也可經 `POST /api/alerts` 即時修改。 |
| `ALERT_COOLDOWN_SEC` | `15` | 未指定 `cooldown_sec` 的規則預設冷卻秒數，避免每格重複觸發。冷卻以「相機 × 規則」計算，前門觸發不會壓住後院的同一條規則。 |
| `ALERT_WEBHOOK_URL` | _(空 / empty)_ | 選用：告警觸發時 POST 的 webhook 端點。留空則僅推送到 Viewer。為防 SSRF，此端點僅能由環境變數設定，不可經執行階段 API 變更。 |
| `ALERT_WEBHOOK_TOKEN` | _(空 / empty)_ | webhook 的選用 bearer token。 |
| `ALERT_WEBHOOK_TIMEOUT` | `5.0` | webhook 逾時（秒）。 |
| `ALERT_WEBHOOK_RETRIES` | `2` | 每次 webhook 發送的重試次數。 |
| `ZONES` | _(空 / empty)_ | ROI 多邊形，JSON 陣列。每個區域：`name`（必填）、`points`（≥3 個 `[x,y]`，正規化 0–1）、`anchor`（`center` 中心點或 `bottom` 底邊中點，預設 `center`）、`camera_id`（選填，預設為第一台相機）。留空＝停用。範例：`[{"name":"門口","points":[[0.1,0.2],[0.4,0.2],[0.4,0.9],[0.1,0.9]]}]`。多相機也可直接給物件形式：`{"front":[...],"back":[...]}`。也可在 Viewer 框選或經 `POST /api/zones` 即時修改。 |
| `EVENT_LOG_ENABLED` | `1` | 啟用偵測歷史：把追蹤到的物件聚合成 per-sighting 紀錄寫入 SQLite，供 `/history` 查詢。設 `0` 關閉。需搭配 `YOLO_TRACK`。 |
| `EVENT_DB_PATH` | `events.db` | 偵測歷史 SQLite 檔路徑（相對路徑以專案根為基準）。 |
| `EVENT_EXPIRY_SEC` | `5.0` | 一個 `track_id` 連續未再出現超過此秒數即視為離開，該筆 sighting 定案並寫入資料庫。多相機下 sighting 以 `(camera_id, track_id)` 為唯一鍵，兩路的 `#1` 不會被併成同一筆。 |
| `METRICS_ENABLED` | `1` | 啟用 Prometheus `GET /metrics` 端點（純讀取，把既有指標以文字格式輸出）。設 `0` 則該端點回 404。 |
| `AUTH_TOKEN` | _(空 / empty)_ | 存取權杖。留空＝完全不啟用驗證（維持現狀）。設定後，頁面 / `/api/*` / `/ws/*` 需帶有效 session cookie（於 `/login` 輸入權杖取得）或 `Authorization: Bearer <token>`。cookie 簽章金鑰由權杖派生，換權杖即令所有既有 session 失效。**（安全規範：驗證機制可實作，但不會代使用者輸入/建立帳密——請自行設定此環境變數。）** |
| `AUTH_SESSION_TTL` | `604800` | session cookie 有效秒數（預設 7 天）。範圍 60 – 2592000（30 天）。 |

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
| `GET /history` | 偵測出現紀錄時間軸 / sighting history timeline. |
| `GET /login` | 登入頁（啟用 `AUTH_TOKEN` 時輸入權杖）/ login page when auth is on. |

**JSON API**

| 路由 / Route | 用途 / Purpose |
| --- | --- |
| `GET /health` | 存活探測 / liveness probe (`{"status": "ok"}`)；驗證豁免。 |
| `POST /api/login` | 以權杖換取 session cookie。Body：`{"token": "..."}`。驗證失敗回 401。 |
| `POST /api/logout` | 清除 session cookie / clear the session cookie. |
| `GET /metrics` | Prometheus 文字格式指標（可接 Grafana）。`METRICS_ENABLED=0` 時回 404。 |
| `GET /api/status` | 完整執行階段快照：串流統計、偵測器狀態、錄影、遠端儲存。頂層數字是全機彙總，`cameras` 為各路明細；`camera_id` 參數選擇 `zones` 要回哪一路。 |
| `GET /api/cameras` | 相機允許清單、預設相機與各路串流狀態 / configured cameras and their stream state. |
| `POST /api/detector/mode` | 切換 preset。Body：`{"mode": "fast"}` 或 `{"mode": "accurate"}`。 |
| `POST /api/detector/task` | 切換偵測頭。Body：`{"task": "segment"}` 只跑一個頭；`{"tasks": ["detect", "pose"]}` 同時跑多個（最多 4 個，`semantic`／`depth` 擇一，違反即 400）。兩者同時給則以 `tasks` 為準。新權重在背景載入，可輪詢 `/api/status` 的 `detector.loaded`（多頭時全部載完才為 `true`）。 |
| `GET /api/detector/config` | 目前偵測器設定 / current detector configuration. |
| `POST /api/detector/config` | 執行階段更新設定（部分更新，只變更傳入的鍵）。可傳 `mode`、`task`、`task_models`（如 `{"pose": "yolo26x-pose.pt"}`）、`fast_model`、`accurate_model`、`classes`、`conf_thresh`、`img_size`、`max_det`、`end2end`（`auto`/`on`/`off`）、`classifier_model`、`classifier_min_conf`、`classifier_max_boxes`。 |
| `GET /api/alerts` | 目前告警規則與觸發狀態 / current alert rules and firing state. |
| `POST /api/alerts` | 執行階段替換告警規則。Body：`{"rules": [...]}` 或直接傳陣列 / replace alert rules at runtime. |
| `GET /api/zones` | 目前 ROI 區域。Query：`camera_id`（省略＝預設相機）；回應含全部相機的 `cameras` 對照。 |
| `POST /api/zones` | 執行階段替換某一路的 ROI 區域。Body：`{"camera_id": "front", "zones": [...]}`；省略 `camera_id` 即預設相機，也可直接傳陣列。只影響指定的那一路。 |
| `GET /api/events` | 查詢偵測出現紀錄。Query：`limit`、`since`、`until`（epoch 秒）、`label`、`zone`、`camera_id` / query sighting history. |
| `POST /api/recordings` | 上傳錄影，依 `X-Yolo-Elf-Storage-Mode` 標頭路由儲存 / upload a recording. |
| `POST /api/recordings/{id}/metadata` | 為錄影附加逐格偵測 sidecar / attach detection sidecar. |
| `GET /api/recordings/{id}` | 下載已儲存的錄影 / download a recording. |
| `GET /api/recordings/{id}/metadata` | 下載錄影的 `.detections.json` sidecar / download the sidecar. |

**WebSockets**

| 路由 / Route | 流向 / Flow | 內容 / Payload |
| --- | --- | --- |
| `/ws/camera?camera_id=` | recorder → server | 二進位 JPEG 畫面；文字 `client_state` 訊息回報儲存模式與錄影狀態。伺服器連線時回覆 `config` 訊息（含 `camera_id` 與相機清單）。`camera_id` 省略＝預設相機；不在 `CAMERAS` 清單內則回一則 `fatal` 錯誤並以 4404 關閉。 |
| `/ws/viewer?camera_id=` | server → viewer | 每格 JSON 中繼資料（含 `camera_id`）後接二進位 JPEG，並可依請求附上 `status` 快照；規則告警觸發時另推送 `alert` 訊息。`camera_id` 省略＝訂閱全部相機。 |

---

## 專案架構說明 | Project Structure

| 路徑 / Path | 內容 / Contents |
| --- | --- |
| `app/main.py` | FastAPI 應用、路由、WebSocket 處理器、偵測工作者。 |
| `app/detector.py` | YOLO 載入、推論、框擷取、第二階段分類器。 |
| `app/stream_state.py` | 串流登錄：每路相機的錄影端 / 畫面佇列 / 指標，檢視端訂閱，與共用的就緒佇列排程。 |
| `app/recordings.py` | 錄影上傳、中繼資料 sidecar、本機儲存。 |
| `app/remote_storage.py` | 偵測中繼資料與錄影的背景上傳佇列。 |
| `app/alerts.py` | 規則告警引擎：規則評估、冷卻去抖動、webhook 背景發送。 |
| `app/zones.py` | ROI 區域引擎：多邊形點內判斷、偵測框區域標記與佔用計數。 |
| `app/events.py` | 偵測歷史：追蹤結果聚合成 per-sighting 紀錄、SQLite 儲存與查詢。 |
| `app/metrics.py` | 把 `/api/status` 快照格式化成 Prometheus 文字指標（純函式）。 |
| `app/auth.py` | 存取控制：共享權杖驗證與簽章 session cookie。 |
| `app/config.py` | 環境變數驅動的 `Settings` 與驗證。 |
| `static/` | 瀏覽器頁面與資產（recorder、viewer、settings）。 |
| `scripts/` | PowerShell / Node / Python 輔助腳本：setup、run、bench、tailscale、靜態建置、模型 export（`export_engine.py`）。 |
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
