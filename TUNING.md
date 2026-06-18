# YOLO Elf GPU 與辨識率調校

## 先判斷是不是 GPU 問題

GPU 主要影響推論速度，不會直接讓同一個模型變得更準。若 `/api/status`
顯示：

- `cuda_available: true`
- `resolved_device: 0`
- `cuda_device_name` 有列出顯卡

代表 `YOLO_DEVICE=auto` 會跑在 CUDA GPU。若想強制指定 GPU：

```powershell
$env:YOLO_DEVICE = "0"
.\scripts\run.ps1
```

可用 benchmark 快速比對 GPU 與 CPU：

```powershell
.\scripts\bench.ps1 -Frames 20 -Warmup 3 -Device 0 -ImgSize 1280 -Quality 0.9
.\scripts\bench.ps1 -Frames 20 -Warmup 3 -Device cpu -ImgSize 1280 -Quality 0.9
```

## 快速 / 精準模式切換

不必重啟即可在兩個預設之間切換：在 Viewer 右側面板按 **快速 / 精準**，或呼叫
`POST /api/detector/mode`（body 為 `{"mode": "fast"}` 或 `{"mode": "accurate"}`）。

- **快速 (fast)**：使用 `YOLO_MODEL`（預設 `yolov8s.pt`），速度優先。
- **精準 (accurate)**：使用 `YOLO_MODEL_ACCURATE`（預設 `yolov8x.pt`，YOLOv8 系列中最準），
  首次切換會自動下載權重；想要最新、最高準確度可設成 `yolo11x.pt`。

起始模式由 `DETECT_MODE` 決定（預設 `fast`）。切到精準模式後，下一張影格才會載入較大的模型，
因此第一張的延遲會略高，之後維持快取不再重載。

## 換用更多類別 / 專用模型

`YOLO_MODEL` 與 `YOLO_MODEL_ACCURATE` 可指向任何 Ultralytics 格式的偵測權重，框體格式相容、
標籤會自動跟著模型的 `names` 變動，因此多數情況只要改環境變數、不需改程式碼：

- **更多一般類別**：`yolov8x-oiv7.pt`（Open Images V7，600 類，COCO 只有 80 類），首次使用自動下載。
- **專用模型**：Hugging Face / Ultralytics Hub 上有人臉、車牌、工地安全帽 (PPE)、火災煙霧、
  文件版面等現成 `.pt`，下載後把對應環境變數指到該檔即可。
- **自訂類別**：用你自己的資料訓練出的 `best.pt` 同樣直接指過去。

⚠️ 影像分類 (`-cls`)、分割 (`-seg`)、姿態 (`-pose`)、旋轉框 (`-obb`) 模型的輸出格式不同，
直接替換會讓框體解析失效，需要另外改 `detector.py`。

### 開放詞彙（自己打字決定要偵測什麼）

YOLO-World / YOLOE 可用文字 prompt 指定任意類別，不必重新訓練：

1. 把模型設成 world 權重，例如 `YOLO_MODEL_ACCURATE = "yolov8x-worldv2.pt"`。
2. 用 `YOLO_CLASSES` 以逗號分隔列出要偵測的類別，例如：

```powershell
$env:YOLO_MODEL_ACCURATE = "yolov8x-worldv2.pt"
$env:YOLO_CLASSES = "person,backpack,fire extinguisher"
```

載入時會自動呼叫 `set_classes` 套用詞彙；`/api/status` 的 `open_vocabulary` 會顯示是否生效，
`configured_classes` 顯示目前設定的類別。若 `YOLO_CLASSES` 指到的不是 world 模型，會自動忽略並沿用內建類別。

## 三種設定方式

辨識參數（模式、模型、開放詞彙類別、信心門檻、影像尺寸）有三個入口，依「是否需要重啟」區分：

### 1. 設定頁面（免重啟，即時生效）

開 `/settings`（Recorder／Viewer 右上角點 **Settings** 按鈕）。填好欄位按 **套用**，
透過 `POST /api/detector/config` 即時改 detector，不必重啟伺服器。狀態顯示「已套用」即成功；
換模型後下一張影格才載入新權重，第一張會略慢。頁面上有完整「設置流程」說明。

> 注意：設定頁的變更是 runtime 狀態，伺服器重啟後會回到環境變數／預設值。要長期保留請用下面兩種。

### 2. `run.ps1` 啟動參數（開機就帶設定）

```powershell
.\scripts\run.ps1 -DetectMode accurate -AccurateModel "yolov8x-worldv2.pt" `
  -Classes "person,backpack,fire extinguisher" -ConfThresh 0.25 -ImgSize 1280
```

可用參數：`-DetectMode`（fast/accurate）、`-FastModel`、`-AccurateModel`、`-Classes`、
`-ConfThresh`、`-ImgSize`。沒帶的參數維持環境變數或預設值。

### 3. 環境變數（永久 / CI）

`DETECT_MODE`、`YOLO_MODEL`、`YOLO_MODEL_ACCURATE`、`YOLO_CLASSES`、`CONF_THRESH`、`IMG_SIZE`，
細節見 `README.md` 的環境變數表。

## 提高辨識率的優先順序

1. 換更大的模型，例如 `yolov8s.pt`、`yolov8m.pt`，或使用針對你的目標類別訓練的 `best.pt`。
2. 提高輸入解析度與 YOLO `IMG_SIZE`，小物件通常會更容易被看見。
3. 提高 `JPEG_QUALITY`，避免壓縮破壞細節。
4. 降低 `CONF_THRESH` 會提高召回率，但也會增加誤判。

目前預設已偏向辨識率（等同下列設定，可直接執行 `.\scripts\run.ps1`）：

```powershell
$env:YOLO_MODEL = "yolov8s.pt"
$env:YOLO_HALF = "1"
$env:CAPTURE_WIDTH = "1920"
$env:CAPTURE_HEIGHT = "1080"
$env:JPEG_QUALITY = "0.9"
$env:IMG_SIZE = "1280"
$env:CONF_THRESH = "0.2"
$env:YOLO_DEVICE = "0"
.\scripts\run.ps1
```

若你需要更快但可以接受漏檢，可把 `IMG_SIZE` 改回 `640`、`JPEG_QUALITY`
改回 `0.65`、模型換回 `yolov8n.pt`。

## 透過 Tailscale 時的取捨

辨識「在桌機」運算，Tailscale 只是把 JPEG 來回傳輸，不影響「每幀準確度」。
但較大的擷取解析度（1920×1080）會增加每幀位元組數，在 Tailscale 走 DERP
relay 或行動網路時可能塞滿 WebSocket buffer，使 `adaptiveFps` 自動降到 1 fps、
丟棄影格，畫面上的框會延遲、跟不上物體（看起來像「辨識變差」，其實是過時的框）。

判斷方式：

- 看手機頁面的 adaptive 狀態列。若出現 `1.0 fps / cap 1.0 / buffer` 或 `/ socket`，
  代表是網路瓶頸，不是辨識精度。
- 用 `tailscale status` 確認是 `direct` 直連還是 `relay`（走中繼會放大延遲）。

若卡頓，先降 `CAPTURE_WIDTH/HEIGHT`（例如回 1280×720）或 `JPEG_QUALITY`，
而不是降 `IMG_SIZE`——`IMG_SIZE` 不影響傳輸量，只影響桌機端推論成本。

`CONF_THRESH=0.2` 是最容易回退的旋鈕：若誤判（多餘的框）變多，先調回 `0.25`。
