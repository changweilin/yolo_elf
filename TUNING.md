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

## 物件追蹤（track_id）

預設開啟（`YOLO_TRACK=1`）：偵測改用 Ultralytics 內建追蹤器（`model.track(persist=True)`），
跨影格為每個物件維持穩定 `track_id`。Viewer／Recorder 疊圖以 `#id` 前綴標示，錄影的
`.detections.json` sidecar 每個框也會帶 `track_id`，方便事後統計進出、停留時間、軌跡。

- **追蹤器**：`YOLO_TRACKER` 預設 `bytetrack.yaml`（輕量、即時優先）；改 `botsort.yaml` 可加入
  ReID（外觀特徵）在遮擋後更容易接回同一 id，但成本較高。
- **關閉**：`YOLO_TRACK=0`（或 `run.ps1 -Track off`）退回逐格獨立偵測，`track_id` 為 `null`。
- 追蹤狀態是每個模型各自維護：切換 快速／精準 模式時 id 不會延續。這是啟動時的設定，
  不透過設定頁即時切換（避免追蹤器狀態殘留造成誤判）。

## ROI 區域（只看畫面的某一塊）

在 Viewer 面板點「ROI 區域 → 編輯」，直接在畫面上點出多邊形頂點（至少 3 點）、按「完成」命名即可存下；
每個偵測框會標記所屬區域，區域標籤即時顯示佔用數。座標以正規化 0–1 儲存，換相機解析度也不用重畫。

- **判定點**：`anchor` 決定用框的哪個點判斷是否在區域內——`center`（中心，預設，適合一般物件）或 `bottom`（底邊中點，適合人／車站在地面的位置）。
- **搭配告警**：在告警規則加 `"zone":"門口"`，該規則就只計入落在該區域內的框（見下節）。
- **即時修改**：Viewer 框選會 `POST /api/zones`；也可直接呼叫 API 或用 `ZONES` 環境變數開機帶入。

```powershell
# 開機就帶一個門口區域（正規化座標）
$env:ZONES = '[{"name":"門口","points":[[0.1,0.2],[0.4,0.2],[0.4,0.9],[0.1,0.9]],"anchor":"bottom"}]'
```

## 規則告警（偵測到就通知）

當偵測命中設定的規則時觸發告警：即時推送到 Viewer（右上 `alerts` 狀態燈會亮、若已授權會跳瀏覽器通知），
並可送出 webhook 給外部系統（Slack、Home Assistant、自建 endpoint…）。

規則以 JSON 陣列設定，每條欄位：

- `name`（必填）：規則名稱，會出現在告警訊息與狀態中。
- `classes`：要比對的偵測標籤（逗號字串或陣列）；留空＝任意類別。
- `min_count`：命中框數達到才觸發（預設 `1`）。
- `min_confidence`：低於此信心的框不計入（預設 `0`）。
- `cooldown_sec`：同一規則兩次觸發的最短間隔，避免每格狂噴（未填用 `ALERT_COOLDOWN_SEC`，預設 15 秒）。
- `zone`（選用）：只計入落在指定 ROI 區域內的框（見上一節）；留空＝整個畫面。

```powershell
# 一有人就通知，最短間隔 30 秒
$env:ALERT_RULES = '[{"name":"有人","classes":["person"],"cooldown_sec":30}]'
# 選用：把告警 POST 到 webhook（含 bearer token）
$env:ALERT_WEBHOOK_URL = "https://hooks.example/yolo"
$env:ALERT_WEBHOOK_TOKEN = "secret"
```

不必重啟也能改規則：`POST /api/alerts`，body 為 `{"rules":[...]}`（或直接傳陣列）；
`GET /api/alerts`、`/api/status` 的 `alerts` 欄位可看目前規則、觸發次數與最近事件。
webhook 端點基於安全（SSRF 防護）**只能**用環境變數設定，不能透過執行階段 API 變更。

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
