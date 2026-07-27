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

## 推論加速（ONNX / TensorRT）

在 GPU 上，把 `.pt` 換成預先 export 的 **TensorRT `.engine`** 通常能明顯降延遲（尤其 accurate preset）；**ONNX `.onnx`** 則在 CPU 或跨硬體時較通用。Ultralytics 的 `YOLO()` 本來就能直接載入這兩種格式，所以有兩條路：

**路線 A — 自己指定產物（推薦，最可控）**：先離線 export，再把 `YOLO_MODEL` 指向產物。

```powershell
# 在「將來要跑伺服器」的那台機器上 export（engine 綁定該 GPU + 驅動/TensorRT 版本）
python scripts\export_engine.py yolov8s.pt --format engine --half --imgsz 1280
$env:YOLO_MODEL = "yolov8s.engine"
.\scripts\run.ps1
```

**路線 B — 讓伺服器自動 export**：設 `YOLO_EXPORT`，首次載入時把 `.pt` export 成指定格式並改載入產物。

```powershell
$env:YOLO_EXPORT = "onnx"   # 或 "engine"
```

- 產物會**快取**在 `.pt` 同目錄，之後啟動直接載入、不重複 export。
- Export **失敗會自動退回原 `.pt`**（不加速但不中斷），錯誤顯示在 `/api/status` 的 `detector.last_export_error`；實際載入了哪個檔看 `detector.loaded_source`。
- `.engine` **無法在無 GPU 的機器 export**，且綁定裝置 + TensorRT/CUDA 版本，換卡或升級驅動要重新 export（刪掉舊 `.engine` 讓它重建，或重跑上面的腳本）。
- FP16（`--half` / `YOLO_HALF=1`）沿用既有半精度設定，僅 CUDA 生效。

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
- **多相機**：追蹤器狀態存在 model 物件內，兩路共用一個實例會把 `track_id` 混在一起。因此第一台相機
  沿用主 model，其餘每台各自持有一份 model 實例——記憶體成本是「額外相機數 × 權重」，`accurate`
  preset（`yolov8x` 等）尤其吃重。`/api/status` 的 `detector.tracker_streams` 就是額外實例數。
  `track_id` 明確**不跨相機共用**：前門的 `#3` 和後院的 `#3` 是兩個不相干的物件。

## 多相機（輕量 NVR）

留空 `CAMERAS` ＝單相機，一切與過去相同。要接多路就給一份允許清單：

```powershell
# id 或 id:顯示名，最多 MAX_CAMERAS 台（預設 4）
$env:CAMERAS = "front:前門,back:後院"
.\scripts\run.ps1
```

- **錄影端**：每台裝置用 `?camera_id=` 宣告自己是哪一路，例如 `/recorder?camera_id=front`。
  不在清單內的 id 會被拒絕（狀態列顯示「camera id rejected」），避免任意 id 灌爆伺服器。
- **檢視端**：`/viewer` 自動變成格狀版面（2 路→並排、3–4 路→九宮格 2×2，依此類推），
  點一格放大、再點回全景；也可用 `/viewer?camera_id=back` 直接釘住單一路（只訂閱該路，省頻寬）。
- **哪些設定是全域、哪些是每路**：
  | 設定 | 範圍 |
  | --- | --- |
  | preset（快速／精準）、`CONF_THRESH`、`IMG_SIZE`、`YOLO_CLASSES`、分類器 | 全域共用 |
  | ROI 區域、告警規則與冷卻、偵測歷史 | 每路獨立 |
  | 錄影 / 遠端上傳 | 目前仍為單一路（第一版未做多相機錄影） |
- **GPU 排程**：所有路共用**一個** worker（不是每路一條執行緒去搶 GPU）。提交端把「哪一路有新畫面」
  推進一個共用就緒佇列，worker 依序取用——天生輪流，且每路各自的單槽佇列保證永遠處理最新影格、
  丟掉舊的。忙不過來時會自然退化成較低的有效 FPS，而不是排隊爆掉。
- **先量測再加相機**：N 路的總需求是各路 FPS 的總和。用 `scripts/bench_detector.py` 量單卡在你的
  `IMG_SIZE`／preset 下每秒能吃幾張，再決定 `MAX_CAMERAS` 與各路的 `FRAME_FPS`。
- **監看**：`/metrics` 除了原本的全機彙總，另有 `camera` 標籤的每路序列
  （`yolo_elf_camera_frames_processed_total{camera="front"}`、`camera_inference_ms`…）。

```powershell
# 每路各自的 ROI 區域，用物件形式一次帶入
$env:ZONES = '{"front":[{"name":"門口","points":[[0.1,0.2],[0.4,0.2],[0.4,0.9],[0.1,0.9]]}],"back":[{"name":"車道","points":[[0.5,0.4],[0.9,0.4],[0.9,0.95],[0.5,0.95]]}]}'
# 只在後院觸發的規則
$env:ALERT_RULES = '[{"name":"後院有人","classes":["person"],"camera_id":"back"}]'
```

## 偵測歷史（回放誰在什麼時候出現）

預設開啟（`EVENT_LOG_ENABLED=1`）：以 `track_id` 為單位，把每個物件在畫面中的存在聚合成一筆「出現紀錄」
（首次／最後出現時間、停留秒數、經過哪些區域、最高信心、幀數），寫入本機 SQLite（`EVENT_DB_PATH`，預設 `events.db`）。
到 `/history` 頁面即可依類別、區域、時間範圍查詢時間軸。

- **需要追蹤**：紀錄以 `track_id` 聚合，所以要 `YOLO_TRACK=1`（預設開）；關掉追蹤就沒有 `track_id` 可聚合。
- **多相機**：紀錄以 `(camera_id, track_id)` 為唯一鍵，兩路的 `#1` 不會被併成同一筆；`/history` 多一個相機下拉與欄位。
  舊資料庫沒有 `camera_id` 欄，首次讀寫會自動 `ALTER TABLE` 補上，舊資料視為預設（第一台）相機。
- **何時定案**：一個 `track_id` 連續未再出現超過 `EVENT_EXPIRY_SEC`（預設 5 秒）就視為離開，該筆紀錄定案並落地。
  停留時間短的場景可調小、想合併短暫遮擋可調大。
- **關閉**：`EVENT_LOG_ENABLED=0`；資料庫檔已列入 `.gitignore`。

```powershell
# 關閉歷史記錄
$env:EVENT_LOG_ENABLED = "0"
# 或改存到別的位置、放寬離開判定到 10 秒
$env:EVENT_DB_PATH = "D:\yolo-elf\events.db"
$env:EVENT_EXPIRY_SEC = "10"
```

## ROI 區域（只看畫面的某一塊）

在 Viewer 面板點「ROI 區域 → 編輯」，直接在畫面上點出多邊形頂點（至少 3 點）、按「完成」命名即可存下；
每個偵測框會標記所屬區域，區域標籤即時顯示佔用數。座標以正規化 0–1 儲存，換相機解析度也不用重畫。

- **判定點**：`anchor` 決定用框的哪個點判斷是否在區域內——`center`（中心，預設，適合一般物件）或 `bottom`（底邊中點，適合人／車站在地面的位置）。
- **搭配告警**：在告警規則加 `"zone":"門口"`，該規則就只計入落在該區域內的框（見下節）。
- **即時修改**：Viewer 框選會 `POST /api/zones`；也可直接呼叫 API 或用 `ZONES` 環境變數開機帶入。
- **多相機**：區域是每路各自一份（兩台相機拍的是不同地點，共用一份沒有意義）。在格狀版面要先點一格
  放大，才知道要畫在哪一路——沒選之前「編輯」按鈕是停用的。`ZONES` 可用物件形式 `{"front":[…]}`
  分路帶入，或在陣列項目上加 `"camera_id"`；不指定就屬於第一台相機（＝原本的單相機語意）。

```powershell
# 開機就帶一個門口區域（正規化座標）
$env:ZONES = '[{"name":"門口","points":[[0.1,0.2],[0.4,0.2],[0.4,0.9],[0.1,0.9]],"anchor":"bottom"}]'
```

## 檢視端過濾與存圖（只在 Viewer 端，不動偵測）

Viewer 右側面板「顯示過濾」提供純前端的檢視工具，只影響這一個瀏覽器分頁的畫面，**不會改變後端偵測、也不影響其他檢視端或歷史紀錄**：

- **類別過濾**：面板會列出串流中出現過的類別 chip，點一下切換顯示／隱藏（隱藏的 chip 會變灰加刪除線）。新出現的類別預設顯示，你的切換會跨影格保留。
- **最低信心滑桿**：把低於門檻的框先藏起來，快速壓掉雜訊，範圍 0–95%。這是檢視端的視覺過濾，和後端的 `CONF_THRESH`（真正決定要不要送框）不同層。
- **存圖**：按「存圖」把當前含框畫面存成 PNG。輸出用畫面的原始解析度（非拉伸的舞台尺寸），且只畫出目前沒被過濾掉的框與區域，檔名為 `yolo-elf-frame-<frame_id>.png`。

> 這些過濾是顯示層；`Boxes` 指標仍顯示偵測器實際找到的框數，方便對照被藏起來的數量。

## 規則告警（偵測到就通知）

當偵測命中設定的規則時觸發告警：即時推送到 Viewer（右上 `alerts` 狀態燈會亮、若已授權會跳瀏覽器通知），
並可送出 webhook 給外部系統（Slack、Home Assistant、自建 endpoint…）。

規則以 JSON 陣列設定，每條欄位：

- `name`（必填）：規則名稱，會出現在告警訊息與狀態中。
- `classes`：要比對的偵測標籤（逗號字串或陣列）；留空＝任意類別。
- `min_count`：命中框數達到才觸發（預設 `1`）。
- `min_confidence`：低於此信心的框不計入（預設 `0`）。
- `cooldown_sec`：同一規則兩次觸發的最短間隔，避免每格狂噴（未填用 `ALERT_COOLDOWN_SEC`，預設 15 秒）。
  冷卻以「相機 × 規則」計算：前門剛觸發不會把後院的同一條規則一起壓住。
- `zone`（選用）：只計入落在指定 ROI 區域內的框（見上一節）；留空＝整個畫面。
- `camera_id`（選用）：只在指定的那一路相機評估；留空＝所有相機都套用。

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

## 用 Prometheus / Grafana 監控

`GET /metrics` 把 `/api/status` 裡既有的指標以 Prometheus 文字格式輸出，不需額外套件，也不需要有錄影端在線就能抓（沒有串流時多數值為 0）。

- **counter**（單調遞增，名稱以 `_total` 結尾）：`yolo_elf_frames_processed_total`、`yolo_elf_frames_dropped_total`、`yolo_elf_alerts_fired_total`、`yolo_elf_sightings_written_total`、遠端上傳的 `yolo_elf_remote_records_uploaded_total`…
- **gauge**（瞬時值）：`yolo_elf_process_fps`、`yolo_elf_inference_ms`、`yolo_elf_avg_total_latency_ms`、`yolo_elf_queue_depth`、`yolo_elf_viewers`、`yolo_elf_active_sightings`、`yolo_elf_alert_rules`、`yolo_elf_zones`…
- **info**：`yolo_elf_detector_info{mode,model,device}` 值恆為 1，用 label 帶出目前 preset。為避免 label 基數爆炸，`track_id` 這類高基數維度**不會**當 label。
- **每路相機**：帶 `camera` label 的序列 `yolo_elf_camera_frames_received_total`、`camera_frames_processed_total`、`camera_frames_dropped_total`、`camera_process_fps`、`camera_inference_ms`、`camera_total_latency_ms`、`camera_connected`，加上總數 `yolo_elf_cameras`。基數受 `MAX_CAMERAS` 上限保護，所以 `camera` 當 label 是安全的。上面那些不帶 label 的序列仍是全機彙總。

抓取設定（`prometheus.yml`）：

```yaml
scrape_configs:
  - job_name: yolo-elf
    static_configs:
      - targets: ["127.0.0.1:8766"]
```

要關掉端點（例如公開部署時不想外露指標）：設 `METRICS_ENABLED=0`，`/metrics` 會回 404。端點本身不含驗證，遠端暴露時請靠反向代理或防火牆限制來源（存取控制見 roadmap #7）。

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

## VLM 語意通道（Florence-2）

需要「用自然語言描述沒見過的東西」或「讓機器說出畫面在發生什麼」時，可開啟選用的 **VLM 通道**。
它是**加法式**的：跟 YOLO 並存、不取代——YOLO 仍逐格跑（信心 / 追蹤 / 歷史全開），VLM 只是多一條
慢速通道，Viewer 上以「YOLO / VLM」分頁切換。

```powershell
$env:VLM_ENABLED = "1"
$env:VLM_INTERVAL_SEC = "3"          # 每 3 秒掃一輪各相機
$env:YOLO_CLASSES = "person,backpack,fire extinguisher"   # 有值→開放詞彙接地；留空→<OD> 內建詞彙
.\scripts\run.ps1
```

先裝相依（Florence-2 的 `trust_remote_code` 會 import 這些）：

```powershell
.\.venv\Scripts\python.exe -m pip install "transformers>=4.44" "timm>=1.0" "einops>=0.8"
```

### 運作方式

- 一個獨立的定時 worker 每 `VLM_INTERVAL_SEC` 秒取**各相機最新處理過的幀**（不跟 YOLO 搶就緒佇列），
  跑一次 Florence-2，把結果以獨立的 `{"type":"vlm",...}` 訊息廣播給 Viewer。
- **Phase 1 — 開放詞彙偵測框**。task token 由 `VLM_DETECT_TASK` 決定，留空時：有 `YOLO_CLASSES`
  用 `<OPEN_VOCABULARY_DETECTION>`（依提示接地），否則 `<OD>`（模型內建詞彙）。
- **Phase 2 — 場景描述**。`VLM_CAPTION=1`（預設）時每輪同時跑一次描述 task（`VLM_CAPTION_TASK`，
  預設 `<MORE_DETAILED_CAPTION>`），文字隨 `vlm` 訊息附上，顯示在 Viewer VLM 分頁底部的 HUD 疊字。
  描述是 best-effort：它失敗不會拖累框（框照出，錯誤記在 `/api/status` 的 `vlm.last_error`）。
  只想要框、不要描述就設 `VLM_CAPTION=0`（省一次 generate）。
- device / 半精度沿用 `YOLO_DEVICE` / `YOLO_HALF`；`/api/status` 的 `vlm` 區塊顯示 `loaded`、
  `detect_task`、`caption_task`、`last_inference_ms`、`last_caption`、`last_load_error` 等。

### 固有限制（不是實作取捨，是 Florence 的特性）

- **沒有信心分數**：每個 VLM 框帶固定哨兵值 `1.0`，Viewer 的「最低信心」滑桿對 VLM 通道等於無效。
- **沒有追蹤 / 歷史**：Florence 不吐 `track_id`，所以 VLM 框**不進 zones / alerts / history**；那些功能
  只吃 YOLO 通道。要語意事件請搭配 YOLO 追蹤或 webhook。
- **慢**：一張圖數百 ms～數秒。這就是它自成一條低頻通道、而非塞進逐格路徑的原因。純 CPU 請把
  `VLM_INTERVAL_SEC` 調高。
- `.engine` / `.onnx` 的加速 export 對 VLM 不適用（那條路徑只處理 YOLO 的 `.pt`）。

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

## 存取控制（遠端使用前先開）

預設**沒有任何驗證**：只要能連到這台伺服器（例如同一個 tailnet），任何人都能觀看、搶錄影端、或讀 `/metrics`。要遠端使用前，設一個共享權杖：

```powershell
# 自己產一個夠長的隨機權杖（別用範例值），設進環境變數後再啟動伺服器
$env:AUTH_TOKEN = "換成你自己的長隨機字串"
```

啟用後：

- **瀏覽器**：第一次進任何頁面會被導到 `/login`，輸入權杖 → 伺服器發一個 HttpOnly、簽章、預設 7 天（`AUTH_SESSION_TTL`）的 session cookie。之後頁面、`/api/*`、`/ws/*` 全靠這個 cookie；權杖不會出現在網址或 access log。cookie 過期時前端會自動導回 `/login`。
- **程式端 / 抓取器**：改用 `Authorization: Bearer <AUTH_TOKEN>`（Prometheus 也可這樣抓 `/metrics`）。
- **豁免**：`/health`（存活探測）與 `/login` 及其靜態資源永遠可存取；靜態 demo（GitHub Pages）無後端，本來就沒有驗證。
- **換權杖**＝踢掉所有人：cookie 簽章金鑰由 `AUTH_TOKEN` 派生，改權杖會讓所有既有 session 立刻失效。

安全備註：權杖請自己設定，別寫進版本控制或貼進聊天；伺服器不會把它寫進 log。此為單一共享權杖，不是逐使用者帳號；若要多帳號或稽核，需接反向代理的 SSO。

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
