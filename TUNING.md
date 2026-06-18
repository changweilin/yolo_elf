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
