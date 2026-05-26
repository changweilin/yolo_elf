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
.\scripts\bench.ps1 -Frames 20 -Warmup 3 -Device 0 -ImgSize 960 -Quality 0.85
.\scripts\bench.ps1 -Frames 20 -Warmup 3 -Device cpu -ImgSize 960 -Quality 0.85
```

## 提高辨識率的優先順序

1. 換更大的模型，例如 `yolov8s.pt`、`yolov8m.pt`，或使用針對你的目標類別訓練的 `best.pt`。
2. 提高輸入解析度與 YOLO `IMG_SIZE`，小物件通常會更容易被看見。
3. 提高 `JPEG_QUALITY`，避免壓縮破壞細節。
4. 降低 `CONF_THRESH` 會提高召回率，但也會增加誤判。

目前預設已偏向辨識率：

```powershell
$env:CAPTURE_WIDTH = "1280"
$env:CAPTURE_HEIGHT = "720"
$env:JPEG_QUALITY = "0.85"
$env:IMG_SIZE = "960"
$env:CONF_THRESH = "0.25"
$env:YOLO_DEVICE = "0"
.\scripts\run.ps1
```

若你需要更快但可以接受漏檢，可把 `IMG_SIZE` 改回 `640`、`JPEG_QUALITY`
改回 `0.65`。
