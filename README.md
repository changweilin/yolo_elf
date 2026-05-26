# YOLO Elf

手機透過 Tailscale HTTPS 開啟 `/phone`，瀏覽器用相機擷取 JPEG 幀並以 WebSocket 傳到 Windows 電腦；電腦端用 YOLO 偵測，`/viewer` 顯示即時影像與框線。

## 快速開始

```powershell
# 建議 RTX / CUDA 使用這個；第一次會下載 PyTorch、Ultralytics 與 YOLO 權重
.\scripts\setup.ps1 -Cuda

# 啟動本機服務
.\scripts\run.ps1
```

開電腦端 viewer：

```text
http://127.0.0.1:8766/viewer
```

若 `8766` 已被占用，可改用 Node 背景啟動器：

```powershell
node .\scripts\start-server.mjs 8767
```

把服務放到 tailnet：

```powershell
.\scripts\tailscale-serve.ps1 -Port 8766
```

手機登入同一個 Tailscale tailnet 後，開：

```text
https://<pc-tailnet-name>.<tailnet>.ts.net/phone
```

瀏覽器相機 API 需要安全來源；手機端請用 Tailscale Serve 提供的 HTTPS URL，不要用一般 HTTP tailnet IP。

## 設定

可用環境變數：

| 名稱 | 預設 | 用途 |
| --- | --- | --- |
| `YOLO_MODEL` | `yolov8n.pt` | 模型權重，可換成 `.pt` 或 `.onnx` 路徑 |
| `YOLO_DEVICE` | `auto` | `auto` 會優先用 CUDA，或手動設 `cpu`、`0` |
| `YOLO_HALF` | `0` | CUDA 推論時啟用 FP16/half precision |
| `YOLO_WARMUP` | `0` | 啟動時預載模型並先跑 warmup |
| `YOLO_WARMUP_RUNS` | `1` | warmup 推論次數 |
| `CONF_THRESH` | `0.35` | 偵測信心門檻 |
| `IMG_SIZE` | `640` | YOLO 推論尺寸 |
| `FRAME_FPS` | `10` | 手機端預設傳送 FPS |
| `CAPTURE_WIDTH` | `960` | 手機端傳送影像寬度 |
| `CAPTURE_HEIGHT` | `540` | 手機端傳送影像高度 |
| `JPEG_QUALITY` | `0.65` | 手機端 JPEG 品質 |
| `MAX_FRAME_BYTES` | `5242880` | 單張幀大小上限 |

手機端會把 `FRAME_FPS` 視為上限，並依最近推論時間與 WebSocket buffer 自動降低實際送出 FPS，避免後端 queue 持續丟幀。

Ultralytics 設定會寫到專案內的 `.ultralytics` 目錄，避免 Windows 權限或 sandbox 阻擋 `%APPDATA%`。

範例：使用自訂模型。

```powershell
$env:YOLO_MODEL = "C:\models\best.pt"
$env:YOLO_DEVICE = "0"
.\scripts\run.ps1
```

## 端點

| 端點 | 說明 |
| --- | --- |
| `GET /phone` | 手機相機串流頁 |
| `GET /viewer` | 電腦端偵測結果 overlay |
| `GET /health` | 健康檢查 |
| `GET /api/status` | 即時狀態、frame 計數、FPS、延遲與模型/GPU 狀態 |
| `WS /ws/camera` | 手機上傳 JPEG 幀並接收偵測結果 |
| `WS /ws/viewer` | viewer 接收偵測 metadata 與 binary JPEG 幀 |

偵測結果 payload：

```json
{
  "frame_id": 1,
  "width": 960,
  "height": 540,
  "inference_ms": 12.34,
  "boxes": [
    {
      "xyxy": [10.0, 20.0, 120.0, 180.0],
      "class_id": 0,
      "label": "person",
      "confidence": 0.91
    }
  ]
}
```

## 測試

```powershell
.\scripts\run-tests.ps1
```

## Benchmark

```powershell
.\scripts\bench.ps1 -Frames 30 -Warmup 3
.\scripts\bench.ps1 -Frames 30 -Warmup 3 -Device 0 -Half
```

## Remote storage

Set `REMOTE_STORAGE_URL` to enable background uploads of processed detections to a
remote HTTP endpoint. When it is unset, remote storage is disabled.

```powershell
$env:REMOTE_STORAGE_URL = "https://storage.example/events"
$env:REMOTE_STORAGE_TOKEN = "optional-bearer-token"
$env:REMOTE_STORAGE_INCLUDE_FRAME = "0"
.\scripts\run.ps1
```

Each upload is a JSON `POST` with `source`, `frame_id`, `received_at`,
`received_at_iso`, and `detection`. If `REMOTE_STORAGE_INCLUDE_FRAME=1`, the
payload also includes a JPEG `frame` object with `content_type`, `byte_length`,
and base64 data. Upload status is exposed at `GET /api/status` under
`remote_storage`.

Optional settings:

| Name | Default | Description |
| --- | --- | --- |
| `REMOTE_STORAGE_URL` | empty | Remote HTTP endpoint. Empty disables uploads. |
| `REMOTE_STORAGE_TOKEN` | empty | Optional bearer token for the `Authorization` header. |
| `REMOTE_STORAGE_INCLUDE_FRAME` | `0` | Include base64 JPEG frames in each upload. |
| `REMOTE_STORAGE_QUEUE_SIZE` | `100` | Pending upload queue size. Oldest records are dropped when full. |
| `REMOTE_STORAGE_TIMEOUT` | `5.0` | HTTP timeout in seconds. |
| `REMOTE_STORAGE_RETRIES` | `2` | Retry count after a failed upload. |

## 常見狀況

- `tailscale status` 顯示 access denied：Windows 上 Tailscale LocalAPI 可能需要系統權限；Serve 仍可用 `tailscale serve --bg 8766` 設定，必要時用系統管理員 PowerShell 執行。
- 手機沒有跳相機權限：確認 URL 是 `https://...ts.net/phone`，且手機瀏覽器允許相機。
- 第一次偵測很慢：Ultralytics 會下載 `yolov8n.pt` 並初始化 CUDA；第二次通常會快很多。
- GPU 沒有被用到：在 `/api/status` 看 `cuda_available`，或重新執行 `.\scripts\setup.ps1 -Cuda`。
