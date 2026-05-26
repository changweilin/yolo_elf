# YOLO Elf

YOLO Elf is a small FastAPI web app for streaming phone camera frames to a local
YOLO detector and viewing detection boxes in a browser.

## Local Run

```powershell
.\scripts\setup.ps1
.\scripts\run.ps1
```

Open the viewer:

```text
http://127.0.0.1:8766/viewer
```

Open the phone camera page:

```text
http://127.0.0.1:8766/phone
```

For camera capture from another device, use an HTTPS URL such as a Tailscale
Serve URL. Browsers block camera access on non-local HTTP origins.

## GitHub Actions Web Demo

The GitHub Actions workflow builds a static, privacy-safe demo for GitHub Pages:

```powershell
npm run build:github-pages
```

The build writes `dist/` with:

- `index.html`: static viewer demo
- `viewer/index.html`: viewer route
- `phone/index.html`: phone route
- `static/`: shared assets

In this static demo, privacy-sensitive live features are frozen:

- camera access is disabled
- WebSocket streaming is disabled
- remote uploads are disabled
- detection boxes are rendered from a synthetic demo frame

On pushes to the repository default branch, or on manual `workflow_dispatch`,
the workflow uploads `dist/` to GitHub Pages.

## Configuration

Common environment variables:

| Name | Default | Description |
| --- | --- | --- |
| `YOLO_MODEL` | `yolov8n.pt` | Model path. |
| `YOLO_DEVICE` | `auto` | `auto`, `cpu`, `0`, or another Ultralytics device target. |
| `YOLO_HALF` | `0` | Enables FP16 for supported CUDA devices. |
| `YOLO_WARMUP` | `0` | Warms the detector during startup. |
| `CONF_THRESH` | `0.25` | Detection confidence threshold. |
| `IMG_SIZE` | `960` | Detector image size. |
| `FRAME_FPS` | `10` | Requested phone capture FPS. |
| `CAPTURE_WIDTH` | `1280` | Capture width. |
| `CAPTURE_HEIGHT` | `720` | Capture height. |
| `JPEG_QUALITY` | `0.85` | JPEG quality sent over WebSocket. |
| `MAX_FRAME_BYTES` | `5242880` | Maximum accepted frame size. |

Remote storage is disabled unless `REMOTE_STORAGE_URL` is set. If enabled, the
server posts detection metadata in the background; frames are included only when
`REMOTE_STORAGE_INCLUDE_FRAME=1`.

## Tests

```powershell
.\scripts\run-tests.ps1
```

The test script runs Python tests, checks Python syntax for the benchmark script,
and runs `node --check` against browser/build JavaScript.
