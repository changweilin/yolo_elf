const image = document.querySelector("#viewerImage");
const overlay = document.querySelector("#viewerOverlay");
const emptyState = document.querySelector("#emptyState");
const viewerSocketStatus = document.querySelector("#viewerSocketStatus");
const cameraLinkStatus = document.querySelector("#cameraLinkStatus");
const phoneStorageStatus = document.querySelector("#phoneStorageStatus");
const modelStatus = document.querySelector("#modelStatus");
const storageStatus = document.querySelector("#storageStatus");
const frameMetric = document.querySelector("#frameMetric");
const boxesMetric = document.querySelector("#boxesMetric");
const inferenceMetric = document.querySelector("#inferenceMetric");
const droppedMetric = document.querySelector("#droppedMetric");
const recordingMetric = document.querySelector("#recordingMetric");
const uploadMetric = document.querySelector("#uploadMetric");
const errorLine = document.querySelector("#errorLine");
const modeGroup = document.querySelector("#modeGroup");
const modeButtons = modeGroup
  ? Array.from(modeGroup.querySelectorAll("[data-detect-mode]"))
  : [];

const moduleUrl = new URL(import.meta.url);
const demoMode =
  window.YOLO_ELF_DEMO_MODE === true || moduleUrl.searchParams.get("demo") === "1";

const demoDetection = {
  frame_id: 42,
  width: 1280,
  height: 720,
  inference_ms: 18.6,
  boxes: [
    { xyxy: [124, 137, 392, 535], class_id: 0, label: "monitor", confidence: 0.88 },
    { xyxy: [489, 302, 633, 514], class_id: 1, label: "bottle", confidence: 0.76 },
    { xyxy: [759, 219, 1062, 521], class_id: 2, label: "package", confidence: 0.93 },
  ],
  error: "",
};

const state = {
  ws: null,
  latestDetection: null,
  pendingFrame: null,
  imageUrl: null,
  reconnectTimer: null,
};

function staticAsset(name) {
  return new URL(name, import.meta.url).href;
}

function socketUrl(path) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${path}`;
}

function setChip(element, text, tone) {
  element.textContent = text;
  element.classList.remove("good", "warn", "bad");
  element.classList.add(tone);
}

function connectViewer() {
  if (demoMode) {
    renderDemoViewer();
    return;
  }

  setChip(viewerSocketStatus, "viewer connecting", "warn");
  const ws = new WebSocket(socketUrl("/ws/viewer"));
  ws.binaryType = "blob";
  state.ws = ws;

  ws.addEventListener("open", () => {
    setChip(viewerSocketStatus, "viewer connected", "good");
  });

  ws.addEventListener("message", (event) => {
    if (typeof event.data !== "string") {
      renderFrameBytes(event.data);
      return;
    }

    const payload = JSON.parse(event.data);
    if (payload.type === "frame") {
      state.pendingFrame = payload;
      return;
    }
    if (payload.type === "status") {
      renderStatus(payload.status);
    }
  });

  ws.addEventListener("close", () => {
    if (state.ws === ws) {
      state.ws = null;
    }
    state.pendingFrame = null;
    setChip(viewerSocketStatus, "viewer offline", "bad");
    state.reconnectTimer = setTimeout(connectViewer, 1200);
  });

  ws.addEventListener("error", () => {
    setChip(viewerSocketStatus, "viewer error", "bad");
  });
}

function renderDemoViewer() {
  setChip(viewerSocketStatus, "viewer demo", "warn");
  setChip(cameraLinkStatus, "phone frozen", "warn");
  setChip(phoneStorageStatus, "storage frozen", "warn");
  setChip(modelStatus, "demo snapshot", "warn");
  setChip(storageStatus, "storage frozen", "warn");
  for (const button of modeButtons) {
    button.disabled = true;
  }
  droppedMetric.textContent = "0";
  recordingMetric.textContent = "0";
  uploadMetric.textContent = "0";
  state.latestDetection = demoDetection;
  image.src = staticAsset("demo-frame.svg");
  emptyState.hidden = true;
  renderMetrics(demoDetection);
  renderError("");
}

function renderFrameBytes(data) {
  const frame = state.pendingFrame;
  if (!frame) {
    return;
  }
  state.pendingFrame = null;

  const blob =
    data instanceof Blob
      ? data
      : new Blob([data], { type: frame.content_type || "image/jpeg" });
  const nextUrl = URL.createObjectURL(blob);
  const previousUrl = state.imageUrl;
  state.imageUrl = nextUrl;
  image.src = nextUrl;
  if (previousUrl) {
    URL.revokeObjectURL(previousUrl);
  }

  state.latestDetection = frame.detection;
  emptyState.hidden = true;
  renderMetrics(frame.detection);
  renderError(frame.detection.error || "");
}

function releaseImageUrl() {
  if (state.imageUrl) {
    URL.revokeObjectURL(state.imageUrl);
    state.imageUrl = null;
  }
}

function storageModeLabel(mode) {
  if (mode === "remote") {
    return "remote";
  }
  if (mode === "both") {
    return "local + remote";
  }
  if (mode === "local") {
    return "local";
  }
  return "—";
}

function renderPhoneStorage(status) {
  if (!phoneStorageStatus) {
    return;
  }
  if (!status.camera_connected) {
    setChip(phoneStorageStatus, "phone storage idle", "warn");
    return;
  }
  const label = storageModeLabel(status.camera_storage_mode);
  if (status.camera_recording) {
    setChip(phoneStorageStatus, `REC · ${label}`, "bad");
  } else {
    setChip(phoneStorageStatus, `ready · ${label}`, "good");
  }
}

function renderStatus(status) {
  setChip(
    cameraLinkStatus,
    status.camera_connected ? "phone connected" : "phone idle",
    status.camera_connected ? "good" : "warn",
  );
  renderPhoneStorage(status);
  renderModel(status.detector || {});
  droppedMetric.textContent = String(status.frames_dropped ?? "-");
  const recordings = status.recordings || {};
  const remote = status.remote_storage || {};
  recordingMetric.textContent = String(recordings.recordings_saved ?? 0);
  uploadMetric.textContent = String((remote.records_uploaded ?? 0) + (remote.recordings_uploaded ?? 0));
  if (remote.enabled) {
    setChip(storageStatus, remote.last_error ? "storage error" : "remote storage on", remote.last_error ? "bad" : "good");
  } else {
    setChip(storageStatus, "remote storage off", "warn");
  }
  if (status.last_error) {
    renderError(status.last_error);
  }
}

function renderModel(detector) {
  const modelText = detector.loaded ? detector.model : "model not loaded";
  setChip(modelStatus, modelText, detector.last_load_error ? "bad" : detector.loaded ? "good" : "warn");
  renderDetectMode(detector.mode);
}

function renderDetectMode(mode) {
  if (!mode) {
    return;
  }
  for (const button of modeButtons) {
    button.setAttribute("aria-pressed", button.dataset.detectMode === mode ? "true" : "false");
  }
}

async function setDetectMode(mode) {
  renderDetectMode(mode);
  setChip(modelStatus, `switching to ${mode}…`, "warn");
  try {
    const response = await fetch("/api/detector/mode", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    if (response.ok) {
      const payload = await response.json();
      renderModel(payload.detector || {});
    }
  } catch {
    // The status poll will reconcile the chips on the next tick.
  }
}

function renderMetrics(detection) {
  frameMetric.textContent = String(detection.frame_id ?? "-");
  boxesMetric.textContent = String(detection.boxes?.length ?? 0);
  inferenceMetric.textContent = `${detection.inference_ms ?? 0} ms`;
}

function renderError(message) {
  if (!message) {
    errorLine.hidden = true;
    errorLine.textContent = "";
    return;
  }
  errorLine.hidden = false;
  errorLine.textContent = message;
}

function resizeOverlay() {
  const rect = overlay.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.max(1, Math.round(rect.width * dpr));
  overlay.height = Math.max(1, Math.round(rect.height * dpr));
}

function drawOverlay() {
  resizeOverlay();
  const ctx = overlay.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = overlay.width / dpr;
  const height = overlay.height / dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const detection = state.latestDetection;
  if (detection && detection.width > 0 && detection.height > 0) {
    drawBoxes(ctx, detection, width, height);
  }
  requestAnimationFrame(drawOverlay);
}

function drawBoxes(ctx, detection, stageWidth, stageHeight) {
  const fit = fitContain(stageWidth, stageHeight, detection.width, detection.height);
  ctx.lineWidth = 3;
  ctx.font = "600 14px system-ui, sans-serif";
  ctx.textBaseline = "top";

  for (const box of detection.boxes || []) {
    const color = colorForClass(box.class_id);
    const [x1, y1, x2, y2] = box.xyxy;
    const left = fit.x + x1 * fit.scale;
    const top = fit.y + y1 * fit.scale;
    const right = fit.x + x2 * fit.scale;
    const bottom = fit.y + y2 * fit.scale;
    const label = `${box.label} ${(box.confidence * 100).toFixed(0)}%`;

    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.strokeRect(left, top, right - left, bottom - top);

    const textWidth = ctx.measureText(label).width + 12;
    const textTop = Math.max(0, top - 24);
    ctx.fillRect(left, textTop, textWidth, 22);
    ctx.fillStyle = "#10100f";
    ctx.fillText(label, left + 6, textTop + 3);
  }
}

function fitContain(stageWidth, stageHeight, sourceWidth, sourceHeight) {
  const scale = Math.min(stageWidth / sourceWidth, stageHeight / sourceHeight);
  return {
    scale,
    x: (stageWidth - sourceWidth * scale) / 2,
    y: (stageHeight - sourceHeight * scale) / 2,
  };
}

function colorForClass(classId) {
  const colors = ["#48d597", "#64c7ff", "#f0bd49", "#ff6b6b", "#c6a8ff", "#79e0d0"];
  return colors[Math.abs(Number(classId || 0)) % colors.length];
}

async function pollStatus() {
  if (demoMode) {
    return;
  }

  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (response.ok) {
      renderStatus(await response.json());
    }
  } catch {
    setChip(cameraLinkStatus, "status unavailable", "bad");
  } finally {
    setTimeout(pollStatus, 1000);
  }
}

image.addEventListener("load", () => {
  emptyState.hidden = true;
});

for (const button of modeButtons) {
  button.addEventListener("click", () => setDetectMode(button.dataset.detectMode));
}

window.addEventListener("resize", resizeOverlay);
window.addEventListener("beforeunload", releaseImageUrl);
connectViewer();
pollStatus();
requestAnimationFrame(drawOverlay);
