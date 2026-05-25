const image = document.querySelector("#viewerImage");
const overlay = document.querySelector("#viewerOverlay");
const emptyState = document.querySelector("#emptyState");
const viewerSocketStatus = document.querySelector("#viewerSocketStatus");
const cameraLinkStatus = document.querySelector("#cameraLinkStatus");
const modelStatus = document.querySelector("#modelStatus");
const frameMetric = document.querySelector("#frameMetric");
const boxesMetric = document.querySelector("#boxesMetric");
const inferenceMetric = document.querySelector("#inferenceMetric");
const droppedMetric = document.querySelector("#droppedMetric");
const errorLine = document.querySelector("#errorLine");

const state = {
  ws: null,
  latestDetection: null,
  reconnectTimer: null,
};

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
  setChip(viewerSocketStatus, "viewer 連線中", "warn");
  const ws = new WebSocket(socketUrl("/ws/viewer"));
  state.ws = ws;

  ws.addEventListener("open", () => {
    setChip(viewerSocketStatus, "viewer 已連線", "good");
  });

  ws.addEventListener("message", (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === "frame") {
      state.latestDetection = payload.detection;
      image.src = payload.jpeg;
      emptyState.hidden = true;
      renderMetrics(payload.detection);
      renderError(payload.detection.error || "");
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
    setChip(viewerSocketStatus, "viewer 中斷", "bad");
    state.reconnectTimer = setTimeout(connectViewer, 1200);
  });

  ws.addEventListener("error", () => {
    setChip(viewerSocketStatus, "viewer 錯誤", "bad");
  });
}

function renderStatus(status) {
  setChip(cameraLinkStatus, status.camera_connected ? "phone 上線" : "phone 離線", status.camera_connected ? "good" : "warn");
  const detector = status.detector || {};
  const modelText = detector.loaded ? detector.model : "model 待載入";
  setChip(modelStatus, modelText, detector.last_load_error ? "bad" : detector.loaded ? "good" : "warn");
  droppedMetric.textContent = String(status.frames_dropped ?? "-");
  if (status.last_error) {
    renderError(status.last_error);
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
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (response.ok) {
      renderStatus(await response.json());
    }
  } catch {
    setChip(cameraLinkStatus, "status 失敗", "bad");
  } finally {
    setTimeout(pollStatus, 1000);
  }
}

image.addEventListener("load", () => {
  emptyState.hidden = true;
});

window.addEventListener("resize", resizeOverlay);
connectViewer();
pollStatus();
requestAnimationFrame(drawOverlay);
