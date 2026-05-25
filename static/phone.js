const video = document.querySelector("#cameraVideo");
const overlay = document.querySelector("#overlayCanvas");
const capture = document.querySelector("#captureCanvas");
const startButton = document.querySelector("#startButton");
const stopButton = document.querySelector("#stopButton");
const fpsInput = document.querySelector("#fpsInput");
const qualityInput = document.querySelector("#qualityInput");
const cameraStatus = document.querySelector("#cameraStatus");
const socketStatus = document.querySelector("#socketStatus");
const detectStatus = document.querySelector("#detectStatus");

const state = {
  stream: null,
  ws: null,
  captureTimer: null,
  reconnectTimer: null,
  latestDetection: null,
  drawing: false,
  sending: false,
  config: {
    width: 960,
    height: 540,
    fps: 10,
    jpegQuality: 0.65,
  },
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

async function startCamera() {
  startButton.disabled = true;
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        aspectRatio: { ideal: 16 / 9 },
      },
    });
    video.srcObject = state.stream;
    await video.play();
    setChip(cameraStatus, "相機上線", "good");
    stopButton.disabled = false;
    connectSocket();
    resizeOverlay();
    if (!state.drawing) {
      state.drawing = true;
      requestAnimationFrame(drawOverlay);
    }
    scheduleCapture(0);
  } catch (error) {
    startButton.disabled = false;
    setChip(cameraStatus, `相機失敗`, "bad");
    setChip(detectStatus, error.message || String(error), "bad");
  }
}

function stopCamera() {
  if (state.captureTimer) {
    clearTimeout(state.captureTimer);
    state.captureTimer = null;
  }
  if (state.reconnectTimer) {
    clearTimeout(state.reconnectTimer);
    state.reconnectTimer = null;
  }
  if (state.ws) {
    state.ws.close();
    state.ws = null;
  }
  if (state.stream) {
    for (const track of state.stream.getTracks()) {
      track.stop();
    }
    state.stream = null;
  }
  state.latestDetection = null;
  startButton.disabled = false;
  stopButton.disabled = true;
  setChip(cameraStatus, "相機待命", "warn");
  setChip(socketStatus, "連線待命", "warn");
  setChip(detectStatus, "尚無偵測", "warn");
  clearOverlay();
}

function connectSocket() {
  if (!state.stream) {
    return;
  }
  if (state.ws && state.ws.readyState <= WebSocket.OPEN) {
    return;
  }

  setChip(socketStatus, "連線中", "warn");
  const ws = new WebSocket(socketUrl("/ws/camera"));
  ws.binaryType = "arraybuffer";
  state.ws = ws;

  ws.addEventListener("open", () => {
    setChip(socketStatus, "已連線", "good");
  });

  ws.addEventListener("message", (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === "config") {
      applyServerConfig(payload.capture);
      return;
    }
    if (payload.type === "detection") {
      state.latestDetection = payload.detection;
      const boxes = payload.detection.boxes?.length ?? 0;
      const ms = payload.detection.inference_ms ?? 0;
      if (payload.detection.error) {
        setChip(detectStatus, "偵測錯誤", "bad");
      } else {
        setChip(detectStatus, `${boxes} boxes / ${ms} ms`, "good");
      }
      return;
    }
    if (payload.type === "error") {
      setChip(detectStatus, payload.message, "bad");
    }
  });

  ws.addEventListener("close", () => {
    if (state.ws === ws) {
      state.ws = null;
    }
    setChip(socketStatus, "連線中斷", "bad");
    if (state.stream) {
      state.reconnectTimer = setTimeout(connectSocket, 1200);
    }
  });

  ws.addEventListener("error", () => {
    setChip(socketStatus, "連線錯誤", "bad");
  });
}

function applyServerConfig(config) {
  if (!config) {
    return;
  }
  state.config.width = Number(config.width || state.config.width);
  state.config.height = Number(config.height || state.config.height);
  state.config.fps = Number(config.fps || state.config.fps);
  state.config.jpegQuality = Number(config.jpeg_quality || state.config.jpegQuality);
  fpsInput.value = String(state.config.fps);
  qualityInput.value = String(state.config.jpegQuality);
}

function scheduleCapture(delay = null) {
  const fps = Math.max(1, Number(fpsInput.value || state.config.fps));
  const nextDelay = delay ?? Math.round(1000 / fps);
  state.captureTimer = setTimeout(captureFrame, nextDelay);
}

function captureFrame() {
  if (!state.stream) {
    return;
  }
  if (
    state.ws &&
    state.ws.readyState === WebSocket.OPEN &&
    video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    state.ws.bufferedAmount < 2_000_000 &&
    !state.sending
  ) {
    state.sending = true;
    capture.width = state.config.width;
    capture.height = state.config.height;
    const ctx = capture.getContext("2d", { alpha: false });
    ctx.drawImage(video, 0, 0, capture.width, capture.height);
    const quality = Math.max(0.3, Math.min(0.95, Number(qualityInput.value || 0.65)));
    capture.toBlob(
      async (blob) => {
        try {
          if (blob && state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(await blob.arrayBuffer());
          }
        } finally {
          state.sending = false;
        }
      },
      "image/jpeg",
      quality,
    );
  }
  scheduleCapture();
}

function resizeOverlay() {
  const rect = overlay.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.max(1, Math.round(rect.width * dpr));
  overlay.height = Math.max(1, Math.round(rect.height * dpr));
}

function clearOverlay() {
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
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

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
window.addEventListener("resize", resizeOverlay);
