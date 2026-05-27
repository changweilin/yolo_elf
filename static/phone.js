const video = document.querySelector("#cameraVideo");
const demoFrame = document.querySelector("#demoFrame");
const overlay = document.querySelector("#overlayCanvas");
const capture = document.querySelector("#captureCanvas");
const cameraIdle = document.querySelector("#cameraIdle");
const cameraToggleButton =
  document.querySelector("#cameraToggleButton") ||
  document.querySelector("#startButton") ||
  document.querySelector("[data-start-camera]");
const cameraActionButtons = Array.from(
  new Set(
    [cameraToggleButton, ...document.querySelectorAll("[data-start-camera]")].filter(Boolean),
  ),
);
const legacyStopButton = document.querySelector("#stopButton");
const fpsInput = document.querySelector("#fpsInput");
const qualityInput = document.querySelector("#qualityInput");
const cameraStatus = document.querySelector("#cameraStatus");
const socketStatus = document.querySelector("#socketStatus");
const detectStatus = document.querySelector("#detectStatus");
const adaptiveStatus = document.querySelector("#adaptiveStatus");

const moduleUrl = new URL(import.meta.url);
const demoMode =
  window.YOLO_ELF_DEMO_MODE === true || moduleUrl.searchParams.get("demo") === "1";

const MAX_BUFFERED_BYTES = 2_000_000;
const SOFT_BUFFERED_BYTES = 750_000;
const SEND_WINDOW_MS = 3000;
const INFERENCE_HEADROOM = 1.35;

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
};

const state = {
  stream: null,
  ws: null,
  captureTimer: null,
  reconnectTimer: null,
  latestDetection: null,
  drawing: false,
  sending: false,
  config: {
    width: 1280,
    height: 720,
    fps: 10,
    jpegQuality: 0.85,
  },
  pacing: {
    effectiveFps: 10,
    sentFrames: 0,
    skippedFrames: 0,
    lastFrameBytes: 0,
    sendTimes: [],
    lastSkipReason: "",
  },
};

function isLocalHostname(hostname) {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";
}

function cameraNeedsHttps() {
  return !window.isSecureContext && !isLocalHostname(window.location.hostname);
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

function setCameraToggle({ disabled = false, label = state.stream ? "Stop camera" : "Start camera" } = {}) {
  for (const button of cameraActionButtons) {
    button.disabled = disabled;
    button.textContent = label;
    button.setAttribute("aria-label", label);
    button.setAttribute("aria-pressed", state.stream ? "true" : "false");
  }
  if (legacyStopButton) {
    legacyStopButton.disabled = disabled || !state.stream;
  }
}

function setIdleVisible(visible) {
  if (cameraIdle) {
    cameraIdle.hidden = !visible;
  }
}

function initializeDemoMode() {
  if (demoFrame) {
    demoFrame.hidden = false;
  }
  video.hidden = true;
  state.latestDetection = demoDetection;
  setCameraToggle({ disabled: true, label: "Demo mode" });
  setIdleVisible(false);
  fpsInput.disabled = true;
  qualityInput.disabled = true;
  setChip(cameraStatus, "camera frozen", "warn");
  setChip(socketStatus, "socket offline", "warn");
  setChip(detectStatus, "demo boxes", "good");
  setChip(adaptiveStatus, "privacy mode", "warn");
  resizeOverlay();
  if (!state.drawing) {
    state.drawing = true;
    requestAnimationFrame(drawOverlay);
  }
}

async function startCamera() {
  if (demoMode) {
    initializeDemoMode();
    return;
  }

  setCameraToggle({ disabled: true, label: "Starting camera" });
  setIdleVisible(false);
  setChip(cameraStatus, "camera starting", "warn");
  setChip(detectStatus, "allow camera access", "warn");

  try {
    if (cameraNeedsHttps()) {
      throw new Error("Camera requires HTTPS. Use the Tailscale HTTPS URL for phone capture.");
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera API is not available in this browser context.");
    }
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
    setChip(cameraStatus, "camera connected", "good");
    setCameraToggle();
    connectSocket();
    resizeOverlay();
    if (!state.drawing) {
      state.drawing = true;
      requestAnimationFrame(drawOverlay);
    }
    scheduleCapture(0);
  } catch (error) {
    if (state.stream) {
      for (const track of state.stream.getTracks()) {
        track.stop();
      }
      state.stream = null;
      video.srcObject = null;
    }
    setCameraToggle();
    setIdleVisible(true);
    setChip(cameraStatus, "camera error", "bad");
    setChip(detectStatus, error.message || String(error), "bad");
  }
}

function stopCamera() {
  if (demoMode) {
    initializeDemoMode();
    return;
  }

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
    video.srcObject = null;
  }
  state.latestDetection = null;
  resetPacing();
  setCameraToggle();
  setIdleVisible(true);
  setChip(cameraStatus, "camera idle", "warn");
  setChip(socketStatus, "socket idle", "warn");
  setChip(detectStatus, "detection idle", "warn");
  setChip(adaptiveStatus, "adaptive idle", "warn");
  clearOverlay();
}

function connectSocket() {
  if (!state.stream) {
    return;
  }
  if (state.ws && state.ws.readyState <= WebSocket.OPEN) {
    return;
  }

  setChip(socketStatus, "socket connecting", "warn");
  const ws = new WebSocket(socketUrl("/ws/camera"));
  ws.binaryType = "arraybuffer";
  state.ws = ws;

  ws.addEventListener("open", () => {
    setChip(socketStatus, "socket connected", "good");
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
        setChip(detectStatus, "detection error", "bad");
      } else {
        setChip(detectStatus, `${boxes} boxes / ${ms} ms`, "good");
      }
      updateAdaptiveStatus();
      return;
    }
    if (payload.type === "error") {
      setChip(detectStatus, payload.message, "bad");
    }
  });

  ws.addEventListener("close", () => {
    if (state.ws !== ws) {
      return;
    }
    state.ws = null;
    if (state.stream) {
      setChip(socketStatus, "socket offline", "bad");
      state.reconnectTimer = setTimeout(connectSocket, 1200);
    }
  });

  ws.addEventListener("error", () => {
    setChip(socketStatus, "socket error", "bad");
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
  const nextDelay = delay ?? captureIntervalMs();
  state.captureTimer = setTimeout(captureFrame, nextDelay);
}

function captureFrame() {
  if (!state.stream) {
    return;
  }
  const skipReason = frameSkipReason();
  if (!skipReason) {
    state.sending = true;
    capture.width = state.config.width;
    capture.height = state.config.height;
    const ctx = capture.getContext("2d", { alpha: false });
    ctx.drawImage(video, 0, 0, capture.width, capture.height);
    const quality = Math.max(0.3, Math.min(0.95, Number(qualityInput.value || 0.85)));
    capture.toBlob(
      async (blob) => {
        try {
          if (blob && state.ws && state.ws.readyState === WebSocket.OPEN) {
            const bytes = await blob.arrayBuffer();
            state.ws.send(bytes);
            markFrameSent(bytes.byteLength);
          }
          if (!blob) {
            markFrameSkipped("encode");
          }
        } finally {
          state.sending = false;
        }
      },
      "image/jpeg",
      quality,
    );
  } else {
    markFrameSkipped(skipReason);
  }
  scheduleCapture();
}

function requestedFps() {
  const raw = Number(fpsInput.value || state.config.fps);
  return Math.max(1, Math.min(60, Number.isFinite(raw) ? raw : state.config.fps));
}

function captureIntervalMs() {
  const fps = adaptiveFps();
  state.pacing.effectiveFps = fps;
  return Math.max(16, Math.round(1000 / fps));
}

function adaptiveFps() {
  const requested = requestedFps();
  const inferenceMs = Number(state.latestDetection?.inference_ms || 0);
  let effective = requested;

  if (inferenceMs > 0) {
    effective = Math.min(effective, 1000 / Math.max(16, inferenceMs * INFERENCE_HEADROOM));
  }

  const buffered = state.ws?.bufferedAmount ?? 0;
  if (buffered >= MAX_BUFFERED_BYTES) {
    effective = 1;
  } else if (buffered >= SOFT_BUFFERED_BYTES) {
    effective = Math.min(effective, Math.max(1, requested / 2));
  }

  return Math.max(1, Math.min(requested, effective));
}

function frameSkipReason() {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
    return "socket";
  }
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return "video";
  }
  if (state.ws.bufferedAmount >= MAX_BUFFERED_BYTES) {
    return "buffer";
  }
  if (state.sending) {
    return "encode";
  }
  return "";
}

function markFrameSent(byteLength) {
  const now = performance.now();
  state.pacing.sentFrames += 1;
  state.pacing.lastFrameBytes = byteLength;
  state.pacing.lastSkipReason = "";
  state.pacing.sendTimes.push(now);
  trimSendWindow(now);
  updateAdaptiveStatus();
}

function markFrameSkipped(reason) {
  state.pacing.skippedFrames += 1;
  state.pacing.lastSkipReason = reason;
  trimSendWindow(performance.now());
  updateAdaptiveStatus();
}

function trimSendWindow(now) {
  state.pacing.sendTimes = state.pacing.sendTimes.filter((sentAt) => now - sentAt <= SEND_WINDOW_MS);
}

function actualSendFps() {
  const times = state.pacing.sendTimes;
  if (times.length < 2) {
    return times.length;
  }
  const elapsedSec = Math.max(0.001, (times[times.length - 1] - times[0]) / 1000);
  return times.length / elapsedSec;
}

function updateAdaptiveStatus() {
  const effective = state.pacing.effectiveFps || adaptiveFps();
  const actual = actualSendFps();
  const skipped = state.pacing.lastSkipReason;
  const buffered = state.ws?.bufferedAmount ?? 0;
  const tone = skipped || buffered >= SOFT_BUFFERED_BYTES ? "warn" : "good";
  const suffix = skipped ? ` / ${skipped}` : "";
  setChip(adaptiveStatus, `${actual.toFixed(1)} fps / cap ${effective.toFixed(1)}${suffix}`, tone);
}

function resetPacing() {
  state.pacing.effectiveFps = state.config.fps;
  state.pacing.sentFrames = 0;
  state.pacing.skippedFrames = 0;
  state.pacing.lastFrameBytes = 0;
  state.pacing.sendTimes = [];
  state.pacing.lastSkipReason = "";
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

function toggleCamera() {
  if (state.stream) {
    stopCamera();
    return;
  }
  startCamera();
}

setCameraToggle();
for (const button of cameraActionButtons) {
  button.addEventListener("click", toggleCamera);
}
if (legacyStopButton) {
  legacyStopButton.addEventListener("click", stopCamera);
}
window.addEventListener("resize", resizeOverlay);

if (demoMode) {
  initializeDemoMode();
} else if (cameraNeedsHttps()) {
  setChip(cameraStatus, "needs HTTPS", "bad");
  setChip(detectStatus, "Use the HTTPS phone URL", "bad");
}
