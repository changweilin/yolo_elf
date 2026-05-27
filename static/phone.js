const video = document.querySelector("#cameraVideo");
const cameraShell = document.querySelector(".camera-shell");
const cameraStage = document.querySelector(".camera-stage");
const demoFrame = document.querySelector("#demoFrame");
const overlay = document.querySelector("#overlayCanvas");
const capture = document.querySelector("#captureCanvas");
const cameraIdle = document.querySelector("#cameraIdle");
const focusReticle = document.querySelector("#focusReticle");
const settingsToggleButton = document.querySelector("#settingsToggleButton");
const advancedControls = document.querySelector("#advancedControls");
const statusRow = document.querySelector("#statusRow");
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
const lensToggleButton = document.querySelector("#lensToggleButton");
const zoomInput = document.querySelector("#zoomInput");
const zoomValue = document.querySelector("#zoomValue");
const shutterInput = document.querySelector("#shutterInput");
const isoInput = document.querySelector("#isoInput");
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
  camera: {
    facingMode: "environment",
    capabilities: {},
    settings: {},
    zoom: 1,
    pendingAdvanced: {},
    applyPromise: null,
  },
  gesture: {
    pointers: new Map(),
    tap: null,
    pinch: null,
    suppressTap: false,
  },
  ui: {
    settingsExpanded: true,
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

function setSettingsExpanded(expanded) {
  state.ui.settingsExpanded = expanded;
  cameraShell?.classList.toggle("settings-collapsed", !expanded);
  if (settingsToggleButton) {
    const label = expanded ? "Collapse settings" : "Expand settings";
    settingsToggleButton.setAttribute("aria-expanded", expanded ? "true" : "false");
    settingsToggleButton.setAttribute("aria-label", label);
    settingsToggleButton.title = label;
  }
  if (advancedControls) {
    advancedControls.setAttribute("aria-disabled", expanded ? "false" : "true");
  }
  if (statusRow) {
    statusRow.setAttribute("aria-hidden", expanded ? "false" : "true");
  }
}

function getVideoTrack() {
  return state.stream?.getVideoTracks?.()[0] || null;
}

function selectedFacingMode() {
  return state.camera.facingMode || "environment";
}

function cameraVideoConstraints() {
  return {
    facingMode: { ideal: selectedFacingMode() },
    width: { ideal: 1280 },
    height: { ideal: 720 },
    aspectRatio: { ideal: 16 / 9 },
  };
}

function lensLabel(facingMode) {
  return facingMode === "user" ? "Front" : "Back";
}

function nextFacingMode() {
  return selectedFacingMode() === "user" ? "environment" : "user";
}

function supportedConstraint(name) {
  const supported = navigator.mediaDevices?.getSupportedConstraints?.() || {};
  return supported[name] === true;
}

function capabilityRange(capabilities, name) {
  const range = capabilities?.[name];
  if (
    !range ||
    !Number.isFinite(Number(range.min)) ||
    !Number.isFinite(Number(range.max)) ||
    Number(range.max) <= Number(range.min)
  ) {
    return null;
  }
  return {
    min: Number(range.min),
    max: Number(range.max),
    step: Number.isFinite(Number(range.step)) && Number(range.step) > 0 ? Number(range.step) : null,
  };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function clampToRangeStep(value, range) {
  const clamped = clamp(Number(value), range.min, range.max);
  if (!range.step) {
    return clamped;
  }
  const steps = Math.round((clamped - range.min) / range.step);
  return clamp(range.min + steps * range.step, range.min, range.max);
}

function decimalsForStep(step) {
  if (!step || step >= 1) {
    return 0;
  }
  return Math.min(6, Math.ceil(Math.abs(Math.log10(step))));
}

function formatControlNumber(value, step = null) {
  if (!Number.isFinite(Number(value))) {
    return "";
  }
  return Number(value).toFixed(decimalsForStep(step));
}

function formatZoom(value) {
  return `${Number(value || 1).toFixed(1)}x`;
}

function defaultStepForRange(range, fallback = 1) {
  if (range?.step) {
    return range.step;
  }
  const span = Number(range?.max) - Number(range?.min);
  if (!Number.isFinite(span) || span <= 0) {
    return fallback;
  }
  return span <= 1 ? Math.max(span / 100, 0.000001) : fallback;
}

function setNumberInputCapability(input, range, value, fallbackStep = 1) {
  if (!input) {
    return;
  }
  input.disabled = !range;
  if (!range) {
    input.removeAttribute("min");
    input.removeAttribute("max");
    input.removeAttribute("step");
    input.value = "";
    return;
  }
  const step = defaultStepForRange(range, fallbackStep);
  input.min = formatControlNumber(range.min, step);
  input.max = formatControlNumber(range.max, step);
  input.step = formatControlNumber(step, step);
  input.value = Number.isFinite(Number(value)) ? formatControlNumber(clampToRangeStep(value, range), step) : "";
}

function setZoomCapability(range, value) {
  if (!zoomInput || !zoomValue) {
    return;
  }
  zoomInput.disabled = !range;
  if (!range) {
    zoomInput.min = "1";
    zoomInput.max = "1";
    zoomInput.step = "0.1";
    zoomInput.value = "1";
    zoomValue.textContent = "1.0x";
    state.camera.zoom = 1;
    return;
  }

  const step = range.step || 0.1;
  const nextZoom = clampToRangeStep(value, { ...range, step });
  zoomInput.min = formatControlNumber(range.min, step);
  zoomInput.max = formatControlNumber(range.max, step);
  zoomInput.step = formatControlNumber(step, step);
  zoomInput.value = formatControlNumber(nextZoom, step);
  zoomValue.textContent = formatZoom(nextZoom);
  state.camera.zoom = nextZoom;
}

function syncCameraControls() {
  const track = getVideoTrack();
  const capabilities = track?.getCapabilities?.() || {};
  const settings = track?.getSettings?.() || {};
  state.camera.capabilities = capabilities;
  state.camera.settings = settings;

  if (settings.facingMode === "user" || settings.facingMode === "environment") {
    state.camera.facingMode = settings.facingMode;
  }
  if (lensToggleButton) {
    const currentMode = selectedFacingMode();
    const targetMode = currentMode === "user" ? "back" : "front";
    lensToggleButton.disabled = demoMode;
    lensToggleButton.textContent = lensLabel(currentMode);
    lensToggleButton.setAttribute("aria-label", `Switch to ${targetMode} camera`);
  }

  if (!track) {
    setZoomCapability(null, 1);
    setNumberInputCapability(shutterInput, null, null);
    setNumberInputCapability(isoInput, null, null);
    return;
  }

  const zoomRange = capabilityRange(capabilities, "zoom");
  setZoomCapability(zoomRange, settings.zoom ?? state.camera.zoom ?? zoomRange?.min ?? 1);

  const exposureModes = Array.isArray(capabilities.exposureMode) ? capabilities.exposureMode : [];
  const manualExposure = exposureModes.length === 0 || exposureModes.includes("manual");
  setNumberInputCapability(
    shutterInput,
    manualExposure ? capabilityRange(capabilities, "exposureTime") : null,
    settings.exposureTime,
  );
  setNumberInputCapability(
    isoInput,
    manualExposure ? capabilityRange(capabilities, "iso") : null,
    settings.iso,
  );
}

async function tryApplyTrackAdvanced(advanced) {
  const track = getVideoTrack();
  if (!track?.applyConstraints || !advanced || Object.keys(advanced).length === 0) {
    return false;
  }

  const attempts = [{ advanced: [advanced] }, advanced];
  let lastError = null;
  for (const constraints of attempts) {
    try {
      await track.applyConstraints(constraints);
      state.camera.settings = track.getSettings?.() || {};
      return true;
    } catch (error) {
      lastError = error;
    }
  }
  console.warn("Camera constraint rejected", advanced, lastError);
  return false;
}

function queueTrackAdvanced(advanced) {
  state.camera.pendingAdvanced = {
    ...state.camera.pendingAdvanced,
    ...advanced,
  };

  if (!state.camera.applyPromise) {
    state.camera.applyPromise = flushTrackAdvancedQueue();
  }
  return state.camera.applyPromise;
}

async function flushTrackAdvancedQueue() {
  let applied = true;
  try {
    while (Object.keys(state.camera.pendingAdvanced).length > 0 && getVideoTrack()) {
      const advanced = state.camera.pendingAdvanced;
      state.camera.pendingAdvanced = {};
      applied = (await tryApplyTrackAdvanced(advanced)) && applied;
    }
    return applied;
  } finally {
    state.camera.applyPromise = null;
    syncCameraControls();
  }
}

function setCameraZoom(value) {
  const range = capabilityRange(state.camera.capabilities, "zoom");
  if (!range || !Number.isFinite(Number(value))) {
    return false;
  }
  const zoom = clampToRangeStep(value, range);
  state.camera.zoom = zoom;
  setZoomCapability(range, zoom);
  void queueTrackAdvanced({ zoom });
  return true;
}

async function applyManualCameraNumber(input, capabilityName, label) {
  const range = capabilityRange(state.camera.capabilities, capabilityName);
  const value = Number(input?.value);
  if (!range || !Number.isFinite(value)) {
    return;
  }

  const nextValue = clampToRangeStep(value, range);
  const advanced = { [capabilityName]: nextValue };
  const exposureModes = Array.isArray(state.camera.capabilities.exposureMode)
    ? state.camera.capabilities.exposureMode
    : [];
  if (exposureModes.includes("manual")) {
    advanced.exposureMode = "manual";
  }

  input.value = formatControlNumber(nextValue, defaultStepForRange(range));
  const applied = await queueTrackAdvanced(advanced);
  setChip(cameraStatus, applied ? `${label} set` : `${label} unavailable`, applied ? "good" : "warn");
}

async function handleLensToggle() {
  state.camera.facingMode = nextFacingMode();
  if (!state.stream) {
    syncCameraControls();
    return;
  }
  setCameraToggle({ disabled: true, label: "Switching camera" });
  stopCamera();
  await startCamera();
}

function stagePointFromEvent(event) {
  const rect = cameraStage.getBoundingClientRect();
  const sourceWidth = video.videoWidth || rect.width;
  const sourceHeight = video.videoHeight || rect.height;
  const fit = fitContain(rect.width, rect.height, sourceWidth, sourceHeight);
  const x = clamp((event.clientX - rect.left - fit.x) / Math.max(1, sourceWidth * fit.scale), 0, 1);
  const y = clamp((event.clientY - rect.top - fit.y) / Math.max(1, sourceHeight * fit.scale), 0, 1);
  return { x, y };
}

function showFocusReticle(event, ok = true) {
  if (!focusReticle || !cameraStage) {
    return;
  }
  const rect = cameraStage.getBoundingClientRect();
  focusReticle.style.left = `${event.clientX - rect.left}px`;
  focusReticle.style.top = `${event.clientY - rect.top}px`;
  focusReticle.hidden = false;
  focusReticle.classList.toggle("bad", !ok);
  focusReticle.classList.remove("active");
  requestAnimationFrame(() => focusReticle.classList.add("active"));
  window.setTimeout(() => {
    focusReticle.classList.remove("active");
    window.setTimeout(() => {
      focusReticle.hidden = true;
    }, 180);
  }, ok ? 520 : 700);
}

async function focusCameraAt(event) {
  const track = getVideoTrack();
  if (!track?.applyConstraints) {
    showFocusReticle(event, false);
    return;
  }
  if (state.camera.applyPromise) {
    await state.camera.applyPromise;
  }

  const point = stagePointFromEvent(event);
  const capabilities = state.camera.capabilities || {};
  const focusModes = Array.isArray(capabilities.focusMode) ? capabilities.focusMode : [];
  const supportsPointFocus =
    supportedConstraint("pointsOfInterest") ||
    Object.prototype.hasOwnProperty.call(capabilities, "pointsOfInterest");
  const attempts = [];

  if (supportsPointFocus && focusModes.includes("single-shot")) {
    attempts.push({ focusMode: "single-shot", pointsOfInterest: [point] });
  }
  if (supportsPointFocus) {
    attempts.push({ pointsOfInterest: [point] });
  }
  if (focusModes.includes("single-shot")) {
    attempts.push({ focusMode: "single-shot" });
  }
  if (focusModes.includes("continuous")) {
    attempts.push({ focusMode: "continuous" });
  }

  let focused = false;
  for (const constraints of attempts) {
    focused = await tryApplyTrackAdvanced(constraints);
    if (focused) {
      break;
    }
  }

  showFocusReticle(event, focused);
  setChip(cameraStatus, focused ? "focus set" : "focus unavailable", focused ? "good" : "warn");
}

function pointerDistance(points) {
  const [first, second] = Array.from(points.values());
  if (!first || !second) {
    return 0;
  }
  return Math.hypot(first.clientX - second.clientX, first.clientY - second.clientY);
}

function handleStagePointerDown(event) {
  if (!state.stream || demoMode || (event.pointerType === "mouse" && event.button !== 0)) {
    return;
  }
  cameraStage.setPointerCapture?.(event.pointerId);
  state.gesture.pointers.set(event.pointerId, {
    clientX: event.clientX,
    clientY: event.clientY,
  });

  if (state.gesture.pointers.size === 1) {
    state.gesture.tap = {
      id: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      moved: false,
      startedAt: performance.now(),
    };
    state.gesture.suppressTap = false;
  }

  if (state.gesture.pointers.size === 2) {
    state.gesture.pinch = {
      startDistance: pointerDistance(state.gesture.pointers),
      startZoom: state.camera.zoom || 1,
    };
    state.gesture.suppressTap = true;
  }
}

function handleStagePointerMove(event) {
  if (!state.gesture.pointers.has(event.pointerId)) {
    return;
  }
  state.gesture.pointers.set(event.pointerId, {
    clientX: event.clientX,
    clientY: event.clientY,
  });

  if (state.gesture.tap?.id === event.pointerId) {
    const travel = Math.hypot(event.clientX - state.gesture.tap.x, event.clientY - state.gesture.tap.y);
    state.gesture.tap.moved = state.gesture.tap.moved || travel > 12;
  }

  if (state.gesture.pinch && state.gesture.pointers.size >= 2) {
    const distance = pointerDistance(state.gesture.pointers);
    if (state.gesture.pinch.startDistance > 0 && distance > 0) {
      event.preventDefault();
      setCameraZoom(state.gesture.pinch.startZoom * (distance / state.gesture.pinch.startDistance));
    }
  }
}

function handleStagePointerUp(event) {
  const tap = state.gesture.tap;
  const wasSinglePointer = state.gesture.pointers.size === 1;
  const shouldFocus =
    tap?.id === event.pointerId &&
    wasSinglePointer &&
    !tap.moved &&
    !state.gesture.suppressTap &&
    performance.now() - tap.startedAt < 650;

  state.gesture.pointers.delete(event.pointerId);
  if (state.gesture.pointers.size < 2) {
    state.gesture.pinch = null;
  }
  if (state.gesture.pointers.size === 0) {
    state.gesture.tap = null;
    state.gesture.suppressTap = false;
  }

  if (shouldFocus) {
    void focusCameraAt(event);
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
  syncCameraControls();
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
      video: cameraVideoConstraints(),
    });
    video.srcObject = state.stream;
    await video.play();
    syncCameraControls();
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
    syncCameraControls();
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
  state.camera.pendingAdvanced = {};
  state.gesture.pointers.clear();
  state.gesture.tap = null;
  state.gesture.pinch = null;
  state.gesture.suppressTap = false;
  state.latestDetection = null;
  resetPacing();
  syncCameraControls();
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
if (settingsToggleButton) {
  settingsToggleButton.addEventListener("click", () =>
    setSettingsExpanded(!state.ui.settingsExpanded),
  );
}
if (lensToggleButton) {
  lensToggleButton.addEventListener("click", handleLensToggle);
}
if (zoomInput) {
  zoomInput.addEventListener("input", () => setCameraZoom(Number(zoomInput.value)));
}
if (shutterInput) {
  shutterInput.addEventListener("change", () =>
    applyManualCameraNumber(shutterInput, "exposureTime", "shutter"),
  );
}
if (isoInput) {
  isoInput.addEventListener("change", () => applyManualCameraNumber(isoInput, "iso", "ISO"));
}
if (cameraStage) {
  cameraStage.addEventListener("pointerdown", handleStagePointerDown);
  cameraStage.addEventListener("pointermove", handleStagePointerMove);
  cameraStage.addEventListener("pointerup", handleStagePointerUp);
  cameraStage.addEventListener("pointercancel", handleStagePointerUp);
}
window.addEventListener("resize", resizeOverlay);
setSettingsExpanded(state.ui.settingsExpanded);
syncCameraControls();

if (demoMode) {
  initializeDemoMode();
} else if (cameraNeedsHttps()) {
  setChip(cameraStatus, "needs HTTPS", "bad");
  setChip(detectStatus, "Use the HTTPS phone URL", "bad");
}
