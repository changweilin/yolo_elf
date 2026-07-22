const image = document.querySelector("#viewerImage");
const overlay = document.querySelector("#viewerOverlay");
const emptyState = document.querySelector("#emptyState");
const viewerSocketStatus = document.querySelector("#viewerSocketStatus");
const cameraLinkStatus = document.querySelector("#cameraLinkStatus");
const phoneStorageStatus = document.querySelector("#phoneStorageStatus");
const modelStatus = document.querySelector("#modelStatus");
const storageStatus = document.querySelector("#storageStatus");
const alertStatus = document.querySelector("#alertStatus");
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

import { createModelSwitch } from "./mode-switch.js";

const modelSwitch = createModelSwitch({
  progressEl: document.querySelector("#modelSwitchProgress"),
  fillEl: document.querySelector("#modelSwitchFill"),
  lock(on) {
    modeGroup?.classList.toggle("is-locked", on);
  },
});

const moduleUrl = new URL(import.meta.url);
const demoMode =
  window.YOLO_ELF_DEMO_MODE === true || moduleUrl.searchParams.get("demo") === "1";

const demoDetection = {
  frame_id: 42,
  width: 1280,
  height: 720,
  inference_ms: 18.6,
  boxes: [
    { xyxy: [124, 137, 392, 535], class_id: 0, label: "monitor", confidence: 0.88, track_id: 1, zones: ["doorway"] },
    { xyxy: [489, 302, 633, 514], class_id: 1, label: "bottle", confidence: 0.76, track_id: 2, zones: [] },
    { xyxy: [759, 219, 1062, 521], class_id: 2, label: "package", confidence: 0.93, track_id: 3, zones: [] },
  ],
  zone_counts: { doorway: 1 },
  error: "",
};

const demoZones = [
  { name: "doorway", anchor: "center", points: [[0.05, 0.2], [0.35, 0.2], [0.35, 0.95], [0.05, 0.95]] },
];

const zoneEditToggle = document.querySelector("#zoneEditToggle");
const zoneDraftBar = document.querySelector("#zoneDraftBar");
const zoneDraftHint = document.querySelector("#zoneDraftHint");
const zoneFinishButton = document.querySelector("#zoneFinishButton");
const zoneUndoButton = document.querySelector("#zoneUndoButton");
const zoneClearButton = document.querySelector("#zoneClearButton");
const zoneList = document.querySelector("#zoneList");

const state = {
  ws: null,
  latestDetection: null,
  pendingFrame: null,
  imageUrl: null,
  reconnectTimer: null,
  zones: [],
  zoneCounts: {},
  editor: { active: false, draft: [] },
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
    if (payload.type === "alert") {
      handleAlert(payload);
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
  setChip(alertStatus, "⚠ person ×2 (demo)", "bad");
  for (const button of modeButtons) {
    button.disabled = true;
  }
  if (zoneEditToggle) {
    zoneEditToggle.disabled = true;
  }
  state.zones = demoZones;
  renderZoneList();
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
  renderAlerts(status.alerts);
  renderZones(status.zones);
  if (status.last_error) {
    renderError(status.last_error);
  }
}

function renderZones(zones) {
  // Saved zones are server-owned; the in-progress draft lives in state.editor,
  // so refreshing this list never disturbs an active drawing.
  state.zones = (zones && zones.zones) || [];
  renderZoneList();
}

// Keep a just-fired alert on screen briefly instead of letting the 1s status
// poll immediately reset the chip back to its steady "armed" state.
const alertFlash = { until: 0 };

function handleAlert(event) {
  if (!alertStatus) {
    return;
  }
  setChip(alertStatus, `⚠ ${event.rule} ×${event.count}`, "bad");
  alertFlash.until = Date.now() + 6000;
  notifyBrowser(event);
}

function renderAlerts(alerts) {
  if (!alertStatus || Date.now() < alertFlash.until) {
    return;
  }
  if (!alerts || !alerts.enabled) {
    setChip(alertStatus, "alerts off", "warn");
    return;
  }
  const fired = alerts.events_fired || 0;
  setChip(alertStatus, fired ? `alerts armed · ${fired}` : "alerts armed", "good");
}

function notifyBrowser(event) {
  if (typeof Notification === "undefined") {
    return;
  }
  const show = () => {
    try {
      new Notification("YOLO Elf", { body: `${event.rule} ×${event.count}` });
    } catch {
      // Notifications are best-effort; ignore construction failures.
    }
  };
  if (Notification.permission === "granted") {
    show();
  } else if (Notification.permission === "default") {
    Notification.requestPermission()
      .then((permission) => permission === "granted" && show())
      .catch(() => {});
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
  modelSwitch.begin(mode);
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
  state.zoneCounts = detection.zone_counts || {};
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
    drawZones(ctx, detection, width, height);
    drawBoxes(ctx, detection, width, height);
    drawDraft(ctx, detection, width, height);
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
    // When the second-stage classifier named a species, show that as the
    // primary label (圖鑑); otherwise fall back to the detection class. A
    // tracker id (when present) is prefixed so an object is followable.
    const idPrefix = box.track_id != null ? `#${box.track_id} ` : "";
    const label = box.species
      ? `${idPrefix}${box.species} ${Math.round((box.species_confidence ?? 0) * 100)}%`
      : `${idPrefix}${box.label} ${(box.confidence * 100).toFixed(0)}%`;

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

// ROI zones are stored normalized (0..1) to the source frame, so map them onto
// the same letterboxed rectangle the frame is drawn in.
function zonePointToStage(fit, detection, nx, ny) {
  return [fit.x + nx * detection.width * fit.scale, fit.y + ny * detection.height * fit.scale];
}

function zoneColor(index) {
  const colors = ["#ffd166", "#06d6a0", "#ef476f", "#118ab2", "#c77dff"];
  return colors[Math.abs(Number(index || 0)) % colors.length];
}

function tracePolygon(ctx, points) {
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i][0], points[i][1]);
  }
}

function drawZones(ctx, detection, stageWidth, stageHeight) {
  if (!state.zones.length) {
    return;
  }
  const fit = fitContain(stageWidth, stageHeight, detection.width, detection.height);
  ctx.lineWidth = 2;
  ctx.font = "600 13px system-ui, sans-serif";
  ctx.textBaseline = "top";

  state.zones.forEach((zone, index) => {
    const points = (zone.points || []).map(([nx, ny]) => zonePointToStage(fit, detection, nx, ny));
    if (points.length < 2) {
      return;
    }
    const color = zoneColor(index);
    tracePolygon(ctx, points);
    ctx.closePath();
    ctx.fillStyle = withAlpha(color, 0.14);
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.stroke();

    const count = state.zoneCounts?.[zone.name] ?? 0;
    const label = `${zone.name} · ${count}`;
    const [lx, ly] = points[0];
    const top = Math.max(0, ly - 22);
    ctx.fillStyle = color;
    ctx.fillRect(lx, top, ctx.measureText(label).width + 12, 20);
    ctx.fillStyle = "#10100f";
    ctx.fillText(label, lx + 6, top + 3);
  });
}

function drawDraft(ctx, detection, stageWidth, stageHeight) {
  const draft = state.editor.draft;
  if (!state.editor.active || !draft.length) {
    return;
  }
  const fit = fitContain(stageWidth, stageHeight, detection.width, detection.height);
  const points = draft.map(([nx, ny]) => zonePointToStage(fit, detection, nx, ny));
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#ffd166";
  ctx.fillStyle = "#ffd166";
  if (points.length > 1) {
    tracePolygon(ctx, points);
    ctx.stroke();
  }
  for (const [x, y] of points) {
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function withAlpha(hex, alpha) {
  const value = hex.replace("#", "");
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function round4(value) {
  return Math.round(value * 10000) / 10000;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function stageToNormalized(clientX, clientY) {
  const detection = state.latestDetection;
  if (!detection || !detection.width || !detection.height) {
    return null;
  }
  const rect = overlay.getBoundingClientRect();
  const fit = fitContain(rect.width, rect.height, detection.width, detection.height);
  const px = (clientX - rect.left - fit.x) / fit.scale;
  const py = (clientY - rect.top - fit.y) / fit.scale;
  return [clamp01(px / detection.width), clamp01(py / detection.height)];
}

function updateDraftHint(message) {
  if (!zoneDraftHint) {
    return;
  }
  if (message) {
    zoneDraftHint.textContent = message;
    return;
  }
  zoneDraftHint.textContent = state.latestDetection
    ? `已加入 ${state.editor.draft.length} 點（至少 3 點）`
    : "等待畫面出現後再點擊繪製";
}

function setEditor(active) {
  state.editor.active = active;
  state.editor.draft = [];
  if (zoneDraftBar) {
    zoneDraftBar.hidden = !active;
  }
  if (zoneEditToggle) {
    zoneEditToggle.textContent = active ? "完成編輯" : "編輯";
  }
  // The overlay is pointer-events:none by default; enable it only while editing
  // so normal viewing never intercepts clicks.
  overlay.style.pointerEvents = active ? "auto" : "";
  overlay.style.cursor = active ? "crosshair" : "";
  updateDraftHint();
}

function onStageClick(event) {
  if (!state.editor.active) {
    return;
  }
  const point = stageToNormalized(event.clientX, event.clientY);
  if (!point) {
    updateDraftHint("等待畫面出現後再點擊繪製");
    return;
  }
  state.editor.draft.push(point);
  updateDraftHint();
}

function publicZone(zone) {
  return { name: zone.name, points: zone.points, anchor: zone.anchor || "center" };
}

async function finishZone() {
  if (state.editor.draft.length < 3) {
    updateDraftHint("至少需要 3 點");
    return;
  }
  const name = (window.prompt("區域名稱", `zone-${state.zones.length + 1}`) || "").trim();
  if (!name) {
    return;
  }
  const zones = [
    ...state.zones.map(publicZone),
    { name, points: state.editor.draft.map(([x, y]) => [round4(x), round4(y)]), anchor: "center" },
  ];
  if (await saveZones(zones)) {
    state.editor.draft = [];
    updateDraftHint();
  }
}

async function deleteZone(name) {
  await saveZones(state.zones.filter((zone) => zone.name !== name).map(publicZone));
}

async function saveZones(zones) {
  try {
    const response = await fetch("/api/zones", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ zones }),
    });
    if (!response.ok) {
      updateDraftHint("儲存失敗，請檢查頂點");
      return false;
    }
    const payload = await response.json();
    state.zones = (payload.zones && payload.zones.zones) || [];
    renderZoneList();
    return true;
  } catch {
    updateDraftHint("無法連線到伺服器");
    return false;
  }
}

function renderZoneList() {
  if (!zoneList) {
    return;
  }
  zoneList.textContent = "";
  for (const zone of state.zones) {
    const item = document.createElement("li");
    item.className = "zone-list-item";
    const label = document.createElement("span");
    label.textContent = `${zone.name} (${(zone.points || []).length})`;
    item.append(label);
    if (!demoMode) {
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "compact-button";
      remove.textContent = "刪除";
      remove.addEventListener("click", () => deleteZone(zone.name));
      item.append(remove);
    }
    zoneList.append(item);
  }
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

zoneEditToggle?.addEventListener("click", () => setEditor(!state.editor.active));
zoneFinishButton?.addEventListener("click", finishZone);
zoneUndoButton?.addEventListener("click", () => {
  state.editor.draft.pop();
  updateDraftHint();
});
zoneClearButton?.addEventListener("click", () => {
  state.editor.draft = [];
  updateDraftHint();
});
overlay.addEventListener("click", onStageClick);

window.addEventListener("resize", resizeOverlay);
window.addEventListener("beforeunload", releaseImageUrl);
connectViewer();
pollStatus();
requestAnimationFrame(drawOverlay);
