const stage = document.querySelector("#viewerStage");
const primaryPaneEl = document.querySelector("#viewerPanePrimary");
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

// Channel switch (YOLO ⇄ VLM). Hidden until /api/status reports the VLM channel
// is enabled server-side. Purely a viewer-side toggle: it only changes which
// channel's boxes this tab draws, never the backend.
const channelGroupControl = document.querySelector("#channelGroupControl");
const channelGroup = document.querySelector("#channelGroup");
const channelButtons = channelGroup
  ? Array.from(channelGroup.querySelectorAll("[data-channel]"))
  : [];
const vlmCaption = document.querySelector("#vlmCaption");

// Task switch (detect / segment / pose / obb / open-vocab / semantic / depth).
// Unlike the channel toggle this one *is* server-side: it swaps which head the
// single detection worker runs, so it affects every viewer at once.
const taskGroup = document.querySelector("#taskGroup");
const taskButtons = taskGroup
  ? Array.from(taskGroup.querySelectorAll("[data-detect-task]"))
  : [];
const rasterLegend = document.querySelector("#rasterLegend");

// COCO-17 limb pairs, 0-indexed (Ultralytics publishes them 1-indexed).
const POSE_SKELETON = [
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
  [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
  [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
];
// Occluded joints come back at (0, 0) with near-zero confidence; anything below
// this is skipped so the skeleton is not dragged to the frame corner.
const KEYPOINT_MIN_CONF = 0.25;
// Full-frame overlay; opaque enough to read, sheer enough to keep the scene.
const RASTER_ALPHA = 0.55;

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

// /viewer?camera_id=back pins this tab to one camera: the socket subscribes to
// that stream only (no bandwidth spent on frames this tab will not draw) and the
// stage shows just its pane. Read from the page URL, not the module URL.
const pinnedCamera = new URLSearchParams(window.location.search).get("camera_id") || "";

// The front camera also carries a segmentation contour and a rotated box, so the
// static build shows what the segment / obb task channels look like without a
// server. The renderers key off the presence of these fields, not the task name.
const demoDetection = {
  frame_id: 42,
  width: 1280,
  height: 720,
  inference_ms: 18.6,
  boxes: [
    {
      xyxy: [124, 137, 392, 535], class_id: 0, label: "monitor", confidence: 0.88,
      track_id: 1, zones: ["doorway"],
      mask: [
        [150, 160], [330, 145], [378, 220], [392, 380], [360, 505], [250, 535],
        [160, 500], [128, 360], [134, 230],
      ],
    },
    { xyxy: [489, 302, 633, 514], class_id: 1, label: "bottle", confidence: 0.76, track_id: 2, zones: [] },
    {
      xyxy: [759, 219, 1062, 521], class_id: 2, label: "package", confidence: 0.93,
      track_id: 3, zones: [],
      obb: [812, 219, 1062, 300, 1010, 521, 759, 440],
    },
  ],
  zone_counts: { doorway: 1 },
  error: "",
};

// A second demo stream so the static build shows what the multi-camera grid
// looks like. Track ids restart per camera — #1 here is a different object from
// #1 above, which is exactly the property the backend guarantees. The person
// carries COCO-17 keypoints so the pose skeleton is on show too.
const demoDetectionBack = {
  frame_id: 41,
  width: 1280,
  height: 720,
  inference_ms: 21.4,
  boxes: [
    {
      xyxy: [212, 244, 470, 604], class_id: 3, label: "person", confidence: 0.81,
      track_id: 1, zones: [],
      keypoints: [
        [341, 268, 0.99], [349, 261, 0.97], [333, 261, 0.97], [361, 264, 0.93], [321, 264, 0.94],
        [385, 305, 0.96], [297, 305, 0.96], [406, 366, 0.92], [276, 366, 0.91],
        [420, 424, 0.88], [262, 424, 0.87], [372, 428, 0.94], [310, 428, 0.94],
        [378, 516, 0.90], [304, 516, 0.90], [382, 598, 0.85], [300, 598, 0.85],
      ],
    },
    { xyxy: [820, 188, 1104, 452], class_id: 4, label: "bicycle", confidence: 0.69, track_id: 2, zones: [] },
  ],
  zone_counts: {},
  error: "",
};

// Open-vocab boxes for the VLM channel demo. Free-form phrases and no
// confidence/track id are exactly what a real Florence-2 pass returns.
const demoVlm = {
  type: "vlm",
  camera_id: "front",
  frame_id: 42,
  width: 1280,
  height: 720,
  inference_ms: 486.0,
  boxes: [
    { xyxy: [124, 137, 392, 535], class_id: -1, label: "a person standing by the door", confidence: 1.0, track_id: null },
    { xyxy: [759, 219, 1062, 521], class_id: -1, label: "a cardboard delivery box", confidence: 1.0, track_id: null },
  ],
  description: "A person in a dark jacket stands at a front door beside a cardboard delivery box on the step.",
};

const demoCameras = [
  { camera_id: "front", display_name: "前門" },
  { camera_id: "back", display_name: "後院" },
];

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

const cameraBar = document.querySelector("#cameraBar");
const cameraTabs = document.querySelector("#cameraTabs");

const classFilterList = document.querySelector("#classFilterList");
const viewerMinConf = document.querySelector("#viewerMinConf");
const viewerMinConfValue = document.querySelector("#viewerMinConfValue");
const saveFrameButton = document.querySelector("#saveFrameButton");
const saveFrameHint = document.querySelector("#saveFrameHint");

const state = {
  ws: null,
  pendingFrame: null,
  reconnectTimer: null,
  // camera_id -> pane. Insertion order drives the grid layout, so it mirrors
  // the server's CAMERAS order and stays stable while a camera is offline.
  panes: new Map(),
  cameras: [],
  multiCamera: false,
  // The camera enlarged to fill the stage; null = grid view. Zone editing needs
  // an unambiguous pane, so it is only offered when exactly one pane is showing.
  focus: null,
  layoutSignature: null,
  lastCameraId: null,
  // Which channel this tab draws: "yolo" (per-frame, tracked) or "vlm"
  // (periodic open-vocab). Both ride the same JPEG underlay.
  channel: "yolo",
  // Mirrors the server's active detect preset so the task switch can restart
  // the progress bar without inventing a mode.
  detectMode: "fast",
  zonesByCamera: {},
  editor: { active: false, draft: [] },
  // Viewer-only display filters (never touch the backend / detector).
  knownClasses: new Set(),
  hiddenClasses: new Set(),
  viewerMinConf: 0,
};

function staticAsset(name) {
  return new URL(name, import.meta.url).href;
}

function socketUrl(path) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${path}`;
}

// When access control is on and the session cookie lapses, the server answers
// fetches with 401 and closes sockets with 1008; bounce back to the login page.
function redirectToLogin() {
  if (demoMode) {
    return;
  }
  window.location.href = `/login?next=${encodeURIComponent(window.location.pathname)}`;
}

function setChip(element, text, tone) {
  element.textContent = text;
  element.classList.remove("good", "warn", "bad");
  element.classList.add(tone);
}

// --- panes --------------------------------------------------------------------

// The primary pane wraps the markup already in viewer.html, so a single-camera
// viewer never builds any DOM at runtime and behaves exactly as before. It is
// always pane index 0 and is re-labelled rather than replaced when the server
// names the first camera (the bootstrap "" id becomes e.g. "default").
let primaryPane = null;

function newPane(root, paneImage, paneOverlay, paneEmpty, paneLabel) {
  const pane = {
    cameraId: "",
    displayName: "",
    root,
    image: paneImage,
    overlay: paneOverlay,
    empty: paneEmpty,
    label: paneLabel,
    detection: null,
    // Latest VLM payload for this camera (open-vocab boxes + Phase 2 caption).
    vlm: null,
    imageUrl: null,
    zoneCounts: {},
  };
  pane.image.addEventListener("load", () => {
    pane.empty.hidden = true;
  });
  // Click a pane to enlarge it; click the enlarged pane to go back to the grid.
  pane.root.addEventListener("click", (event) => {
    if (state.editor.active || !state.multiCamera) {
      return;
    }
    event.preventDefault();
    setFocus(state.focus === pane.cameraId ? null : pane.cameraId);
  });
  return pane;
}

function createPrimaryPane() {
  return newPane(
    primaryPaneEl,
    image,
    overlay,
    emptyState,
    primaryPaneEl.querySelector(".pane-label"),
  );
}

function createExtraPane(displayName) {
  const root = document.createElement("div");
  root.className = "viewer-pane";
  const paneImage = document.createElement("img");
  paneImage.className = "pane-image";
  paneImage.alt = `Detection preview (${displayName})`;
  const paneOverlay = document.createElement("canvas");
  paneOverlay.className = "pane-overlay";
  const paneEmpty = document.createElement("div");
  paneEmpty.className = "empty-state";
  paneEmpty.textContent = "Waiting for frames";
  const paneLabel = document.createElement("span");
  paneLabel.className = "pane-label";
  root.append(paneImage, paneOverlay, paneEmpty, paneLabel);
  stage.append(root);
  return newPane(root, paneImage, paneOverlay, paneEmpty, paneLabel);
}

function rebindPane(pane, cameraId, displayName) {
  pane.cameraId = cameraId;
  pane.displayName = displayName;
  pane.root.dataset.cameraId = cameraId;
  if (pane.label) {
    pane.label.textContent = displayName || cameraId;
    pane.label.hidden = false;
  }
}

function syncPanes(cameras) {
  const wanted = cameras.length ? cameras : [{ camera_id: "", display_name: "" }];
  const next = new Map();

  wanted.forEach((camera, index) => {
    const cameraId = camera.camera_id || "";
    const displayName = camera.display_name || cameraId;
    let pane;
    if (index === 0) {
      primaryPane = primaryPane || createPrimaryPane();
      pane = primaryPane;
    } else {
      pane = state.panes.get(cameraId);
      if (!pane || pane === primaryPane) {
        pane = createExtraPane(displayName);
      }
    }
    rebindPane(pane, cameraId, displayName);
    next.set(cameraId, pane);
  });

  for (const pane of state.panes.values()) {
    if (pane === primaryPane || next.has(pane.cameraId)) {
      continue;
    }
    releasePaneUrl(pane);
    pane.root.remove();
  }
  state.panes = next;

  state.multiCamera = state.panes.size > 1;
  if (state.focus && !state.panes.has(state.focus)) {
    state.focus = null;
  }
  const signature = `${Array.from(state.panes.keys()).join("|")}#${state.focus ?? ""}`;
  if (signature !== state.layoutSignature) {
    state.layoutSignature = signature;
    applyFocus();
    renderCameraTabs();
    syncZoneEditorAvailability();
  }
}

function setFocus(cameraId) {
  state.focus = cameraId;
  state.layoutSignature = `${Array.from(state.panes.keys()).join("|")}#${cameraId ?? ""}`;
  applyFocus();
  renderCameraTabs();
  syncZoneEditorAvailability();
  renderZoneList();
  updateVlmCaption();
}

function applyFocus() {
  stage.classList.toggle("is-focused", Boolean(state.focus));
  for (const [cameraId, pane] of state.panes) {
    pane.root.classList.toggle("is-focused", cameraId === state.focus);
  }
  // Square-ish grid: 2 cameras -> 1x2, 3-4 -> 2x2, up to 16 -> 4x4. Driven from
  // JS so any MAX_CAMERAS value lays out sensibly without a CSS rule per count.
  const visible = state.focus ? 1 : state.panes.size;
  const columns = Math.max(1, Math.ceil(Math.sqrt(visible)));
  stage.dataset.count = String(visible);
  stage.style.gridTemplateColumns = `repeat(${columns}, 1fr)`;
  stage.style.gap = visible > 1 ? "2px" : "0";
}

function renderCameraTabs() {
  if (!cameraBar || !cameraTabs) {
    return;
  }
  cameraBar.hidden = !state.multiCamera;
  if (!state.multiCamera) {
    return;
  }
  cameraTabs.textContent = "";
  const entries = [{ id: null, name: "全部" }].concat(
    Array.from(state.panes.values()).map((pane) => ({
      id: pane.cameraId,
      name: pane.displayName || pane.cameraId,
    })),
  );
  for (const entry of entries) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "filter-chip";
    chip.textContent = entry.name;
    const active = state.focus === entry.id;
    chip.classList.toggle("is-off", !active);
    chip.setAttribute("aria-pressed", active ? "true" : "false");
    chip.addEventListener("click", () => setFocus(entry.id));
    cameraTabs.append(chip);
  }
}

// The pane whose numbers the metrics panel and the save/zone tools act on: the
// focused one, or the only one, else whichever camera last delivered a frame.
function activePane() {
  if (state.focus) {
    return state.panes.get(state.focus) || null;
  }
  if (state.panes.size === 1) {
    return state.panes.values().next().value;
  }
  return state.panes.get(state.lastCameraId) || null;
}

// A single pane is showing, so a click on the stage maps to exactly one camera.
function soloPane() {
  if (state.panes.size === 1) {
    return state.panes.values().next().value;
  }
  return state.focus ? state.panes.get(state.focus) || null : null;
}

function visiblePanes() {
  const solo = state.focus ? state.panes.get(state.focus) : null;
  return solo ? [solo] : Array.from(state.panes.values());
}

function releasePaneUrl(pane) {
  if (pane.imageUrl) {
    URL.revokeObjectURL(pane.imageUrl);
    pane.imageUrl = null;
  }
}

function releaseImageUrls() {
  for (const pane of state.panes.values()) {
    releasePaneUrl(pane);
  }
}

// --- transport ----------------------------------------------------------------

function connectViewer() {
  if (demoMode) {
    renderDemoViewer();
    return;
  }

  setChip(viewerSocketStatus, "viewer connecting", "warn");
  const path = pinnedCamera
    ? `/ws/viewer?camera_id=${encodeURIComponent(pinnedCamera)}`
    : "/ws/viewer";
  const ws = new WebSocket(socketUrl(path));
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
    if (payload.type === "vlm") {
      handleVlm(payload);
      return;
    }
    if (payload.type === "status") {
      renderStatus(payload.status);
    }
  });

  ws.addEventListener("close", (event) => {
    if (state.ws === ws) {
      state.ws = null;
    }
    state.pendingFrame = null;
    // 1008 (policy violation) is how the server rejects an unauthenticated WS —
    // the session cookie is missing or expired, so send the user to log in.
    if (event.code === 1008) {
      redirectToLogin();
      return;
    }
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
  // The static demo has no server, so switching heads would silently do nothing.
  for (const button of taskButtons) {
    button.disabled = true;
  }
  if (zoneEditToggle) {
    zoneEditToggle.disabled = true;
  }
  syncPanes(demoCameras);
  renderVlmChannel({ enabled: true });
  state.zonesByCamera = { front: demoZones };
  renderZoneList();
  droppedMetric.textContent = "0";
  recordingMetric.textContent = "0";
  uploadMetric.textContent = "0";
  const frames = [
    ["front", demoDetection],
    ["back", demoDetectionBack],
  ];
  for (const [cameraId, detection] of frames) {
    const pane = state.panes.get(cameraId);
    if (!pane) {
      continue;
    }
    pane.detection = detection;
    pane.zoneCounts = detection.zone_counts || {};
    pane.image.src = staticAsset("demo-frame.svg");
    pane.empty.hidden = true;
    collectClasses(detection);
  }
  const frontPane = state.panes.get("front");
  if (frontPane) {
    frontPane.vlm = demoVlm;
  }
  updateVlmCaption();
  state.lastCameraId = "front";
  renderMetrics(demoDetection);
  renderError("");
}

function renderFrameBytes(data) {
  const frame = state.pendingFrame;
  if (!frame) {
    return;
  }
  state.pendingFrame = null;

  const pane = state.panes.get(frame.camera_id || "") || activePane();
  if (!pane) {
    return;
  }

  const blob =
    data instanceof Blob
      ? data
      : new Blob([data], { type: frame.content_type || "image/jpeg" });
  const nextUrl = URL.createObjectURL(blob);
  const previousUrl = pane.imageUrl;
  pane.imageUrl = nextUrl;
  pane.image.src = nextUrl;
  if (previousUrl) {
    URL.revokeObjectURL(previousUrl);
  }

  pane.detection = frame.detection;
  pane.zoneCounts = frame.detection.zone_counts || {};
  pane.empty.hidden = true;
  state.lastCameraId = pane.cameraId;
  collectClasses(frame.detection);
  const active = activePane();
  if (!active || active === pane) {
    renderMetrics(frame.detection);
    renderError(frame.detection.error || "");
    updateVlmCaption();
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

function renderCameraLink(streams) {
  if (streams.length > 1) {
    const online = streams.filter((stream) => stream.camera_connected).length;
    setChip(
      cameraLinkStatus,
      `${online}/${streams.length} cameras`,
      online === streams.length ? "good" : online ? "warn" : "bad",
    );
    return;
  }
  const connected = streams.some((stream) => stream.camera_connected);
  setChip(
    cameraLinkStatus,
    connected ? "phone connected" : "phone idle",
    connected ? "good" : "warn",
  );
}

function renderStatus(status) {
  const streams = Object.values(status.cameras || {}).filter(
    (stream) => !pinnedCamera || stream.camera_id === pinnedCamera,
  );
  syncPanes(
    streams.map((stream) => ({
      camera_id: stream.camera_id,
      display_name: stream.display_name,
    })),
  );
  renderCameraLink(streams);
  renderPhoneStorage(streams.length === 1 ? streams[0] : status);
  renderModel(status.detector || {});
  renderVlmChannel(status.vlm);
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
  // Saved zones are server-owned and per camera; the in-progress draft lives in
  // state.editor, so refreshing this list never disturbs an active drawing.
  state.zonesByCamera = (zones && zones.cameras) || {};
  renderZoneList();
}

function zonesFor(cameraId) {
  return state.zonesByCamera[cameraId] || state.zonesByCamera[""] || [];
}

// Keep a just-fired alert on screen briefly instead of letting the 1s status
// poll immediately reset the chip back to its steady "armed" state.
const alertFlash = { until: 0 };

function handleAlert(event) {
  if (!alertStatus) {
    return;
  }
  const pane = event.camera_id ? state.panes.get(event.camera_id) : null;
  const prefix = state.multiCamera && pane ? `${pane.displayName} · ` : "";
  setChip(alertStatus, `⚠ ${prefix}${event.rule} ×${event.count}`, "bad");
  alertFlash.until = Date.now() + 6000;
  notifyBrowser(event, pane);
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

function notifyBrowser(event, pane) {
  if (typeof Notification === "undefined") {
    return;
  }
  const where = pane ? `${pane.displayName}: ` : "";
  const show = () => {
    try {
      new Notification("YOLO Elf", { body: `${where}${event.rule} ×${event.count}` });
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

// Stash the newest VLM payload on its camera's pane. Drawn only while the VLM
// channel is active; the box drawing loop reads pane.vlm via activeDetection().
function handleVlm(payload) {
  const pane = state.panes.get(payload.camera_id || "") || activePane();
  if (!pane) {
    return;
  }
  pane.vlm = payload;
  updateVlmCaption();
}

// The scene caption bar tracks the active pane's latest VLM description, and
// only while the VLM channel is showing. Called on new payloads and whenever
// the active pane or channel changes.
function updateVlmCaption() {
  if (!vlmCaption) {
    return;
  }
  const pane = activePane();
  const text =
    state.channel === "vlm" && pane && pane.vlm ? pane.vlm.description || "" : "";
  vlmCaption.textContent = text;
  vlmCaption.hidden = !text;
}

// Reveal the channel toggle once the server reports the VLM channel is on. When
// it is off, force back to the YOLO channel so a stale toggle can't blank the
// stage with an empty VLM overlay.
function renderVlmChannel(vlm) {
  const enabled = Boolean(vlm && vlm.enabled);
  if (channelGroupControl) {
    channelGroupControl.hidden = !enabled;
  }
  if (!enabled && state.channel !== "yolo") {
    setChannel("yolo");
  }
}

function setChannel(channel) {
  state.channel = channel === "vlm" ? "vlm" : "yolo";
  for (const button of channelButtons) {
    button.setAttribute("aria-pressed", button.dataset.channel === state.channel ? "true" : "false");
  }
  updateVlmCaption();
}

function renderModel(detector) {
  const modelText = detector.loaded ? detector.model : "model not loaded";
  setChip(modelStatus, modelText, detector.last_load_error ? "bad" : detector.loaded ? "good" : "warn");
  if (detector.mode) {
    state.detectMode = detector.mode;
  }
  renderDetectMode(detector.mode);
  renderDetectTask(detector.task);
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

function renderDetectTask(task) {
  if (!task) {
    return;
  }
  for (const button of taskButtons) {
    button.setAttribute("aria-pressed", button.dataset.detectTask === task ? "true" : "false");
  }
  // Fast/accurate is a detect-only axis: the other heads have a single
  // checkpoint each, so the toggle would be a no-op that looks like a control.
  modeGroup?.classList.toggle("is-disabled", task !== "detect");
  for (const button of modeButtons) {
    button.disabled = task !== "detect";
  }
}

async function setDetectTask(task) {
  renderDetectTask(task);
  // Each task is its own checkpoint, so reuse the model-switch progress bar.
  modelSwitch.begin(state.detectMode || "fast");
  setChip(modelStatus, `switching to ${task}…`, "warn");
  try {
    const response = await fetch("/api/detector/task", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ task }),
    });
    if (response.ok) {
      const payload = await response.json();
      renderModel(payload.detector || {});
    }
  } catch {
    // The status poll will reconcile the chips on the next tick.
  }
}

// Legend for the raster channels: class swatches for semantic, the metre range
// for depth. Hidden entirely for the box tasks, which have their own labels.
function renderRasterLegend(detection) {
  if (!rasterLegend) {
    return;
  }
  const raster = detection?.raster;
  if (!raster) {
    rasterLegend.hidden = true;
    rasterLegend.textContent = "";
    return;
  }
  rasterLegend.hidden = false;
  if (raster.kind === "depth") {
    const low = raster.min_m;
    const high = raster.max_m;
    rasterLegend.textContent =
      low == null ? "深度：無有效值" : `深度 ${low.toFixed(2)}–${high.toFixed(2)} m（亮＝近）`;
    return;
  }
  rasterLegend.textContent = "";
  for (const entry of (raster.legend || []).slice(0, 12)) {
    const chip = document.createElement("span");
    chip.className = "legend-chip";
    const swatch = document.createElement("i");
    swatch.style.background = `rgb(${entry.color.join(",")})`;
    chip.append(swatch, document.createTextNode(entry.label));
    rasterLegend.append(chip);
  }
}

function renderMetrics(detection) {
  frameMetric.textContent = String(detection.frame_id ?? "-");
  boxesMetric.textContent = String(detection.boxes?.length ?? 0);
  inferenceMetric.textContent = `${detection.inference_ms ?? 0} ms`;
  renderRasterLegend(detection);
}

// Class chips are built from labels seen on the stream. New labels default to
// visible; the chip set only re-renders when the known set actually grows, so a
// user's toggles survive across frames. Labels pool across cameras on purpose —
// hiding "car" should hide it everywhere.
function collectClasses(detection) {
  let grew = false;
  for (const box of detection.boxes || []) {
    if (box.label && !state.knownClasses.has(box.label)) {
      state.knownClasses.add(box.label);
      grew = true;
    }
  }
  if (grew) {
    renderClassChips();
  }
}

function renderClassChips() {
  if (!classFilterList) {
    return;
  }
  classFilterList.textContent = "";
  if (!state.knownClasses.size) {
    const empty = document.createElement("span");
    empty.className = "settings-hint";
    empty.textContent = "尚無偵測類別";
    classFilterList.append(empty);
    return;
  }
  for (const label of Array.from(state.knownClasses).sort()) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "filter-chip";
    chip.textContent = label;
    const shown = !state.hiddenClasses.has(label);
    chip.classList.toggle("is-off", !shown);
    chip.setAttribute("aria-pressed", shown ? "true" : "false");
    chip.addEventListener("click", () => {
      if (state.hiddenClasses.has(label)) {
        state.hiddenClasses.delete(label);
      } else {
        state.hiddenClasses.add(label);
      }
      renderClassChips();
    });
    classFilterList.append(chip);
  }
}

function setViewerMinConf(value) {
  state.viewerMinConf = clamp01(Number(value) || 0);
  if (viewerMinConfValue) {
    viewerMinConfValue.textContent = `${Math.round(state.viewerMinConf * 100)}%`;
  }
}

// Composite the current frame + overlay at the source resolution (scale 1, no
// letterbox offset) so the PNG matches the model's pixels, not the stretched
// stage. Honors the same class / confidence filters the viewer is showing.
function saveSnapshot() {
  const pane = activePane();
  const detection = pane?.detection;
  if (!detection || !detection.width || !detection.height) {
    return;
  }
  try {
    const canvas = document.createElement("canvas");
    canvas.width = detection.width;
    canvas.height = detection.height;
    const ctx = canvas.getContext("2d");
    const fit = { scale: 1, x: 0, y: 0 };
    ctx.drawImage(pane.image, 0, 0, detection.width, detection.height);
    // Same layer order as the live overlay, so the PNG matches what is on
    // screen — including the semantic/depth raster, which is the *entire*
    // result for those tasks and would otherwise save as a bare frame.
    drawRaster(ctx, pane, detection, fit);
    drawZones(ctx, pane, detection, fit);
    drawBoxes(ctx, detection, fit);
    canvas.toBlob((blob) => {
      if (!blob) {
        showSaveHint("存圖失敗");
        return;
      }
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      const suffix = state.multiCamera ? `${pane.cameraId}-` : "";
      link.download = `yolo-elf-frame-${suffix}${detection.frame_id ?? 0}.png`;
      document.body.append(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }, "image/png");
  } catch {
    // A cross-origin / tainted frame makes toBlob throw SecurityError.
    showSaveHint("存圖失敗（畫面受保護）");
  }
}

function showSaveHint(message) {
  if (!saveFrameHint) {
    return;
  }
  saveFrameHint.textContent = message;
  saveFrameHint.hidden = !message;
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

function resizeOverlay(pane) {
  const rect = pane.overlay.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  pane.overlay.width = Math.max(1, Math.round(rect.width * dpr));
  pane.overlay.height = Math.max(1, Math.round(rect.height * dpr));
}

function resizeOverlays() {
  for (const pane of state.panes.values()) {
    resizeOverlay(pane);
  }
}

function drawOverlay() {
  // Only visible panes are drawn — a hidden pane has a zero-sized rect, so
  // fitContain would divide by zero and paint nothing useful anyway.
  for (const pane of visiblePanes()) {
    drawPane(pane);
  }
  requestAnimationFrame(drawOverlay);
}

// The detection the active channel draws. The YOLO channel uses the per-frame
// tracked boxes; the VLM channel uses the periodic open-vocab payload. Both
// carry width/height so the overlay letterboxes onto the same JPEG underlay.
function activeDetection(pane) {
  return state.channel === "vlm" ? pane.vlm : pane.detection;
}

function drawPane(pane) {
  resizeOverlay(pane);
  const ctx = pane.overlay.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = pane.overlay.width / dpr;
  const height = pane.overlay.height / dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const yoloChannel = state.channel === "yolo";
  const detection = activeDetection(pane);
  if (detection && detection.width > 0 && detection.height > 0) {
    const fit = fitContain(width, height, detection.width, detection.height);
    // Semantic/depth rasters sit under everything else: they cover the whole
    // frame, so drawing them last would bury the boxes and zones.
    drawRaster(ctx, pane, detection, fit);
    // Zones and the zone-drawing draft belong to the YOLO channel only (VLM
    // has no confidence/tracking to feed them).
    if (yoloChannel) {
      drawZones(ctx, pane, detection, fit);
    }
    drawBoxes(ctx, detection, fit);
    if (yoloChannel && soloPane() === pane) {
      drawDraft(ctx, detection, fit);
    }
  }
}

// Decoding a fresh data URL every frame would burn the frame budget, so each
// pane keeps the decoded Image until the payload's PNG actually changes. The
// rAF loop redraws continuously, so a frame decoded a beat late simply appears
// on the next pass rather than needing a load callback.
function rasterImage(pane, dataUrl) {
  if (pane.rasterSrc !== dataUrl) {
    const image = new Image();
    image.src = dataUrl;
    pane.rasterSrc = dataUrl;
    pane.rasterImage = image;
  }
  return pane.rasterImage;
}

function drawRaster(ctx, pane, detection, fit) {
  const raster = detection.raster;
  if (!raster || !raster.png) {
    return;
  }
  const image = rasterImage(pane, raster.png);
  if (!image.complete || !image.naturalWidth) {
    return;
  }
  ctx.save();
  ctx.globalAlpha = RASTER_ALPHA;
  // The class map is nearest-sampled server-side so its palette stays exact;
  // smoothing it back up here would invent colours between two classes.
  ctx.imageSmoothingEnabled = raster.kind === "depth";
  ctx.drawImage(
    image,
    fit.x,
    fit.y,
    detection.width * fit.scale,
    detection.height * fit.scale,
  );
  ctx.restore();
}

// Viewer-side visibility: hidden classes and a min-confidence floor are pure
// display filters — the detector still runs on everything.
function isBoxVisible(box) {
  if (state.hiddenClasses.has(box.label)) {
    return false;
  }
  return (box.confidence ?? 0) >= state.viewerMinConf;
}

function drawBoxes(ctx, detection, fit) {
  ctx.lineWidth = 3;
  ctx.font = "600 14px system-ui, sans-serif";
  ctx.textBaseline = "top";

  for (const box of detection.boxes || []) {
    if (!isBoxVisible(box)) {
      continue;
    }
    const color = colorForClass(box.class_id);
    const [x1, y1, x2, y2] = box.xyxy;
    const left = fit.x + x1 * fit.scale;
    const top = fit.y + y1 * fit.scale;
    const right = fit.x + x2 * fit.scale;
    const bottom = fit.y + y2 * fit.scale;

    // Segmentation contour, drawn before the outline so the label stays legible
    // on top of the tint.
    drawMask(ctx, box, fit, color);
    // When the second-stage classifier named a species, show that as the
    // primary label (圖鑑); otherwise fall back to the detection class. A
    // tracker id (when present) is prefixed so an object is followable.
    // Ids are per-camera: #3 on one pane is unrelated to #3 on another.
    const idPrefix = box.track_id != null ? `#${box.track_id} ` : "";
    const label = box.species
      ? `${idPrefix}${box.species} ${Math.round((box.species_confidence ?? 0) * 100)}%`
      : `${idPrefix}${box.label} ${(box.confidence * 100).toFixed(0)}%`;

    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    if (box.obb && box.obb.length === 8) {
      // Rotated detection: outline the real quad. The axis-aligned xyxy is still
      // what positions the label (and what zones/alerts upstream reason about).
      ctx.beginPath();
      for (let i = 0; i < 8; i += 2) {
        const px = fit.x + box.obb[i] * fit.scale;
        const py = fit.y + box.obb[i + 1] * fit.scale;
        if (i === 0) {
          ctx.moveTo(px, py);
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.closePath();
      ctx.stroke();
    } else {
      ctx.strokeRect(left, top, right - left, bottom - top);
    }

    drawKeypoints(ctx, box, fit, color);

    const textWidth = ctx.measureText(label).width + 12;
    const textTop = Math.max(0, top - 24);
    ctx.fillStyle = color;
    ctx.fillRect(left, textTop, textWidth, 22);
    ctx.fillStyle = "#10100f";
    ctx.fillText(label, left + 6, textTop + 3);
  }
}

function drawMask(ctx, box, fit, color) {
  const mask = box.mask;
  if (!mask || mask.length < 3) {
    return;
  }
  ctx.save();
  ctx.beginPath();
  for (let i = 0; i < mask.length; i += 1) {
    const px = fit.x + mask[i][0] * fit.scale;
    const py = fit.y + mask[i][1] * fit.scale;
    if (i === 0) {
      ctx.moveTo(px, py);
    } else {
      ctx.lineTo(px, py);
    }
  }
  ctx.closePath();
  ctx.fillStyle = withAlpha(color, 0.3);
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = color;
  ctx.stroke();
  ctx.restore();
}

function drawKeypoints(ctx, box, fit, color) {
  const points = box.keypoints;
  if (!points || !points.length) {
    return;
  }
  const toStage = (point) => [fit.x + point[0] * fit.scale, fit.y + point[1] * fit.scale];
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  // Limbs first, joints on top, so a dot is never half-hidden under a bone.
  // A keypoint the model could not see comes back at (0, 0) with ~0 confidence;
  // drawing those would rope every occluded joint back to the frame corner.
  for (const [from, to] of POSE_SKELETON) {
    const a = points[from];
    const b = points[to];
    if (!a || !b || a[2] < KEYPOINT_MIN_CONF || b[2] < KEYPOINT_MIN_CONF) {
      continue;
    }
    const [ax, ay] = toStage(a);
    const [bx, by] = toStage(b);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }
  ctx.fillStyle = color;
  for (const point of points) {
    if (point[2] < KEYPOINT_MIN_CONF) {
      continue;
    }
    const [px, py] = toStage(point);
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
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
// the same letterboxed rectangle the frame is drawn in. Each pane computes its
// own fit, so a grid cell with a different aspect still lands correctly.
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

function drawZones(ctx, pane, detection, fit) {
  const zones = zonesFor(pane.cameraId);
  if (!zones.length) {
    return;
  }
  ctx.lineWidth = 2;
  ctx.font = "600 13px system-ui, sans-serif";
  ctx.textBaseline = "top";

  zones.forEach((zone, index) => {
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

    const count = pane.zoneCounts?.[zone.name] ?? 0;
    const label = `${zone.name} · ${count}`;
    const [lx, ly] = points[0];
    const top = Math.max(0, ly - 22);
    ctx.fillStyle = color;
    ctx.fillRect(lx, top, ctx.measureText(label).width + 12, 20);
    ctx.fillStyle = "#10100f";
    ctx.fillText(label, lx + 6, top + 3);
  });
}

function drawDraft(ctx, detection, fit) {
  const draft = state.editor.draft;
  if (!state.editor.active || !draft.length) {
    return;
  }
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

// --- zone editor --------------------------------------------------------------

function stageToNormalized(pane, clientX, clientY) {
  const detection = pane?.detection;
  if (!detection || !detection.width || !detection.height) {
    return null;
  }
  const rect = pane.overlay.getBoundingClientRect();
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
  const pane = soloPane();
  zoneDraftHint.textContent = pane?.detection
    ? `已加入 ${state.editor.draft.length} 點（至少 3 點）`
    : "等待畫面出現後再點擊繪製";
}

// Drawing needs one unambiguous frame to map clicks into, so editing is offered
// only when a single pane fills the stage. In the grid, pick a camera first.
function syncZoneEditorAvailability() {
  if (!zoneEditToggle || demoMode) {
    return;
  }
  const solo = soloPane();
  zoneEditToggle.disabled = !solo;
  zoneEditToggle.title = solo ? "" : "先點選一路相機再編輯區域";
  if (!solo && state.editor.active) {
    setEditor(false);
  }
}

function setEditor(active) {
  const pane = soloPane();
  if (active && !pane) {
    updateDraftHint("先點選一路相機再編輯區域");
    return;
  }
  state.editor.active = active;
  state.editor.draft = [];
  if (zoneDraftBar) {
    zoneDraftBar.hidden = !active;
  }
  if (zoneEditToggle) {
    zoneEditToggle.textContent = active ? "完成編輯" : "編輯";
  }
  // Overlays are pointer-events:none by default; enable only the pane being
  // edited so normal viewing never intercepts clicks.
  for (const candidate of state.panes.values()) {
    const editing = active && candidate === pane;
    candidate.overlay.style.pointerEvents = editing ? "auto" : "";
    candidate.overlay.style.cursor = editing ? "crosshair" : "";
  }
  updateDraftHint();
}

function onStageClick(event) {
  if (!state.editor.active) {
    return;
  }
  const pane = soloPane();
  const point = stageToNormalized(pane, event.clientX, event.clientY);
  if (!point) {
    updateDraftHint("等待畫面出現後再點擊繪製");
    return;
  }
  event.stopPropagation();
  state.editor.draft.push(point);
  updateDraftHint();
}

function publicZone(zone) {
  return { name: zone.name, points: zone.points, anchor: zone.anchor || "center" };
}

async function finishZone() {
  const pane = soloPane();
  if (!pane) {
    updateDraftHint("先點選一路相機再編輯區域");
    return;
  }
  if (state.editor.draft.length < 3) {
    updateDraftHint("至少需要 3 點");
    return;
  }
  const existing = zonesFor(pane.cameraId);
  const name = (window.prompt("區域名稱", `zone-${existing.length + 1}`) || "").trim();
  if (!name) {
    return;
  }
  const zones = [
    ...existing.map(publicZone),
    { name, points: state.editor.draft.map(([x, y]) => [round4(x), round4(y)]), anchor: "center" },
  ];
  if (await saveZones(pane.cameraId, zones)) {
    state.editor.draft = [];
    updateDraftHint();
  }
}

async function deleteZone(cameraId, name) {
  await saveZones(
    cameraId,
    zonesFor(cameraId)
      .filter((zone) => zone.name !== name)
      .map(publicZone),
  );
}

async function saveZones(cameraId, zones) {
  try {
    const response = await fetch("/api/zones", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ zones, camera_id: cameraId || undefined }),
    });
    if (!response.ok) {
      updateDraftHint("儲存失敗，請檢查頂點");
      return false;
    }
    const payload = await response.json();
    state.zonesByCamera = (payload.zones && payload.zones.cameras) || {};
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
  // In the grid, list every camera's zones (prefixed) so nothing is hidden;
  // focused on one camera, list just that camera's.
  const pane = soloPane();
  const cameraIds = pane
    ? [pane.cameraId]
    : Array.from(state.panes.keys()).filter((id) => zonesFor(id).length);
  for (const cameraId of cameraIds) {
    for (const zone of zonesFor(cameraId)) {
      const item = document.createElement("li");
      item.className = "zone-list-item";
      const label = document.createElement("span");
      const prefix =
        state.multiCamera && !pane
          ? `${state.panes.get(cameraId)?.displayName || cameraId} · `
          : "";
      label.textContent = `${prefix}${zone.name} (${(zone.points || []).length})`;
      item.append(label);
      if (!demoMode) {
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "compact-button";
        remove.textContent = "刪除";
        remove.addEventListener("click", (event) => {
          event.stopPropagation();
          deleteZone(cameraId, zone.name);
        });
        item.append(remove);
      }
      zoneList.append(item);
    }
  }
}

async function pollStatus() {
  if (demoMode) {
    return;
  }

  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (response.status === 401) {
      redirectToLogin();
      return;
    }
    if (response.ok) {
      renderStatus(await response.json());
    }
  } catch {
    setChip(cameraLinkStatus, "status unavailable", "bad");
  } finally {
    setTimeout(pollStatus, 1000);
  }
}

for (const button of modeButtons) {
  button.addEventListener("click", () => setDetectMode(button.dataset.detectMode));
}

for (const button of channelButtons) {
  button.addEventListener("click", () => setChannel(button.dataset.channel));
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
stage.addEventListener("click", onStageClick);
for (const button of taskButtons) {
  button.addEventListener("click", () => setDetectTask(button.dataset.detectTask));
}

viewerMinConf?.addEventListener("input", (event) => setViewerMinConf(event.target.value));
saveFrameButton?.addEventListener("click", saveSnapshot);

window.addEventListener("resize", resizeOverlays);
window.addEventListener("beforeunload", releaseImageUrls);

// Bootstrap with the single implicit pane so the first paint (and the entire
// single-camera path) needs no round trip; /api/status reconciles the rest.
syncPanes([]);
renderClassChips();
if (viewerMinConf) {
  setViewerMinConf(viewerMinConf.value);
}
connectViewer();
pollStatus();
requestAnimationFrame(drawOverlay);
