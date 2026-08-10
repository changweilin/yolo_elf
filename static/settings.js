const form = document.querySelector("#settingsForm");
const stateChip = document.querySelector("#settingsState");
const errorLine = document.querySelector("#settingsError");
const modeGroup = document.querySelector("#modeGroup");
const modeButtons = Array.from(modeGroup.querySelectorAll("[data-detect-mode]"));
const fastModelInput = document.querySelector("#fastModelInput");
const accurateModelInput = document.querySelector("#accurateModelInput");
const classesInput = document.querySelector("#classesInput");
const classifierModelInput = document.querySelector("#classifierModelInput");
const classifierMinConfInput = document.querySelector("#classifierMinConfInput");
const classifierMaxBoxesInput = document.querySelector("#classifierMaxBoxesInput");
const confInput = document.querySelector("#confInput");
const imgSizeInput = document.querySelector("#imgSizeInput");
const maxDetInput = document.querySelector("#maxDetInput");
const end2endGroup = document.querySelector("#end2endGroup");
const end2endButtons = Array.from(end2endGroup.querySelectorAll("[data-end2end]"));
const saveButton = document.querySelector("#saveButton");
const resetButton = document.querySelector("#resetButton");

const taskStateChip = document.querySelector("#taskState");
const taskError = document.querySelector("#taskError");
const taskHint = document.querySelector("#taskHint");
const taskButtons = Array.from(document.querySelectorAll("[data-detect-task]"));
const taskTabs = Array.from(document.querySelectorAll("[data-task-tab]"));
const taskPanels = {
  box: document.querySelector("#taskPanelBox"),
  raster: document.querySelector("#taskPanelRaster"),
};

// Mirrors app/config.py. Only one raster head may run at a time (each repaints
// every pixel), and the order here is the order tasks are sent to the server.
const BOX_TASKS = ["detect", "segment", "pose", "obb", "openvocab"];
const RASTER_TASKS = ["semantic", "depth"];
const TASK_ORDER = [...BOX_TASKS, ...RASTER_TASKS];
const TASK_LABELS = {
  detect: "物件",
  segment: "分割",
  pose: "姿態",
  obb: "旋轉框",
  openvocab: "開放詞彙",
  semantic: "語意",
  depth: "深度",
};

import { createModelSwitch } from "./mode-switch.js";

const modelSwitch = createModelSwitch({
  progressEl: document.querySelector("#modelSwitchProgress"),
  fillEl: document.querySelector("#modelSwitchFill"),
  lock(on) {
    form?.classList.toggle("is-locked", on);
  },
});

// Its own bar rather than the form's: a task switch is a separate, instant
// action, and pinning its progress under the form would read as a failed save.
const taskSwitch = createModelSwitch({
  progressEl: document.querySelector("#taskSwitchProgress"),
  fillEl: document.querySelector("#taskSwitchFill"),
  lock(on) {
    for (const panel of Object.values(taskPanels)) {
      panel?.classList.toggle("is-locked", on);
    }
  },
});

// `mode` is what the form shows (possibly edited and not yet applied);
// `serverMode` is what the detector is actually on, which is what the task
// switch has to hand the progress bar so it polls for the right preset.
const state = { mode: "fast", serverMode: "fast", end2end: "auto", tasks: ["detect"], maxTasks: 4 };

function setChip(text, tone) {
  stateChip.textContent = text;
  stateChip.classList.remove("good", "warn", "bad");
  stateChip.classList.add(tone);
}

function showError(message) {
  if (!message) {
    errorLine.hidden = true;
    errorLine.textContent = "";
    return;
  }
  errorLine.hidden = false;
  errorLine.textContent = message;
}

function renderMode(mode) {
  if (mode === "fast" || mode === "accurate") {
    state.mode = mode;
  }
  for (const button of modeButtons) {
    button.setAttribute("aria-pressed", button.dataset.detectMode === state.mode ? "true" : "false");
  }
}

function renderEnd2End(mode) {
  if (mode === "auto" || mode === "on" || mode === "off") {
    state.end2end = mode;
  }
  for (const button of end2endButtons) {
    button.setAttribute("aria-pressed", button.dataset.end2end === state.end2end ? "true" : "false");
  }
}

function setTaskChip(text, tone) {
  if (!taskStateChip) {
    return;
  }
  taskStateChip.textContent = text;
  taskStateChip.classList.remove("good", "warn", "bad");
  taskStateChip.classList.add(tone);
}

function showTaskError(message) {
  if (!taskError) {
    return;
  }
  taskError.hidden = !message;
  taskError.textContent = message || "";
}

// Chips are toggled far faster than a round trip, and each POST answers with the
// set *it* installed. Without a sequence number the reply to an older click
// would repaint the chips over a newer one — and so would a config save, whose
// reply also carries the task list. Only the newest request writes the chips.
let taskRequestSeq = 0;
let taskRequestsInFlight = 0;

// `authoritative` marks the reply to the newest task switch: it may write the
// chips even while another request is still open.
function renderTasks(detector, authoritative = false) {
  if (Number.isFinite(detector.max_active_tasks)) {
    state.maxTasks = detector.max_active_tasks;
  }
  if (!authoritative && taskRequestsInFlight > 0) {
    return;
  }
  // `tasks` is the multi-head list; older servers only report `task`.
  const tasks = detector.tasks || (detector.task ? [detector.task] : []);
  if (!Array.isArray(tasks) || tasks.length === 0) {
    return;
  }
  renderTaskChips(tasks);
}

function renderTaskChips(tasks) {
  state.tasks = tasks;
  const active = new Set(tasks);
  const atCap = tasks.length >= state.maxTasks;
  for (const button of taskButtons) {
    const task = button.dataset.detectTask;
    const on = active.has(task);
    button.setAttribute("aria-pressed", on ? "true" : "false");
    // A raster chip is never capped out: picking it replaces the other raster
    // rather than adding a pass, so the cap cannot be what blocks it.
    button.disabled = !on && atCap && !RASTER_TASKS.includes(task);
  }
  // Fast/accurate is a detect-only axis: the other heads have a single
  // checkpoint each, so the toggle would be a no-op that looks like a control.
  const hasDetect = active.has("detect");
  modeGroup?.classList.toggle("is-disabled", !hasDetect);
  for (const button of modeButtons) {
    button.disabled = !hasDetect;
  }
  renderTaskHint();
}

function renderTaskHint() {
  if (!taskHint) {
    return;
  }
  const names = state.tasks.map((task) => TASK_LABELS[task] || task).join("＋");
  taskHint.textContent =
    state.tasks.length > 1
      ? `同時執行 ${names}：每格影像跑 ${state.tasks.length} 次推論，延遲約為單一任務的 ${state.tasks.length} 倍。`
      : `執行 ${names}。可再勾選其他任務同時疊加，代價是每多一項就多跑一次推論。`;
}

// Which tab's chips are on screen. Purely a display choice: a task stays active
// while its tab is hidden, so a raster underlay can run beneath box overlays.
function setTaskTab(name) {
  for (const tab of taskTabs) {
    tab.setAttribute("aria-selected", tab.dataset.taskTab === name ? "true" : "false");
  }
  for (const [key, panel] of Object.entries(taskPanels)) {
    if (panel) {
      panel.hidden = key !== name;
    }
  }
}

// Checkbox semantics: toggling never replaces the whole set, except between the
// two rasters, which cannot both be drawn. The last active task cannot be
// switched off — the worker always runs something.
function nextTasks(task) {
  const active = new Set(state.tasks);
  if (active.has(task)) {
    if (active.size === 1) {
      return null;
    }
    active.delete(task);
  } else {
    if (RASTER_TASKS.includes(task)) {
      for (const other of RASTER_TASKS) {
        active.delete(other);
      }
    } else if (active.size >= state.maxTasks) {
      return null;
    }
    active.add(task);
  }
  return TASK_ORDER.filter((entry) => active.has(entry));
}

async function setDetectTask(task) {
  const tasks = nextTasks(task);
  if (!tasks) {
    return;
  }
  renderTaskChips(tasks);
  showTaskError("");
  setTaskChip("切換中…", "warn");
  // Each task is its own checkpoint, so reuse the model-switch progress bar.
  taskSwitch.begin(state.serverMode);
  taskRequestSeq += 1;
  taskRequestsInFlight += 1;
  const seq = taskRequestSeq;
  try {
    const response = await fetch("/api/detector/task", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ tasks }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      setTaskChip("切換失敗", "bad");
      showTaskError(payload.detail || "任務組合無效。");
      return;
    }
    renderTasks(payload.detector || {}, seq === taskRequestSeq);
    setTaskChip("已套用", "good");
  } catch {
    setTaskChip("連線失敗", "bad");
    showTaskError("無法連線到伺服器。");
  } finally {
    taskRequestsInFlight -= 1;
  }
}

function populate(detector) {
  if (detector.mode) {
    state.serverMode = detector.mode;
  }
  renderMode(detector.mode);
  renderEnd2End(detector.end2end);
  renderTasks(detector);
  const models = detector.models || {};
  fastModelInput.value = models.fast ?? "";
  accurateModelInput.value = models.accurate ?? "";
  classesInput.value = (detector.configured_classes || []).join(", ");
  classifierModelInput.value = detector.classifier_model ?? "";
  classifierMinConfInput.value = detector.classifier_min_conf ?? "";
  classifierMaxBoxesInput.value = detector.classifier_max_boxes ?? "";
  confInput.value = detector.conf_thresh ?? "";
  imgSizeInput.value = detector.img_size ?? "";
  maxDetInput.value = detector.max_det ?? "";
}

function buildPayload() {
  return {
    mode: state.mode,
    fast_model: fastModelInput.value.trim(),
    accurate_model: accurateModelInput.value.trim(),
    classes: classesInput.value,
    classifier_model: classifierModelInput.value.trim(),
    classifier_min_conf:
      classifierMinConfInput.value === "" ? null : Number(classifierMinConfInput.value),
    classifier_max_boxes:
      classifierMaxBoxesInput.value === "" ? null : Number(classifierMaxBoxesInput.value),
    conf_thresh: confInput.value === "" ? null : Number(confInput.value),
    img_size: imgSizeInput.value === "" ? null : Number(imgSizeInput.value),
    max_det: maxDetInput.value === "" ? null : Number(maxDetInput.value),
    end2end: state.end2end,
  };
}

async function loadConfig() {
  setChip("載入中…", "warn");
  setTaskChip("載入中…", "warn");
  showError("");
  showTaskError("");
  try {
    const response = await fetch("/api/detector/config", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    populate(payload.detector || {});
    setChip("目前設定", "good");
    setTaskChip("目前任務", "good");
  } catch {
    setChip("讀取失敗", "bad");
    setTaskChip("讀取失敗", "bad");
    showError("無法讀取目前設定，請確認伺服器正在執行。");
  }
}

async function saveConfig(event) {
  event.preventDefault();
  saveButton.disabled = true;
  setChip("套用中…", "warn");
  showError("");
  try {
    const response = await fetch("/api/detector/config", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      setChip("套用失敗", "bad");
      showError(payload.detail || "設定無效，請檢查欄位。");
      return;
    }
    populate(payload.detector || {});
    setChip("已套用", "good");
    // Models load lazily/in the background; show the progress bar and freeze
    // the form until the new weights report ready.
    modelSwitch.begin(state.mode);
  } catch {
    setChip("連線失敗", "bad");
    showError("無法連線到伺服器。");
  } finally {
    saveButton.disabled = false;
  }
}

for (const button of modeButtons) {
  button.addEventListener("click", () => renderMode(button.dataset.detectMode));
}
for (const button of end2endButtons) {
  button.addEventListener("click", () => renderEnd2End(button.dataset.end2end));
}
for (const button of taskButtons) {
  button.addEventListener("click", () => setDetectTask(button.dataset.detectTask));
}
for (const tab of taskTabs) {
  tab.addEventListener("click", () => setTaskTab(tab.dataset.taskTab));
}
form.addEventListener("submit", saveConfig);
resetButton.addEventListener("click", loadConfig);

renderTaskHint();
loadConfig();
