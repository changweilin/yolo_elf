const form = document.querySelector("#settingsForm");
const stateChip = document.querySelector("#settingsState");
const errorLine = document.querySelector("#settingsError");
const modeGroup = document.querySelector("#modeGroup");
const modeButtons = Array.from(modeGroup.querySelectorAll("[data-detect-mode]"));
const fastModelInput = document.querySelector("#fastModelInput");
const accurateModelInput = document.querySelector("#accurateModelInput");
const classesInput = document.querySelector("#classesInput");
const confInput = document.querySelector("#confInput");
const imgSizeInput = document.querySelector("#imgSizeInput");
const saveButton = document.querySelector("#saveButton");
const resetButton = document.querySelector("#resetButton");

import { createModelSwitch } from "./mode-switch.js";

const modelSwitch = createModelSwitch({
  progressEl: document.querySelector("#modelSwitchProgress"),
  fillEl: document.querySelector("#modelSwitchFill"),
  lock(on) {
    form?.classList.toggle("is-locked", on);
  },
});

const state = { mode: "fast" };

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

function populate(detector) {
  renderMode(detector.mode);
  const models = detector.models || {};
  fastModelInput.value = models.fast ?? "";
  accurateModelInput.value = models.accurate ?? "";
  classesInput.value = (detector.configured_classes || []).join(", ");
  confInput.value = detector.conf_thresh ?? "";
  imgSizeInput.value = detector.img_size ?? "";
}

function buildPayload() {
  return {
    mode: state.mode,
    fast_model: fastModelInput.value.trim(),
    accurate_model: accurateModelInput.value.trim(),
    classes: classesInput.value,
    conf_thresh: confInput.value === "" ? null : Number(confInput.value),
    img_size: imgSizeInput.value === "" ? null : Number(imgSizeInput.value),
  };
}

async function loadConfig() {
  setChip("載入中…", "warn");
  showError("");
  try {
    const response = await fetch("/api/detector/config", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    populate(payload.detector || {});
    setChip("目前設定", "good");
  } catch {
    setChip("讀取失敗", "bad");
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
form.addEventListener("submit", saveConfig);
resetButton.addEventListener("click", loadConfig);

loadConfig();
