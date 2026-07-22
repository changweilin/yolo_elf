const filters = document.querySelector("#historyFilters");
const stateChip = document.querySelector("#historyState");
const errorLine = document.querySelector("#historyError");
const labelFilter = document.querySelector("#labelFilter");
const zoneFilter = document.querySelector("#zoneFilter");
const rangeButtons = Array.from(document.querySelectorAll("[data-range]"));
const rows = document.querySelector("#historyRows");
const emptyHint = document.querySelector("#historyEmpty");

// Default window: the last hour. 0 means "all time" (no lower bound).
const state = { rangeSec: 3600 };

function setChip(text, tone) {
  stateChip.textContent = text;
  stateChip.classList.remove("good", "warn", "bad");
  stateChip.classList.add(tone);
}

function showError(message) {
  errorLine.hidden = !message;
  errorLine.textContent = message || "";
}

function renderRange() {
  for (const button of rangeButtons) {
    const active = Number(button.dataset.range) === state.rangeSec;
    button.setAttribute("aria-pressed", active ? "true" : "false");
    button.classList.toggle("is-active", active);
  }
}

function formatTime(epochSec) {
  if (!epochSec) {
    return "—";
  }
  return new Date(epochSec * 1000).toLocaleString();
}

function buildQuery() {
  const params = new URLSearchParams({ limit: "500" });
  if (state.rangeSec > 0) {
    params.set("since", String(Math.floor(Date.now() / 1000 - state.rangeSec)));
  }
  const label = labelFilter.value.trim();
  const zone = zoneFilter.value.trim();
  if (label) {
    params.set("label", label);
  }
  if (zone) {
    params.set("zone", zone);
  }
  return params.toString();
}

function renderRows(events) {
  rows.textContent = "";
  emptyHint.hidden = events.length > 0;
  for (const event of events) {
    const tr = document.createElement("tr");
    appendCell(tr, formatTime(event.first_seen));
    appendCell(tr, event.label || "—");
    appendCell(tr, event.track_id != null ? `#${event.track_id}` : "—");
    appendCell(tr, (event.zones || []).join(", ") || "—");
    appendCell(tr, String(event.dwell_sec ?? 0));
    appendCell(tr, `${Math.round((event.max_confidence ?? 0) * 100)}%`);
    appendCell(tr, String(event.frames ?? 0));
    rows.append(tr);
  }
}

function appendCell(tr, text) {
  const td = document.createElement("td");
  td.textContent = text;
  tr.append(td);
}

async function loadHistory() {
  setChip("查詢中…", "warn");
  showError("");
  try {
    const response = await fetch(`/api/events?${buildQuery()}`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    renderRows(payload.events || []);
    setChip(`${payload.count ?? 0} 筆`, "good");
  } catch {
    setChip("讀取失敗", "bad");
    showError("無法讀取偵測歷史，請確認伺服器正在執行且已啟用事件記錄。");
  }
}

for (const button of rangeButtons) {
  button.addEventListener("click", () => {
    state.rangeSec = Number(button.dataset.range);
    renderRange();
    loadHistory();
  });
}

filters.addEventListener("submit", (event) => {
  event.preventDefault();
  loadHistory();
});

renderRange();
loadHistory();
