// Drives the "切換模型中…" progress bar shared by the recorder, viewer and
// settings pages. Switching fast/accurate (or saving a new model name) makes
// the server preload the new weights in the background; this controller polls
// /api/status until that preset reports `loaded`, animating a progress bar and
// locking the page's parameters so nothing changes mid-load.

const POLL_INTERVAL_MS = 500;
const MAX_WAIT_MS = 30000;
// Simulated climb ceiling: we ease toward this while waiting, then jump to 100%
// the moment the new model confirms loaded.
const CLIMB_CEILING = 92;
const CLIMB_TAU_MS = 2600;

export function createModelSwitch({ progressEl, fillEl, lock } = {}) {
  let active = false;
  let target = null;
  let startedAt = 0;
  let pollTimer = null;
  let rafId = null;
  let finishTimer = null;

  function setFill(pct) {
    if (fillEl) {
      fillEl.style.width = `${Math.max(0, Math.min(100, pct))}%`;
    }
  }

  function setActive(on) {
    active = on;
    progressEl?.classList.toggle("active", on);
    if (typeof lock === "function") {
      lock(on);
    }
  }

  function climb() {
    if (!active) {
      return;
    }
    const elapsed = performance.now() - startedAt;
    const pct = CLIMB_CEILING * (1 - Math.exp(-elapsed / CLIMB_TAU_MS));
    setFill(Math.max(8, pct));
    rafId = requestAnimationFrame(climb);
  }

  function finish(ok) {
    if (!active) {
      return;
    }
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    if (!ok) {
      progressEl?.classList.add("bad");
    }
    setFill(100);
    finishTimer = window.setTimeout(
      () => {
        setActive(false);
        progressEl?.classList.remove("bad");
        setFill(0);
      },
      ok ? 350 : 1200,
    );
  }

  async function poll() {
    if (!active) {
      return;
    }
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (response.ok) {
        const status = await response.json();
        const detector = status.detector || {};
        if (detector.mode === target && detector.last_load_error) {
          finish(false);
          return;
        }
        if (detector.mode === target && detector.loaded) {
          finish(true);
          return;
        }
      }
    } catch {
      // Transient: keep waiting until the deadline.
    }
    if (performance.now() - startedAt > MAX_WAIT_MS) {
      finish(true);
      return;
    }
    pollTimer = window.setTimeout(poll, POLL_INTERVAL_MS);
  }

  function begin(mode) {
    target = mode;
    startedAt = performance.now();
    if (finishTimer) {
      clearTimeout(finishTimer);
      finishTimer = null;
    }
    progressEl?.classList.remove("bad");
    setFill(8);
    if (!active) {
      setActive(true);
    }
    if (rafId) {
      cancelAnimationFrame(rafId);
    }
    rafId = requestAnimationFrame(climb);
    if (pollTimer) {
      clearTimeout(pollTimer);
    }
    pollTimer = window.setTimeout(poll, POLL_INTERVAL_MS);
  }

  return { begin, get active() {
    return active;
  } };
}
