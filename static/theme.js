(() => {
  const storageKey = "yolo-elf-theme";
  const systemThemeQuery = window.matchMedia("(prefers-color-scheme: light)");
  let savedTheme = readSavedTheme();

  function readSavedTheme() {
    try {
      const value = window.localStorage.getItem(storageKey);
      return value === "light" || value === "dark" ? value : "";
    } catch {
      return "";
    }
  }

  function saveTheme(theme) {
    savedTheme = theme;
    try {
      window.localStorage.setItem(storageKey, theme);
    } catch {
      // Theme still applies for this page view when localStorage is unavailable.
    }
  }

  function preferredTheme() {
    if (savedTheme) {
      return savedTheme;
    }
    return systemThemeQuery.matches ? "light" : "dark";
  }

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    updateToggleButtons(theme);
  }

  function updateToggleButtons(theme) {
    const nextTheme = theme === "dark" ? "light" : "dark";
    const currentLabel = theme === "dark" ? "深色模式" : "亮色模式";
    const nextLabel = nextTheme === "dark" ? "深色模式" : "亮色模式";

    for (const button of document.querySelectorAll("[data-theme-toggle]")) {
      button.setAttribute("aria-label", `${currentLabel}，切換${nextLabel}`);
      button.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
      button.title = `切換${nextLabel}`;

      const icon = button.querySelector("[data-theme-icon]");
      if (icon) {
        icon.textContent = theme === "dark" ? "☾" : "☀";
      }
    }
  }

  function toggleTheme() {
    const nextTheme = preferredTheme() === "dark" ? "light" : "dark";
    saveTheme(nextTheme);
    applyTheme(nextTheme);
  }

  function bindThemeButtons() {
    for (const button of document.querySelectorAll("[data-theme-toggle]")) {
      button.addEventListener("click", toggleTheme);
    }
    updateToggleButtons(preferredTheme());
  }

  systemThemeQuery.addEventListener("change", () => {
    if (!savedTheme) {
      applyTheme(preferredTheme());
    }
  });

  applyTheme(preferredTheme());

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindThemeButtons, { once: true });
  } else {
    bindThemeButtons();
  }
})();
