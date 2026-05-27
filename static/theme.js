(() => {
  const storageKey = "yolo-elf-theme";
  const systemThemeQuery = window.matchMedia
    ? window.matchMedia("(prefers-color-scheme: light)")
    : { matches: false, addEventListener: () => {} };
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
    const currentLabel = theme === "dark" ? "dark theme" : "light theme";
    const nextLabel = nextTheme === "dark" ? "dark theme" : "light theme";

    for (const button of document.querySelectorAll("[data-theme-toggle]")) {
      button.setAttribute(
        "aria-label",
        `Current ${currentLabel}. Switch to ${nextLabel}.`,
      );
      button.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
      button.title = `Switch to ${nextLabel}`;

      const sunIcon = button.querySelector("[data-theme-icon-sun]");
      const moonIcon = button.querySelector("[data-theme-icon-moon]");
      if (sunIcon && moonIcon) {
        const showSun = theme === "dark";
        sunIcon.hidden = !showSun;
        moonIcon.hidden = showSun;
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
