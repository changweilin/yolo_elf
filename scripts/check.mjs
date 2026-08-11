#!/usr/bin/env node
/**
 * The whole check set — lint, tests, syntax — in one cross-platform place.
 *
 * This used to live only in run-tests.ps1, which meant CI could prove the suite
 * on Windows and nowhere else, and the list of JavaScript files to syntax-check
 * was hand-maintained (it had silently fallen three files behind). Both are
 * fixed here: file lists are globbed, and Linux runs exactly the same checks.
 *
 * Usage: node scripts/check.mjs [--lint-only | --no-lint]
 */

import { spawnSync } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const IS_WINDOWS = process.platform === "win32";
const MODE = process.argv[2] ?? "";

/**
 * Prefer the project venv. A git worktree under .claude/worktrees/ has none of
 * its own, and the global Python it would otherwise fall back to lags whole
 * minor versions of Ultralytics behind — a green run there proves less than it
 * looks like it does. Set YOLO_ELF_PYTHON to override.
 */
function resolvePython() {
  if (process.env.YOLO_ELF_PYTHON) return process.env.YOLO_ELF_PYTHON;

  const relativeExe = IS_WINDOWS ? ["Scripts", "python.exe"] : ["bin", "python"];
  const roots = [ROOT, resolve(ROOT, "..", "..", "..")]; // repo, then worktree parent
  for (const root of roots) {
    const candidate = join(root, ".venv", ...relativeExe);
    if (existsSync(candidate)) return candidate;
  }
  return IS_WINDOWS ? "python" : "python3";
}

function listFiles(dir, extensions) {
  const base = join(ROOT, dir);
  if (!existsSync(base)) return [];
  return readdirSync(base)
    .filter((name) => extensions.some((ext) => name.endsWith(ext)))
    .map((name) => relative(ROOT, join(base, name)))
    .sort();
}

const failures = [];

function run(label, command, args) {
  process.stdout.write(`\n→ ${label}\n`);
  const result = spawnSync(command, args, { cwd: ROOT, stdio: "inherit" });
  if (result.error) {
    process.stdout.write(`  cannot start: ${result.error.message}\n`);
    failures.push(label);
  } else if (result.status !== 0) {
    failures.push(label);
  }
}

const python = resolvePython();

// Ultralytics writes a settings file on import; keep it in the repo so a test
// run never touches the user's home directory.
process.env.YOLO_CONFIG_DIR ??= join(ROOT, ".ultralytics");

if (MODE !== "--no-lint") {
  // Advisory when ruff is missing — a contributor running tests locally should
  // not be blocked by an extra dependency. CI installs it, so CI enforces it.
  if (spawnSync(python, ["-m", "ruff", "--version"], { cwd: ROOT }).status === 0) {
    run("ruff check", python, ["-m", "ruff", "check", "app", "tests", "scripts"]);
  } else {
    process.stdout.write("\n→ ruff check (skipped: ruff is not installed)\n");
  }
}

if (MODE !== "--lint-only") {
  run("pytest", python, ["-m", "pytest", "-q"]);

  // bench_detector.py is not imported by the suite; compiling it is the only
  // thing between a typo there and a broken benchmark run.
  run("py_compile bench_detector.py", python, [
    "-m",
    "py_compile",
    join("scripts", "bench_detector.py"),
  ]);

  const javascript = [...listFiles("static", [".js"]), ...listFiles("scripts", [".mjs"])];
  process.stdout.write(`\n→ node --check (${javascript.length} files)\n`);
  for (const file of javascript) {
    const result = spawnSync(process.execPath, ["--check", file], { cwd: ROOT, stdio: "inherit" });
    if (result.status !== 0) failures.push(`node --check ${file}`);
  }
}

if (failures.length === 0) {
  process.stdout.write("\nAll checks passed.\n");
  process.exit(0);
}
process.stdout.write(`\nFailed: ${failures.join(", ")}\n`);
process.exit(1);
