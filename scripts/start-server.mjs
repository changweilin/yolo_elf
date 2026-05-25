import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "..");
const host = process.env.HOST || "127.0.0.1";
const port = process.argv[2] || process.env.PORT || "8000";
const python = path.join(root, ".venv", "Scripts", "python.exe");

if (!fs.existsSync(python)) {
  console.error(`Missing virtualenv Python at ${python}. Run scripts/setup.ps1 first.`);
  process.exit(1);
}

const ultralyticsDir = path.join(root, ".ultralytics");
fs.mkdirSync(ultralyticsDir, { recursive: true });

const out = fs.openSync(path.join(root, `.server-${port}.out.log`), "a");
const err = fs.openSync(path.join(root, `.server-${port}.err.log`), "a");
const child = spawn(
  python,
  ["-m", "uvicorn", "app.main:app", "--host", host, "--port", String(port)],
  {
    cwd: root,
    detached: true,
    env: {
      ...process.env,
      HOST: host,
      PORT: String(port),
      YOLO_CONFIG_DIR: ultralyticsDir,
    },
    stdio: ["ignore", out, err],
    windowsHide: true,
  },
);

child.unref();
console.log(`Started YOLO Elf on http://${host}:${port} (pid ${child.pid})`);
