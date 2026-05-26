import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "..");
const args = process.argv.slice(2);
const hostFlagIndex = args.indexOf("--host");
const host =
  hostFlagIndex >= 0 && args[hostFlagIndex + 1]
    ? args[hostFlagIndex + 1]
    : process.env.HOST || "127.0.0.1";
const positionalArgs = args.filter(
  (_, index) => index !== hostFlagIndex && index !== hostFlagIndex + 1,
);
const port = positionalArgs[0] || process.env.PORT || "8000";
const python = path.join(root, ".venv", "Scripts", "python.exe");

if (!fs.existsSync(python)) {
  console.error(`Missing virtualenv Python at ${python}. Run scripts/setup.ps1 first.`);
  process.exit(1);
}

const ultralyticsDir = path.join(root, ".ultralytics");
fs.mkdirSync(ultralyticsDir, { recursive: true });

function normalizedEnvironment(extra) {
  const env = {};
  for (const [key, value] of Object.entries(process.env)) {
    const normalizedKey = key.toLowerCase() === "path" ? "Path" : key;
    if (env[normalizedKey] === undefined) {
      env[normalizedKey] = value;
    }
  }
  return { ...env, ...extra };
}

function localUrls() {
  if (host !== "0.0.0.0" && host !== "::") {
    return [`http://${host}:${port}`];
  }

  const addresses = ["127.0.0.1"];
  for (const interfaces of Object.values(os.networkInterfaces())) {
    for (const network of interfaces || []) {
      if (network.family === "IPv4" && !network.internal) {
        addresses.push(network.address);
      }
    }
  }
  return [...new Set(addresses)].map((address) => `http://${address}:${port}`);
}

const out = fs.openSync(path.join(root, `.server-${port}.out.log`), "a");
const err = fs.openSync(path.join(root, `.server-${port}.err.log`), "a");
const child = spawn(
  python,
  ["-m", "uvicorn", "app.main:app", "--host", host, "--port", String(port)],
  {
    cwd: root,
    detached: true,
    env: normalizedEnvironment({
      HOST: host,
      PORT: String(port),
      YOLO_CONFIG_DIR: ultralyticsDir,
    }),
    stdio: ["ignore", out, err],
    windowsHide: true,
  },
);

child.unref();
console.log(`Started YOLO Elf listening on ${host}:${port} (pid ${child.pid})`);
for (const url of localUrls()) {
  console.log(`Open: ${url}`);
}
