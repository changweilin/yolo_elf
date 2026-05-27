import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "..");
const staticDir = path.join(root, "static");
const distDir = path.join(root, "dist");

function assertInsideRoot(target) {
  const resolved = path.resolve(target);
  if (resolved !== root && !resolved.startsWith(`${root}${path.sep}`)) {
    throw new Error(`Refusing to write outside project root: ${resolved}`);
  }
}

function cleanDist() {
  assertInsideRoot(distDir);
  fs.rmSync(distDir, { recursive: true, force: true });
  fs.mkdirSync(distDir, { recursive: true });
}

function copyStaticAssets() {
  fs.cpSync(staticDir, path.join(distDir, "static"), {
    recursive: true,
    filter: (source) => !source.endsWith(".html"),
  });
}

function rewriteHtml(html, { assetPrefix, phoneHref }) {
  return html
    .replaceAll('href="/static/', `href="${assetPrefix}static/`)
    .replaceAll('src="/static/', `src="${assetPrefix}static/`)
    .replaceAll('href="/phone"', `href="${phoneHref}"`)
    .replace(
      new RegExp(`src="${assetPrefix}static/phone\\.js([^"]*)"`),
      (_match, query) =>
        `src="${assetPrefix}static/phone.js${query || ""}${query ? "&" : "?"}demo=1"`,
    )
    .replace(
      new RegExp(`src="${assetPrefix}static/viewer\\.js([^"]*)"`),
      (_match, query) =>
        `src="${assetPrefix}static/viewer.js${query || ""}${query ? "&" : "?"}demo=1"`,
    );
}

function writePage(sourceName, targetRelativePath, options) {
  const sourcePath = path.join(staticDir, sourceName);
  const targetPath = path.join(distDir, targetRelativePath);
  assertInsideRoot(targetPath);
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  const html = fs.readFileSync(sourcePath, "utf8");
  fs.writeFileSync(targetPath, rewriteHtml(html, options));
}

cleanDist();
copyStaticAssets();

writePage("viewer.html", "index.html", {
  assetPrefix: "./",
  phoneHref: "./phone/",
});
writePage("viewer.html", path.join("viewer", "index.html"), {
  assetPrefix: "../",
  phoneHref: "../phone/",
});
writePage("phone.html", path.join("phone", "index.html"), {
  assetPrefix: "../",
  phoneHref: "../phone/",
});

fs.writeFileSync(
  path.join(distDir, ".nojekyll"),
  "Static GitHub Pages build for YOLO Elf.\n",
);

console.log(`Built static demo at ${path.relative(root, distDir)}`);
