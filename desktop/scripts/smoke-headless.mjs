#!/usr/bin/env node
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const desktopRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(desktopRoot, "..");
const distMain = path.join(desktopRoot, "dist-main", "main.js");
const frontendDist = path.join(repoRoot, "frontend", "dist");
const electronCli = path.join(desktopRoot, "node_modules", "electron", "cli.js");

if (!fs.existsSync(distMain)) {
  console.error("Run npm run build:main in desktop/ first");
  process.exit(1);
}
if (!fs.existsSync(path.join(frontendDist, "index.html"))) {
  console.error("Build frontend first (cd frontend && ELECTRON_BUILD=1 npm run build)");
  process.exit(1);
}

const env = {
  ...process.env,
  SPOT_AI_FY_SMOKE: "1",
};

const child = spawn(process.execPath, [electronCli, desktopRoot], {
  env,
  stdio: "inherit",
  cwd: desktopRoot,
});

child.on("exit", (code, signal) => {
  if (signal) {
    console.error(`electron killed: ${signal}`);
    process.exit(1);
  }
  if (code !== 0) {
    console.error(`electron exited with code ${code}`);
    process.exit(code ?? 1);
  }
  process.exit(0);
});
