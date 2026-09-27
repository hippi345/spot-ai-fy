import fs from "node:fs";
import path from "node:path";

import { app } from "electron";

export function repoRootFromDesktop(): string {
  return path.resolve(__dirname, "..", "..");
}

export function packagedResourcesRoot(): string {
  if (app.isPackaged) {
    return process.resourcesPath;
  }
  return repoRootFromDesktop();
}

export function frontendDistDir(): string {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, "frontend");
  }
  return path.join(repoRootFromDesktop(), "frontend", "dist");
}

export function backendDir(): string {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, "backend");
  }
  return path.join(repoRootFromDesktop(), "backend");
}

export function resolvePythonExecutable(): string {
  const envPython = process.env.SPOT_AI_FY_PYTHON?.trim();
  if (envPython && fs.existsSync(envPython)) {
    return envPython;
  }
  const venvCandidates = [
    path.join(backendDir(), ".venv", "bin", "python"),
    path.join(backendDir(), ".venv", "Scripts", "python.exe"),
    path.join(repoRootFromDesktop(), "backend", ".venv", "bin", "python"),
    path.join(repoRootFromDesktop(), "backend", ".venv", "Scripts", "python.exe"),
  ];
  for (const candidate of venvCandidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return process.platform === "win32" ? "python" : "python3";
}
