#!/usr/bin/env node
import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "frontend", "package.json"));
const { chromium } = require("playwright");

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const outDir = process.env.SPOT_AI_FY_ARTIFACT_DIR || "/opt/cursor/artifacts";
const electronCli = path.join(repoRoot, "desktop", "node_modules", "electron", "cli.js");

const completeSetup = {
  spotify_configured: true,
  spotify_signed_in: true,
  llm_ready: true,
  provider: "ollama",
  redirect_uri: "http://127.0.0.1:8765/callback",
  setup_complete: true,
};

const defaultLlm = {
  provider: "ollama",
  configured_model: "qwen2.5:3b",
  reachable: true,
  models: ["qwen2.5:3b"],
  error: null,
};

function waitForHttp(url, timeoutMs = 90_000) {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const tick = async () => {
      try {
        const res = await fetch(url);
        if (res.ok) return resolve();
      } catch {
        /* retry */
      }
      if (Date.now() - start > timeoutMs) reject(new Error(`Timeout waiting for ${url}`));
      else setTimeout(tick, 400);
    };
    tick();
  });
}

function mockRoutes(page, { signedIn, mockNp }) {
  return page.route("**/*", async (route) => {
    const url = route.request().url();
    if (url.includes("/api/session")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ signed_in: signedIn, device_id: null }),
      });
    }
    if (url.includes("/api/setup/status")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(completeSetup),
      });
    }
    if (url.includes("/api/llm")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(defaultLlm),
      });
    }
    if (url.includes("/api/devices")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ devices: [] }),
      });
    }
    if (mockNp && url.includes("/api/now-playing")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          is_playing: true,
          track: {
            id: "demo",
            name: "Neon Harbor Lights",
            artists: ["The Glass Foxes"],
            album: "Velvet Lanterns",
            art_url: "https://placehold.co/512x512/e10600/f5f5f5/png?text=Blinding",
            duration_ms: 240000,
          },
          progress_ms: 90000,
          device: { name: "Demo Speaker", type: "Computer" },
          queue: [],
        }),
      });
    }
    return route.continue();
  });
}

function runElectronShot({ file, mockNp, variant, signedIn }) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [electronCli, path.join(repoRoot, "desktop")], {
      cwd: path.join(repoRoot, "desktop"),
      env: {
        ...process.env,
        ELECTRON_DISABLE_SANDBOX: "1",
        SPOT_AI_FY_MOCK_NP: mockNp ? "1" : "0",
        SPOT_AI_FY_SCREENSHOT_DIR: outDir,
        SPOT_AI_FY_SCREENSHOT_FILE: file,
        SPOT_AI_FY_SCREENSHOT_VARIANT: variant,
        SPOT_AI_FY_UI_DEMO_SIGNED_IN: signedIn ? "1" : "0",
      },
      stdio: "inherit",
    });
    child.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`electron exit ${code}`))));
  });
}

async function main() {
  fs.mkdirSync(outDir, { recursive: true });

  const backend = spawn("python3", ["-m", "spot_backend"], {
    cwd: path.join(repoRoot, "backend"),
    env: { ...process.env, API_PORT: "8765" },
    stdio: "pipe",
  });
  const vite = spawn("npm", ["run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"], {
    cwd: path.join(repoRoot, "frontend"),
    stdio: "pipe",
  });

  try {
    await waitForHttp("http://127.0.0.1:8765/api/health");
    await waitForHttp("http://127.0.0.1:5173/");

    const browser = await chromium.launch({ headless: true });

    const webShots = [
      { file: "web-playing-mock.png", signedIn: true, mockNp: true, openSettings: false, viewport: { width: 720, height: 900 } },
      { file: "web-idle-signed-out.png", signedIn: false, mockNp: false, openSettings: false, viewport: { width: 720, height: 900 } },
      { file: "web-settings-sheet-mocknp.png", signedIn: true, mockNp: true, openSettings: true, viewport: { width: 720, height: 900 } },
      { file: "web-settings-sheet-mocknp-420x640.png", signedIn: true, mockNp: true, openSettings: true, viewport: { width: 420, height: 640 } },
    ];

    for (const shot of webShots) {
      const page = await browser.newPage({ viewport: shot.viewport });
      await mockRoutes(page, { signedIn: shot.signedIn, mockNp: shot.mockNp });
      const q = shot.mockNp ? "?mockNp=1" : "";
      await page.goto(`http://127.0.0.1:5173/${q}`, { waitUntil: "networkidle" });
      await page.waitForTimeout(700);
      if (shot.openSettings) {
        await page.getByRole("button", { name: /Model and Spotify settings/i }).click();
        await page.waitForTimeout(400);
      }
      await page.screenshot({ path: path.join(outDir, shot.file), fullPage: true });
      await page.close();
    }
    await browser.close();

    const electronShots = [
      { file: "electron-playing-mock.png", mockNp: true, variant: "playing", signedIn: true },
      { file: "electron-idle-signed-out.png", mockNp: false, variant: "idle", signedIn: false },
      { file: "electron-settings-sheet-mocknp.png", mockNp: true, variant: "settings", signedIn: true },
      {
        file: "electron-queue-expanded-min.png",
        mockNp: true,
        variant: "queue-expanded",
        signedIn: true,
      },
    ];

    for (const shot of electronShots) {
      await runElectronShot(shot);
    }
  } finally {
    backend.kill("SIGTERM");
    vite.kill("SIGTERM");
  }

  for (const name of [
    "web-playing-mock.png",
    "web-idle-signed-out.png",
    "web-settings-sheet-mocknp.png",
    "web-settings-sheet-mocknp-420x640.png",
    "electron-playing-mock.png",
    "electron-idle-signed-out.png",
    "electron-settings-sheet-mocknp.png",
    "electron-queue-expanded-min.png",
  ]) {
    const p = path.join(outDir, name);
    if (!fs.existsSync(p)) throw new Error(`Missing screenshot ${p}`);
  }
  console.log("Screenshots written to", outDir);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
