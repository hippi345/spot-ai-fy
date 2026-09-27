import path from "node:path";

import { app, BrowserWindow, ipcMain, session, shell } from "electron";

import { pickFreePort, startBackend, type BackendHandle } from "./backendProcess";
import { frontendDistDir } from "./paths";
import { startUiServer } from "./staticServer";
import {
  ensureOnScreen,
  loadWindowState,
  saveWindowState,
  windowMinSize,
} from "./windowState";

let mainWindow: BrowserWindow | null = null;
let backend: BackendHandle | null = null;
let uiServerClose: (() => Promise<void>) | null = null;

const CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' https: data:",
  "connect-src 'self' http://127.0.0.1:* ws://127.0.0.1:*",
  "font-src 'self' data:",
  "object-src 'none'",
  "base-uri 'self'",
  "frame-ancestors 'none'",
].join("; ");

function applySecurity(sessionInstance: Electron.Session): void {
  sessionInstance.webRequest.onHeadersReceived((details, callback) => {
    const headers = { ...details.responseHeaders };
    headers["Content-Security-Policy"] = [CSP];
    callback({ responseHeaders: headers });
  });
}

function persistBounds(win: BrowserWindow): void {
  if (win.isDestroyed()) return;
  const bounds = win.getBounds();
  saveWindowState({
    width: bounds.width,
    height: bounds.height,
    x: bounds.x,
    y: bounds.y,
    isMaximized: win.isMaximized(),
  });
}

async function createMainWindow(uiOrigin: string, apiOrigin: string): Promise<void> {
  const saved = ensureOnScreen(loadWindowState());
  const preloadPath = path.join(__dirname, "preload.js");

  mainWindow = new BrowserWindow({
    width: saved.width,
    height: saved.height,
    x: saved.x,
    y: saved.y,
    minWidth: windowMinSize.width,
    minHeight: windowMinSize.height,
    frame: false,
    show: false,
    backgroundColor: "#0c0c0f",
    webPreferences: {
      preload: preloadPath,
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      additionalArguments: [`--spot-api-base=${apiOrigin}`],
    },
  });

  if (saved.isMaximized) {
    mainWindow.maximize();
  }

  mainWindow.on("resize", () => persistBounds(mainWindow!));
  mainWindow.on("move", () => persistBounds(mainWindow!));
  mainWindow.on("close", () => persistBounds(mainWindow!));

  const loadUrl = process.env.SPOT_AI_FY_MOCK_NP === "1" ? `${uiOrigin}?mockNp=1` : uiOrigin;
  await mainWindow.loadURL(loadUrl);
  mainWindow.once("ready-to-show", () => {
    mainWindow?.show();
    if (process.env.SPOT_AI_FY_SMOKE === "1") {
      setTimeout(() => app.quit(), 500);
    }
  });
}

async function bootstrap(): Promise<void> {
  applySecurity(session.defaultSession);
  const dist = frontendDistDir();
  const backendPort = await pickFreePort(8765);
  const uiPort = await pickFreePort(9240);
  const uiOrigin = `http://127.0.0.1:${uiPort}`;
  backend = await startBackend(uiOrigin, backendPort);
  const apiOrigin = `http://127.0.0.1:${backend.port}`;
  const ui = await startUiServer(dist, apiOrigin, uiPort);
  uiServerClose = ui.close;
  await createMainWindow(uiOrigin, apiOrigin);
}

function registerIpc(): void {
  ipcMain.handle("window:minimize", () => {
    mainWindow?.minimize();
  });
  ipcMain.handle("window:toggleMaximize", () => {
    if (!mainWindow) return;
    if (mainWindow.isMaximized()) mainWindow.unmaximize();
    else mainWindow.maximize();
  });
  ipcMain.handle("window:close", () => {
    mainWindow?.close();
  });
  ipcMain.handle("shell:openExternal", (_event, url: string) => {
    if (typeof url === "string" && url.startsWith("http")) {
      return shell.openExternal(url);
    }
    return Promise.resolve();
  });
}

app.whenReady().then(() => {
  registerIpc();
  return bootstrap();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  void (async () => {
    if (uiServerClose) await uiServerClose();
    if (backend) await backend.stop();
  })();
});
