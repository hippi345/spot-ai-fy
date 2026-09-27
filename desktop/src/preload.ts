import { contextBridge, ipcRenderer } from "electron";

function readApiBase(): string {
  const flag = process.argv.find((arg) => arg.startsWith("--spot-api-base="));
  return flag ? flag.slice("--spot-api-base=".length) : "";
}

const apiBaseUrl = readApiBase();

contextBridge.exposeInMainWorld("spotAiFy", {
  apiBaseUrl,
  isDesktop: true,
  platform: process.platform,
  windowMinimize: () => ipcRenderer.invoke("window:minimize"),
  windowToggleMaximize: () => ipcRenderer.invoke("window:toggleMaximize"),
  windowClose: () => ipcRenderer.invoke("window:close"),
  openExternal: (url: string) => ipcRenderer.invoke("shell:openExternal", url),
});

document.addEventListener(
  "click",
  (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const anchor = target.closest("a");
    if (!anchor) return;
    const href = anchor.getAttribute("href");
    if (!href) return;
    if (href === "/login" || href.endsWith("/login") || href.includes("/login?")) {
      event.preventDefault();
      const url = apiBaseUrl ? `${apiBaseUrl.replace(/\/$/, "")}/login` : "/login";
      void ipcRenderer.invoke("shell:openExternal", url);
    }
  },
  true,
);
