import { isDesktopShell } from "../lib/api";
import { IconClose, IconMaximize, IconMinimize } from "./icons/AppIcons";

export function DesktopTitleBar() {
  if (!isDesktopShell()) {
    return null;
  }
  const api = window.spotAiFy;
  const isWin = api?.platform === "win32";
  return (
    <div className="desktop-titlebar" data-platform={api?.platform ?? "unknown"}>
      <div className="desktop-titlebar-drag">
        <span className="desktop-titlebar-title">Spot-AI-fy</span>
      </div>
      {isWin ? (
        <div className="desktop-titlebar-controls">
          <button
            type="button"
            className="desktop-win-btn"
            aria-label="Minimize"
            onClick={() => api?.windowMinimize?.()}
          >
            <IconMinimize />
          </button>
          <button
            type="button"
            className="desktop-win-btn"
            aria-label="Maximize"
            onClick={() => api?.windowToggleMaximize?.()}
          >
            <IconMaximize />
          </button>
          <button
            type="button"
            className="desktop-win-btn desktop-win-btn--close"
            aria-label="Close"
            onClick={() => api?.windowClose?.()}
          >
            <IconClose />
          </button>
        </div>
      ) : null}
    </div>
  );
}
