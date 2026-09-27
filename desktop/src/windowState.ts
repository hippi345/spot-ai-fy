import fs from "node:fs";
import path from "node:path";

import { app, screen } from "electron";

export type PersistedWindowState = {
  width: number;
  height: number;
  x?: number;
  y?: number;
  isMaximized?: boolean;
};

const MIN_WIDTH = 420;
const MIN_HEIGHT = 640;
const DEFAULT_WIDTH = 960;
const DEFAULT_HEIGHT = 800;

function stateFile(): string {
  return path.join(app.getPath("userData"), "window-state.json");
}

export function loadWindowState(): PersistedWindowState {
  try {
    const raw = fs.readFileSync(stateFile(), "utf8");
    const data = JSON.parse(raw) as PersistedWindowState;
    if (typeof data.width === "number" && typeof data.height === "number") {
      return {
        width: Math.max(MIN_WIDTH, data.width),
        height: Math.max(MIN_HEIGHT, data.height),
        x: data.x,
        y: data.y,
        isMaximized: Boolean(data.isMaximized),
      };
    }
  } catch {
    /* first run */
  }
  return { width: DEFAULT_WIDTH, height: DEFAULT_HEIGHT };
}

export function saveWindowState(state: PersistedWindowState): void {
  try {
    fs.mkdirSync(path.dirname(stateFile()), { recursive: true });
    fs.writeFileSync(stateFile(), JSON.stringify(state, null, 2), "utf8");
  } catch {
    /* non-fatal */
  }
}

export function ensureOnScreen(state: PersistedWindowState): PersistedWindowState {
  const displays = screen.getAllDisplays();
  if (!displays.length || state.x === undefined || state.y === undefined) {
    return state;
  }
  const inBounds = displays.some((d) => {
    const b = d.workArea;
    return (
      state.x! >= b.x &&
      state.y! >= b.y &&
      state.x! + state.width <= b.x + b.width &&
      state.y! + state.height <= b.y + b.height
    );
  });
  if (!inBounds) {
    const { x, y, width, height } = screen.getPrimaryDisplay().workArea;
    return {
      ...state,
      x: Math.round(x + (width - state.width) / 2),
      y: Math.round(y + (height - state.height) / 2),
    };
  }
  return state;
}

export const windowMinSize = { width: MIN_WIDTH, height: MIN_HEIGHT };
