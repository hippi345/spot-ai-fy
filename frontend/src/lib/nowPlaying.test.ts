import { afterEach, describe, expect, it, vi } from "vitest";

import {
  interpolateProgress,
  MOCK_NOW_PLAYING_PLAYING,
  pollIntervalMs,
} from "./nowPlaying";

describe("nowPlaying helpers", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("pollIntervalMs uses 3s visible and 15s hidden", () => {
    expect(pollIntervalMs(true)).toBe(3000);
    expect(pollIntervalMs(false)).toBe(15000);
  });

  it("interpolateProgress advances while playing", () => {
    const base = {
      ...MOCK_NOW_PLAYING_PLAYING,
      progress_ms: 10_000,
      fetched_at: 1000,
      is_playing: true,
    };
    const at = 1000 * 1000 + 2000;
    const v = interpolateProgress(base, at);
    expect(v).toBe(12_000);
  });

  it("interpolateProgress clamps to duration", () => {
    const base = {
      ...MOCK_NOW_PLAYING_PLAYING,
      progress_ms: 230_000,
      duration_ms: 240_000,
      fetched_at: 1,
      is_playing: true,
      track: { ...MOCK_NOW_PLAYING_PLAYING.track!, duration_ms: 240_000 },
    };
    const v = interpolateProgress(base, 1 * 1000 + 20_000);
    expect(v).toBe(240_000);
  });
});
