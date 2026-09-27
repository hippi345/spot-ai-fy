import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { NowPlayingBar } from "./NowPlayingBar";
import * as np from "../lib/nowPlaying";

describe("NowPlayingBar", () => {
  beforeEach(() => {
    vi.spyOn(np, "fetchNowPlaying");
    vi.spyOn(np, "postPlayerNext");
    vi.spyOn(np, "postPlayerPrevious");
    vi.spyOn(np, "postPlayerToggle");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("hidden when signed out", () => {
    const { container } = render(<NowPlayingBar signedIn={false} />);
    expect(container.firstChild).toBeNull();
  });

  it("shows playing state", async () => {
    vi.mocked(np.fetchNowPlaying).mockResolvedValue({
      ...np.MOCK_NOW_PLAYING_PLAYING,
      fetched_at: Date.now() / 1000,
    });
    render(<NowPlayingBar signedIn={true} />);
    await waitFor(() => {
      expect(screen.getByText("Neon Harbor Lights")).toBeInTheDocument();
    });
  });

  it("shows idle bar", async () => {
    vi.mocked(np.fetchNowPlaying).mockResolvedValue({
      is_playing: false,
      track: null,
      queue: [],
      fetched_at: Date.now() / 1000,
    });
    render(<NowPlayingBar signedIn={true} />);
    await waitFor(() => {
      expect(screen.getByText("Nothing playing")).toBeInTheDocument();
    });
  });

  it("expand/collapse queue on click and keyboard", async () => {
    vi.mocked(np.fetchNowPlaying).mockResolvedValue({
      ...np.MOCK_NOW_PLAYING_PLAYING,
      fetched_at: Date.now() / 1000,
    });
    render(<NowPlayingBar signedIn={true} />);
    await waitFor(() => screen.getByText("Neon Harbor Lights"));
    const bar = screen.getByRole("button", { name: /now playing/i });
    expect(bar).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(bar);
    expect(bar).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("Midnight Relay")).toBeInTheDocument();
    fireEvent.keyDown(bar, { key: " ", code: "Space" });
    expect(bar).toHaveAttribute("aria-expanded", "false");
  });

  it("control buttons call APIs", async () => {
    vi.mocked(np.fetchNowPlaying).mockResolvedValue({
      ...np.MOCK_NOW_PLAYING_PLAYING,
      fetched_at: Date.now() / 1000,
    });
    vi.mocked(np.postPlayerNext).mockResolvedValue({
      ...np.MOCK_NOW_PLAYING_PLAYING,
      fetched_at: Date.now() / 1000,
    });
    render(<NowPlayingBar signedIn={true} />);
    await waitFor(() => screen.getByLabelText("Next track"));
    fireEvent.click(screen.getByLabelText("Next track"));
    await waitFor(() => expect(np.postPlayerNext).toHaveBeenCalled());
  });

  it("keeps stale payload without error", async () => {
    vi.mocked(np.fetchNowPlaying)
      .mockResolvedValueOnce({
        ...np.MOCK_NOW_PLAYING_PLAYING,
        stale: true,
        fetched_at: Date.now() / 1000,
      })
      .mockRejectedValueOnce(new Error("network"));
    render(<NowPlayingBar signedIn={true} />);
    await waitFor(() => screen.getByText("Neon Harbor Lights"));
    await waitFor(() => expect(np.fetchNowPlaying).toHaveBeenCalledTimes(1));
  });

  it("poll interval follows visibility", async () => {
    vi.useFakeTimers();
    vi.mocked(np.fetchNowPlaying).mockResolvedValue({
      ...np.MOCK_NOW_PLAYING_PLAYING,
      fetched_at: Date.now() / 1000,
    });
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      get: () => "visible",
    });
    render(<NowPlayingBar signedIn={true} />);
    await vi.waitFor(() => expect(np.fetchNowPlaying).toHaveBeenCalled());
    const callsBefore = vi.mocked(np.fetchNowPlaying).mock.calls.length;
    await vi.advanceTimersByTimeAsync(3000);
    expect(vi.mocked(np.fetchNowPlaying).mock.calls.length).toBeGreaterThan(callsBefore);
    vi.useRealTimers();
  });
});
