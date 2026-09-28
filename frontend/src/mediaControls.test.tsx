import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ModelSpotifySettings } from "./components/ModelSpotifySettings";
import { NowPlayingBar } from "./components/NowPlayingBar";

describe("Media control SVG icons", () => {
  it("renders svg play control instead of emoji", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(
          JSON.stringify({
            is_playing: false,
            track: {
              id: "t1",
              name: "Demo",
              artists: ["A"],
              album: "Al",
              art_url: null,
              duration_ms: 1000,
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      ),
    );
    render(<NowPlayingBar signedIn />);
    const playBtn = await screen.findByRole("button", { name: "Play" });
    expect(playBtn.querySelector("svg")).toBeTruthy();
    expect(playBtn.textContent).not.toMatch(/[⏮⏸⏭▶]/);
  });
});

describe("Settings reconnect button", () => {
  const baseProps = {
    llm: null,
    llmPick: "ollama" as const,
    llmSaving: false,
    ollamaModelSelect: "__env__",
    ollamaCustomModel: "",
    geminiModelSelect: "__env__",
    geminiCustomModel: "",
    ollamaModelApplyDisabled: true,
    geminiModelApplyDisabled: true,
    deviceId: "",
    deviceOptions: [],
    loadingDevices: false,
    onLlmPickChange: () => undefined,
    onApplyLlmProvider: () => undefined,
    onResetLlmProvider: () => undefined,
    onRefreshLlm: () => undefined,
    onOllamaModelSelectChange: () => undefined,
    onOllamaCustomModelChange: () => undefined,
    onApplyOllamaModel: () => undefined,
    onGeminiModelSelectChange: () => undefined,
    onGeminiCustomModelChange: () => undefined,
    onApplyGeminiModel: () => undefined,
    onOpenSetupWizard: () => undefined,
    onLogout: () => undefined,
    onDeviceIdChange: () => undefined,
    onRefreshDevices: () => undefined,
    onSaveDevice: () => undefined,
  };

  it("shows Reconnect as a secondary link when signed in", () => {
    render(
      <ModelSpotifySettings
        {...baseProps}
        session={{ signed_in: true, spotify_playlist_write_ok: true }}
      />,
    );
    const link = screen.getByRole("link", { name: "Reconnect" });
    expect(link.className).toMatch(/secondary/);
  });

  it("shows Connect Spotify when signed out", () => {
    render(
      <ModelSpotifySettings {...baseProps} session={{ signed_in: false }} />,
    );
    expect(screen.getByRole("link", { name: "Connect Spotify" })).toBeInTheDocument();
  });
});
