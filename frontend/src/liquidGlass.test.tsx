import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { LiquidBackground } from "./components/LiquidBackground";
import { SettingsSheet } from "./components/SettingsSheet";
import * as api from "./lib/api";
import type { SetupStatus } from "./lib/api";

const defaultLlm = {
  configured_model: "",
  reachable: false,
  models: null as string[] | null,
  error: null as string | null,
};

function completeStatus(): SetupStatus {
  return {
    spotify_configured: true,
    spotify_signed_in: true,
    llm_ready: true,
    provider: "ollama",
    redirect_uri: "http://127.0.0.1:8765/callback",
    setup_complete: true,
  };
}

function mockSessionFetch(signedIn: boolean) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/api/session")) {
        return new Response(JSON.stringify({ signed_in: signedIn, device_id: null }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.includes("/api/llm")) {
        return new Response(JSON.stringify(defaultLlm), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.includes("/api/now-playing")) {
        return new Response(JSON.stringify({ track: null, is_playing: false }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response("{}", { status: 404 });
    }),
  );
}

describe("LiquidBackground idle fallback", () => {
  it("uses idle gradient mode when idle prop is true", () => {
    const { rerender } = render(<LiquidBackground idle={true} artUrl="https://example.com/art.jpg" />);
    expect(screen.getByTestId("liquid-background")).toHaveAttribute("data-mode", "idle");

    rerender(<LiquidBackground idle={false} artUrl="https://example.com/art.jpg" />);
    expect(screen.getByTestId("liquid-background")).toHaveAttribute("data-mode", "art");
  });
});

describe("SettingsSheet", () => {
  it("closes on Escape, backdrop click, and close button", () => {
    const onClose = vi.fn();
    render(
      <SettingsSheet open title="Test settings" onClose={onClose}>
        <p>Body</p>
      </SettingsSheet>,
    );

    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);

    onClose.mockClear();
    fireEvent.click(screen.getByTestId("settings-sheet-backdrop"));
    expect(onClose).toHaveBeenCalledTimes(1);

    onClose.mockClear();
    fireEvent.click(screen.getByTestId("settings-sheet-close"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe("App liquid glass integration", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.history.replaceState({}, "", "/");
  });

  it("shows idle background when signed out", async () => {
    mockSessionFetch(false);
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue(completeStatus());

    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId("liquid-background")).toHaveAttribute("data-mode", "idle");
    });
  });

  it("opens and closes the settings sheet from the header gear", async () => {
    mockSessionFetch(true);
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue(completeStatus());

    render(<App />);

    await waitFor(() => {
      expect(screen.queryByTestId("settings-sheet-panel")).not.toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /Model and Spotify settings/i }));

    expect(screen.getByTestId("settings-sheet-panel")).toBeInTheDocument();
    expect(screen.getByRole("dialog")).toHaveAccessibleName(/Model & Spotify/i);

    fireEvent.click(screen.getByTestId("settings-sheet-close"));
    await waitFor(() => {
      expect(screen.queryByTestId("settings-sheet-panel")).not.toBeInTheDocument();
    });
  });
});

describe("liquid glass art backdrop tokens", () => {
  it("keeps subtle album-art read-through (blur, scale, panel alpha)", () => {
    const cssPath = path.join(path.dirname(fileURLToPath(import.meta.url)), "styles.css");
    const css = readFileSync(cssPath, "utf8");
    expect(css).toContain("--liquid-art-blur: 36px");
    expect(css).toContain("--liquid-art-scale: 1.05");
    expect(css).toContain("--glass-panel-alpha: 0.69");
    expect(css).toMatch(/\.liquid-bg-layer[\s\S]*?inset:\s*0/);
    expect(css).toMatch(/\.liquid-bg--paused[\s\S]*?--liquid-art-blur-paused/);
  });
});
