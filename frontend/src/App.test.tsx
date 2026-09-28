import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import * as api from "./lib/api";
import type { SetupStatus } from "./lib/api";

const defaultLlm = {
  configured_model: "",
  reachable: false,
  models: null as string[] | null,
  error: null as string | null,
};

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
      return new Response("{}", { status: 404 });
    }),
  );
}

function incompleteStatus(): SetupStatus {
  return {
    spotify_configured: false,
    spotify_signed_in: false,
    llm_ready: false,
    provider: "ollama",
    redirect_uri: "http://127.0.0.1:8765/callback",
    setup_complete: false,
  };
}

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

describe("App setup wizard (item 6)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.history.replaceState({}, "", "/");
  });

  it("auto-opens the wizard when setup is incomplete", async () => {
    mockSessionFetch(false);
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue(incompleteStatus());

    render(<App />);

    await waitFor(() => {
      expect(screen.getByRole("dialog", { name: /first-time setup/i })).toBeInTheDocument();
    });
  });

  it("keeps the wizard closed when setup is complete", async () => {
    mockSessionFetch(true);
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue(completeStatus());

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/Spotify — Connected/)).toBeInTheDocument();
    });
    expect(screen.queryByRole("dialog", { name: /first-time setup/i })).not.toBeInTheDocument();
  });

  it("opens the wizard from settings → Open setup wizard", async () => {
    mockSessionFetch(true);
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue(completeStatus());

    render(<App />);

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /first-time setup/i })).not.toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /Model and Spotify settings/i }));
    fireEvent.click(screen.getByRole("button", { name: "Open setup wizard" }));

    expect(screen.getByRole("dialog", { name: /first-time setup/i })).toBeInTheDocument();
  });
});

describe("App header Spotify status (item 7)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.history.replaceState({}, "", "/");
  });

  it('shows "Client ID saved, sign in to connect" when configured but not signed in', async () => {
    mockSessionFetch(false);
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue({
      ...incompleteStatus(),
      spotify_configured: true,
      spotify_signed_in: false,
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/Client ID saved, sign in to connect/)).toBeInTheDocument();
    });
  });

  it('shows "Not connected" when there is no Client ID', async () => {
    mockSessionFetch(false);
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue(incompleteStatus());

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/Spotify — Not connected/)).toBeInTheDocument();
    });
  });

  it("shows Connected when signed in", async () => {
    mockSessionFetch(true);
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue({
      ...completeStatus(),
      spotify_signed_in: true,
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/Spotify — Connected/)).toBeInTheDocument();
    });
  });
});
