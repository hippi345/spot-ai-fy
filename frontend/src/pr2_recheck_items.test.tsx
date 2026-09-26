import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { SetupWizard } from "./SetupWizard";
import * as api from "./lib/api";
import { FRIENDLY_SPOTIFY_GUIDANCE, isUnpersistedAssistantFallback } from "./lib/chatMessages";

describe("item10_assistant_message_keeps_trace", () => {
  it("renders Actions taken on assistant bubbles when trace is present", () => {
    render(
      <div className="bubble assistant">
        hello
        <details className="message-trace">
          <summary>Actions taken (1)</summary>
          <ul className="trace-steps compact">
            <li>spotify_me</li>
          </ul>
        </details>
      </div>,
    );
    expect(screen.getByText(/Actions taken \(1\)/)).toBeInTheDocument();
  });
});

describe("item12_frontend_stream_idle_constant", () => {
  it("uses a 45s stream idle threshold in App source", () => {
    const text = readFileSync(resolve(__dirname, "./App.tsx"), "utf8");
    expect(text).toMatch(/streamIdleMs\s*=\s*45_000/);
  });
});

describe("item16_vite_listens_on_127_0_0_1", () => {
  it("sets server.host to 127.0.0.1", () => {
    const text = readFileSync(resolve(__dirname, "../vite.config.ts"), "utf8");
    expect(text).toMatch(/host:\s*["']127\.0\.0\.1["']/);
  });
});

describe("item18_friendly_fallback_detection", () => {
  it("treats stock empty replies as unpersisted", () => {
    expect(isUnpersistedAssistantFallback(FRIENDLY_SPOTIFY_GUIDANCE)).toBe(true);
    expect(isUnpersistedAssistantFallback("The model returned no assistant text")).toBe(true);
  });
});

describe("item08_setup_wizard_manual_open_stays_open", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("does not auto-close when setup_complete unless closeOnComplete", async () => {
    const onComplete = vi.fn();
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue({
      spotify_configured: true,
      spotify_signed_in: true,
      llm_ready: true,
      provider: "ollama",
      redirect_uri: "http://127.0.0.1:8765/callback",
      setup_complete: true,
    });

    render(<SetupWizard allowDismiss closeOnComplete={false} onComplete={onComplete} />);
    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });
    expect(onComplete).not.toHaveBeenCalled();
  });

  it("shows Saved after LLM save", async () => {
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue({
      spotify_configured: true,
      spotify_signed_in: true,
      llm_ready: false,
      provider: "ollama",
      redirect_uri: "http://127.0.0.1:8765/callback",
      setup_complete: false,
      ollama_model: "qwen3:4b-instruct",
    });
    vi.spyOn(api, "saveLlmSetup").mockResolvedValue({ reachable: true, models: ["qwen3:4b-instruct"] });
    render(<SetupWizard allowDismiss={false} closeOnComplete={false} onComplete={() => undefined} />);
    await waitFor(() => expect(screen.getByRole("heading", { name: /Step 2/i })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /Save & test/i }));
    await waitFor(() => expect(screen.getByText(/Saved — settings tested successfully/i)).toBeInTheDocument());
  });
});

describe("item17_header_shows_real_provider_while_wizard_open", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.history.replaceState({}, "", "/");
  });

  it("shows Gemini not connected when wizard open and llm not ready", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/api/session")) {
          return new Response(JSON.stringify({ signed_in: true, device_id: null }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          });
        }
        if (url.includes("/api/llm")) {
          return new Response(
            JSON.stringify({ provider: "gemini", reachable: true, configured_model: "gemini-2.5-flash" }),
            { status: 200, headers: { "Content-Type": "application/json" } },
          );
        }
        return new Response("{}", { status: 404 });
      }),
    );
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue({
      spotify_configured: true,
      spotify_signed_in: true,
      llm_ready: false,
      provider: "gemini",
      redirect_uri: "http://127.0.0.1:8765/callback",
      setup_complete: false,
    });

    render(<App />);
    await waitFor(() => expect(screen.getByText(/Gemini — Not connected/)).toBeInTheDocument());
  });
});
