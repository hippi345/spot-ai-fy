import React from "react";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { SetupWizard } from "./SetupWizard";
import * as api from "./lib/api";
import type { SetupStatus } from "./lib/api";

function completeStatus(overrides: Partial<SetupStatus> = {}): SetupStatus {
  return {
    spotify_configured: true,
    spotify_signed_in: true,
    llm_ready: true,
    provider: "ollama",
    redirect_uri: "http://127.0.0.1:8765/callback",
    setup_complete: true,
    ...overrides,
  };
}

function createBaseFetchHandler(signedIn = true, llmProvider = "ollama") {
  return async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.includes("/api/session")) {
      return new Response(JSON.stringify({ signed_in: signedIn, device_id: null }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    if (url.includes("/api/llm") && !url.includes("provider")) {
      return new Response(
        JSON.stringify({
          provider: llmProvider,
          reachable: true,
          configured_model: llmProvider === "gemini" ? "gemini-2.5-flash" : "qwen3:4b-instruct",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }
    if (url.includes("/api/devices")) {
      return new Response(JSON.stringify({ devices: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response("{}", { status: 404 });
  };
}

function mockBaseFetch(signedIn = true, llmProvider = "ollama") {
  vi.stubGlobal("fetch", vi.fn(createBaseFetchHandler(signedIn, llmProvider)));
}

describe("r2_itemE_strict_mode_single_assistant_message", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.history.replaceState({}, "", "/");
  });

  it("appends exactly one assistant bubble under React.StrictMode", async () => {
    mockBaseFetch(true);
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue(completeStatus());

    const sse =
      'data: {"type":"tool_start","name":"spotify_me"}\n\n' +
      'data: {"type":"tool_done","name":"spotify_me","preview":"{}"}\n\n' +
      'data: {"type":"final","text":"Hello from assistant"}\n\n';

    const baseHandler = createBaseFetchHandler(true);
    const baseFetch = fetch as ReturnType<typeof vi.fn>;
    baseFetch.mockImplementation(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/api/chat/stream")) {
        const body = new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode(sse));
            controller.close();
          },
        });
        return new Response(body, {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        });
      }
      return baseHandler(input);
    });

    render(
      <React.StrictMode>
        <App />
      </React.StrictMode>,
    );

    await waitFor(() => expect(screen.getByText(/Spotify — Connected/)).toBeInTheDocument());

    const input = screen.getByRole("textbox");
    fireEvent.change(input, { target: { value: "hi" } });
    fireEvent.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => expect(screen.getByText("Hello from assistant")).toBeInTheDocument());
    expect(screen.getAllByText("Hello from assistant")).toHaveLength(1);
  });
});

describe("r2_itemG_header_refreshes_provider_after_wizard_save", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("invokes onSettingsSaved after Save & test so the host can refresh header status", async () => {
    const onSettingsSaved = vi.fn();
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue(
      completeStatus({ provider: "gemini", llm_ready: false, setup_complete: false }),
    );
    vi.spyOn(api, "saveLlmSetup").mockResolvedValue({
      reachable: true,
      models: ["qwen3:4b-instruct"],
      ollama_cpu_profile: { message: null },
    });

    render(
      <SetupWizard
        allowDismiss
        closeOnComplete={false}
        onComplete={() => undefined}
        onSettingsSaved={onSettingsSaved}
      />,
    );
    await waitFor(() => expect(screen.getByRole("heading", { name: /Step 2 — Language model/i })).toBeInTheDocument());
    await waitFor(() => expect(screen.getByLabelText(/Provider/i)).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText(/Provider/i), { target: { value: "ollama" } });
    fireEvent.click(screen.getByRole("button", { name: /Save & test/i }));
    await waitFor(() => expect(onSettingsSaved).toHaveBeenCalledWith({ provider: "ollama" }));
  });

  it("shows refreshed setup provider in the App header while the wizard stays open", async () => {
    window.history.replaceState({}, "", "/");
    mockBaseFetch(true, "gemini");
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue(
      completeStatus({ provider: "ollama", llm_ready: true }),
    );
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Setup" }));
    await waitFor(() => expect(screen.getByText(/Ollama — Connected/i)).toBeInTheDocument());
  });
});

describe("r2_itemI_device_dropdown_dedupes_auto_option", () => {
  it("renders only one Auto (active device) option in App source", () => {
    const text = readFileSync(resolve(__dirname, "./App.tsx"), "utf8");
    const matches = text.match(/Auto \(active device\)/g) ?? [];
    expect(matches.length).toBe(1);
  });
});

describe("r2_itemJ_settings_wizard_title_and_first_step", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("uses Settings title and Step 1 when setup is complete", async () => {
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue(completeStatus());
    render(<SetupWizard allowDismiss closeOnComplete={false} onComplete={() => undefined} />);
    await waitFor(() => expect(screen.getByRole("heading", { name: /^Settings$/i })).toBeInTheDocument());
    expect(screen.getByRole("heading", { name: /Step 1 — Spotify app/i })).toBeInTheDocument();
  });
});

describe("r2_itemL_gemini_progress_panel_via_stream", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.history.replaceState({}, "", "/");
  });

  it("shows the live progress panel while Gemini SSE tool events stream", async () => {
    mockBaseFetch(true, "gemini");
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue(completeStatus({ provider: "gemini" }));

    let resolveStream: (() => void) | undefined;
    const sse =
      'data: {"type":"tool_start","name":"spotify_me"}\n\n' +
      'data: {"type":"tool_done","name":"spotify_me","preview":"{}"}\n\n';

    const baseHandler = createBaseFetchHandler(true, "gemini");
    const baseFetch = fetch as ReturnType<typeof vi.fn>;
    baseFetch.mockImplementation(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/api/chat/stream")) {
        const body = new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode(sse));
            resolveStream = () => {
              controller.enqueue(
                new TextEncoder().encode('data: {"type":"final","text":"done"}\n\n'),
              );
              controller.close();
            };
          },
        });
        return new Response(body, {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        });
      }
      return baseHandler(input);
    });

    render(<App />);
    await waitFor(() => expect(screen.getByText(/Gemini — Connected/i)).toBeInTheDocument());

    fireEvent.change(screen.getByRole("textbox"), { target: { value: "who am i" } });
    fireEvent.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => expect(screen.getByText(/spotify_me/i)).toBeInTheDocument());
    resolveStream?.();
    await waitFor(() => expect(screen.getByText("done")).toBeInTheDocument());
  });
});

describe("r2_itemL_keepalive_fake_timers", () => {
  it(
    "flags stream stalled after 45s without SSE activity",
    async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      mockBaseFetch(true);
      vi.spyOn(api, "fetchSetupStatus").mockResolvedValue(completeStatus());

      const baseHandler = createBaseFetchHandler(true);
      const baseFetch = fetch as ReturnType<typeof vi.fn>;
      baseFetch.mockImplementation(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/api/chat/stream")) {
          return new Response(
            new ReadableStream({
              start() {
                /* hang until test ends */
              },
            }),
            { status: 200, headers: { "Content-Type": "text/event-stream" } },
          );
        }
        return baseHandler(input);
      });

      render(<App />);
      await act(async () => {
        await vi.runOnlyPendingTimersAsync();
      });
      await waitFor(() => expect(screen.getByRole("textbox")).toBeInTheDocument());
      fireEvent.change(screen.getByRole("textbox"), { target: { value: "slow" } });
      fireEvent.click(screen.getByRole("button", { name: /^Send$/i }));
      await act(async () => {
        await vi.advanceTimersByTimeAsync(48_000);
      });
      await waitFor(() => expect(screen.getByText(/No response for 45s/i)).toBeInTheDocument());
      vi.useRealTimers();
    },
    20_000,
  );
});
