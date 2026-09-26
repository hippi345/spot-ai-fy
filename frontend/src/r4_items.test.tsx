/** Round-4 item tests (vitest). */

import { useState } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SetupWizard } from "./SetupWizard";
import * as api from "./lib/api";
import type { SetupStatus } from "./lib/api";
import {
  applyChatSseEventToTrace,
  consumeSseReadableStream,
} from "./lib/chatSse";
import { reduceTraceFinishAllRunning, type TraceStep } from "./lib/chatTrace";

function step2Status(overrides: Partial<SetupStatus> = {}): SetupStatus {
  return {
    spotify_configured: true,
    spotify_signed_in: true,
    llm_ready: false,
    provider: "ollama",
    redirect_uri: "http://127.0.0.1:8765/callback",
    setup_complete: false,
    ollama_model: "qwen3:4b-instruct",
    ...overrides,
  };
}

// r4-item6
describe("r4-item6_setup_wizard_save_preserves_step_on_parent_rerender", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("keeps step 2 and saved notice when parent passes a new onComplete", async () => {
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue(step2Status());
    vi.spyOn(api, "saveLlmSetup").mockResolvedValue({
      reachable: true,
      models: ["qwen3:4b-instruct"],
      ollama_cpu_profile: { message: "Runs on CPU only" },
    });

    function Host() {
      const [tick, setTick] = useState(0);
      return (
        <>
          <button type="button" onClick={() => setTick((t) => t + 1)}>
            bump {tick}
          </button>
          <SetupWizard allowDismiss onComplete={() => tick} onSettingsSaved={() => setTick((t) => t + 1)} />
        </>
      );
    }

    render(<Host />);
    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /Step 2 — Language model/i })).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /Save & test/i }));
    await waitFor(() => {
      expect(screen.getByText(/Saved — settings tested successfully/i)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /^bump/i }));
    await waitFor(() => {
      expect(screen.getByText(/Saved — settings tested successfully/i)).toBeInTheDocument();
      expect(screen.getByText(/Runs on CPU only/i)).toBeInTheDocument();
      expect(screen.getByRole("heading", { name: /Step 2 — Language model/i })).toBeInTheDocument();
    });
  });
});

// r4-item11e — real SSE ReadableStream through parser
describe("r4-item11e_sse_single_chunk_parsing", () => {
  it("feeds one SSE chunk through consumeSseReadableStream", async () => {
    const sse =
      'data: {"type":"tool_start","name":"spotify_pause"}\n\n' +
      'data: {"type":"tool_done","name":"spotify_pause","preview":"{}"}\n\n' +
      'data: {"type":"final","text":"Paused."}\n\n';
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(sse));
        controller.close();
      },
    });
    const seen: string[] = [];
    await consumeSseReadableStream(stream, (ev) => {
      seen.push(String(ev.type));
    });
    expect(seen).toEqual(["tool_start", "tool_done", "final"]);
  });

  it("maps parsed SSE events into trace steps", async () => {
    let steps: TraceStep[] = [];
    let nextId = 1;
    const sse = 'data: {"type":"tool_start","name":"spotify_pause"}\n\n';
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(sse));
        controller.close();
      },
    });
    await consumeSseReadableStream(stream, (ev) => {
      steps = applyChatSseEventToTrace(steps, ev, () => nextId++);
    });
    steps = reduceTraceFinishAllRunning(steps);
    expect(steps.some((s) => s.label === "spotify_pause")).toBe(true);
  });
});
