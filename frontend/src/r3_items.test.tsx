/** Round-3 item tests (vitest). */

import { useState } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SetupWizard } from "./SetupWizard";
import * as api from "./lib/api";
import type { SetupStatus } from "./lib/api";
import {
  reduceTraceFinishAllRunning,
  reduceTracePushStep,
  type TraceStep,
} from "./lib/chatTrace";

function step2Status(overrides: Partial<SetupStatus> = {}): SetupStatus {
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

// r3_item06 — wizard stays on step 2 after Save & test
describe("r3_item06_setup_wizard_save_preserves_step", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("keeps step 2 and shows saved notice after Save & test", async () => {
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue(
      step2Status({ setup_complete: false, ollama_model: "qwen3:4b-instruct" }),
    );
    const save = vi.spyOn(api, "saveLlmSetup").mockResolvedValue({
      reachable: true,
      models: ["qwen3:4b-instruct"],
      ollama_cpu_profile: { message: "Runs on CPU only" },
    });

    render(<SetupWizard allowDismiss={true} onComplete={() => undefined} />);
    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /Step 2 — Language model/i })).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /Save & test/i }));

    await waitFor(() => expect(save).toHaveBeenCalled());
    await waitFor(() => {
      expect(screen.getByText(/Saved — settings tested successfully/i)).toBeInTheDocument();
      expect(screen.getByText(/Runs on CPU only/i)).toBeInTheDocument();
      expect(screen.getByRole("heading", { name: /Step 2 — Language model/i })).toBeInTheDocument();
    });
  });
});

// r3_item08 — trace reducer + single-chunk SSE
describe("r3_item08_chat_trace_reducer", () => {
  it("accumulates tool steps for a single SSE chunk", () => {
    let steps: TraceStep[] = [];
    steps = reduceTracePushStep(steps, { kind: "tool", label: "spotify_pause", status: "running" }, 1);
    steps = reduceTraceFinishAllRunning(steps);
    expect(steps.some((s) => s.label === "spotify_pause")).toBe(true);
    expect(steps.every((s) => s.status === "done")).toBe(true);
  });
});

// r3_item09 — header provider chip updates while the wizard is open (host refresh hook)
describe("r3_item09_header_provider_updates_on_save", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  function ProviderChipHost() {
    const [provider, setProvider] = useState<"ollama" | "gemini">("ollama");
    return (
      <>
        <p className="status-chips">{provider === "gemini" ? "Gemini — Connected" : "Ollama — Connected"}</p>
        <SetupWizard
          allowDismiss
          onComplete={() => undefined}
          onSettingsSaved={(patch) => {
            if (patch?.provider === "gemini" || patch?.provider === "ollama") {
              setProvider(patch.provider);
            }
          }}
        />
      </>
    );
  }

  it("updates the host provider chip when the wizard saves", async () => {
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue(
      step2Status({ provider: "ollama", setup_complete: false }),
    );
    vi.spyOn(api, "saveLlmSetup").mockResolvedValue({ reachable: true, models: ["gemini-2.5-flash"] });

    render(<ProviderChipHost />);
    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /Step 2 — Language model/i })).toBeInTheDocument();
    });
    fireEvent.change(screen.getByLabelText(/Provider/i), { target: { value: "gemini" } });
    fireEvent.click(screen.getByRole("button", { name: /Save & test/i }));

    await waitFor(() => {
      expect(screen.getByText(/Gemini — Connected/i)).toBeInTheDocument();
    });
  });
});
