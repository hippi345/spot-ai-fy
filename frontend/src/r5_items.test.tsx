/** Round-5 item tests (vitest). */

import { useState } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SetupWizard } from "./SetupWizard";
import * as api from "./lib/api";
import type { SetupStatus } from "./lib/api";

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

// r5-item6 — fails if didInitialRefresh / preserveStep on save are reverted
describe("r5-item6_setup_wizard_save_preserves_step_when_setup_completes", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("stays on step 2 with saved notice after LLM save marks setup_complete", async () => {
    const fetchSpy = vi
      .spyOn(api, "fetchSetupStatus")
      .mockResolvedValueOnce(step2Status({ setup_complete: false }))
      .mockResolvedValue(step2Status({ setup_complete: true, llm_ready: true }));

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
          <SetupWizard
            allowDismiss
            onComplete={() => tick}
            onSettingsSaved={() => setTick((t) => t + 1)}
          />
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

    expect(fetchSpy).toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /^bump/i }));
    await waitFor(() => {
      expect(screen.getByText(/Saved — settings tested successfully/i)).toBeInTheDocument();
      expect(screen.getByText(/Runs on CPU only/i)).toBeInTheDocument();
      expect(screen.getByRole("heading", { name: /Step 2 — Language model/i })).toBeInTheDocument();
      expect(screen.queryByRole("heading", { name: /Step 1 — Spotify app/i })).not.toBeInTheDocument();
    });
  });
});
