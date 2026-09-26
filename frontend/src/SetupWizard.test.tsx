import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SetupWizard } from "./SetupWizard";
import * as api from "./lib/api";

describe("SetupWizard", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("shows redirect URI from setup status", async () => {
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue({
      spotify_configured: false,
      spotify_signed_in: false,
      llm_ready: false,
      provider: "ollama",
      redirect_uri: "http://127.0.0.1:8765/callback",
      setup_complete: false,
    });

    render(<SetupWizard allowDismiss={false} onComplete={() => undefined} />);

    await waitFor(() => {
      expect(screen.getByText("http://127.0.0.1:8765/callback")).toBeInTheDocument();
    });
  });
});
