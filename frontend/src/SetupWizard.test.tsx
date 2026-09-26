import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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
    ...overrides,
  };
}

async function renderOnStep2(overrides: Partial<SetupStatus> = {}) {
  vi.spyOn(api, "fetchSetupStatus").mockResolvedValue(step2Status(overrides));
  render(<SetupWizard allowDismiss={false} onComplete={() => undefined} />);
  await waitFor(() => {
    expect(screen.getByRole("heading", { name: /Step 2 — Language model/i })).toBeInTheDocument();
  });
}

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

describe("SetupWizard step 2 Ollama validation hints (item 4)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("does not show backend LLM error when default URL and model are present", async () => {
    await renderOnStep2({
      llm_error: "Ollama URL is required",
    });

    expect(screen.getByLabelText(/Ollama URL/i)).toHaveValue("http://127.0.0.1:11434");
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/qwen3:4b-instruct/i)).toHaveValue("qwen3:4b-instruct");
    });
    expect(screen.queryByText("Ollama URL is required")).not.toBeInTheDocument();
    expect(screen.queryByText(/Ollama host and model are required/i)).not.toBeInTheDocument();
  });

  it("shows a clear message when the Ollama URL is empty", async () => {
    await renderOnStep2({
      llm_error: "Ollama URL is required",
      ollama_host: "",
    });

    const hostInput = screen.getByLabelText(/Ollama URL/i);
    fireEvent.change(hostInput, { target: { value: "" } });

    await waitFor(() => {
      expect(screen.getByText("Ollama URL is required")).toBeInTheDocument();
    });
  });

  it("shows a clear message when the model field is empty", async () => {
    await renderOnStep2({ ollama_model: "" });

    const modelInput = screen.getByPlaceholderText(/qwen3:4b-instruct/i);
    fireEvent.change(modelInput, { target: { value: "" } });

    expect(
      screen.getByText(/Select or enter an Ollama model name \(try Detect models\)/i),
    ).toBeInTheDocument();
  });
});

describe("SetupWizard Detect models layout (item 5)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders Detect models in its own row, separate from URL and model inputs", async () => {
    await renderOnStep2();

    const hostInput = screen.getByLabelText(/Ollama URL/i);
    const modelInput = screen.getByPlaceholderText(/qwen3:4b-instruct/i);
    const detectButton = screen.getByRole("button", { name: /Detect models/i });
    const detectRow = detectButton.closest(".btn-row");

    expect(detectRow).toBeTruthy();
    expect(within(detectRow as HTMLElement).queryByRole("button", { name: /Detect models/i })).toBe(detectButton);
    expect(detectRow?.contains(hostInput)).toBe(false);
    expect(detectRow?.contains(modelInput)).toBe(false);
  });
});

describe("SetupWizard LLM setup payload", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("sends ollama_allow_public when the public-host checkbox is checked", async () => {
    const save = vi.spyOn(api, "saveLlmSetup").mockResolvedValue({ reachable: true, models: [] });
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue(step2Status());

    render(<SetupWizard allowDismiss={false} onComplete={() => undefined} />);
    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /Step 2 — Language model/i })).toBeInTheDocument();
    });

    fireEvent.click(
      screen.getByRole("checkbox", { name: /Allow Ollama on the public internet/i }),
    );
    fireEvent.click(screen.getByRole("button", { name: /Save & test/i }));

    await waitFor(() => expect(save).toHaveBeenCalled());
    expect(save.mock.calls[0][0]).toMatchObject({
      ollama_allow_public: true,
      provider: "ollama",
    });
  });

  it("sends ollama_small_model_mode in the setup payload", async () => {
    const save = vi.spyOn(api, "saveLlmSetup").mockResolvedValue({ reachable: true, models: [] });
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue(step2Status());

    render(<SetupWizard allowDismiss={false} onComplete={() => undefined} />);
    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /Step 2 — Language model/i })).toBeInTheDocument();
    });

    fireEvent.change(screen.getByLabelText(/Small-model tool set/i), { target: { value: "on" } });
    fireEvent.click(screen.getByRole("button", { name: /Save & test/i }));

    await waitFor(() => expect(save).toHaveBeenCalled());
    expect(save.mock.calls[0][0]).toMatchObject({
      ollama_small_model_mode: "on",
    });
  });
});
