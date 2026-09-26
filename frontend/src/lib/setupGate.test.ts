import { describe, expect, it } from "vitest";

import type { SetupStatus } from "./api";
import { isChatBlockedBySetup } from "./setupGate";

const base: SetupStatus = {
  spotify_configured: true,
  spotify_signed_in: false,
  llm_ready: false,
  provider: "ollama",
  redirect_uri: "http://127.0.0.1:8765/callback",
  setup_complete: false,
};

describe("isChatBlockedBySetup", () => {
  it("blocks when Spotify is not signed in", () => {
    expect(isChatBlockedBySetup({ ...base, llm_ready: true })).toBe(true);
  });

  it("blocks when LLM is not ready", () => {
    expect(isChatBlockedBySetup({ ...base, spotify_signed_in: true })).toBe(true);
  });

  it("unblocks when both Spotify and LLM are ready", () => {
    expect(
      isChatBlockedBySetup({ ...base, spotify_signed_in: true, llm_ready: true }),
    ).toBe(false);
  });
});
