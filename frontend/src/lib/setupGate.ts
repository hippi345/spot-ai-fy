import type { SetupStatus } from "./api";

/** True while the chat UI should stay disabled (Spotify + LLM not both ready). */
export function isChatBlockedBySetup(status: SetupStatus): boolean {
  return !(status.spotify_signed_in && status.llm_ready);
}
