import { describe, expect, it } from "vitest";

import { FRIENDLY_SPOTIFY_GUIDANCE } from "./lib/chatMessages";
import {
  CHAT_HISTORY_MAX_TURNS,
  buildChatStreamRequestBody,
  coerceChatHistoryTurns,
} from "./lib/chatSession";

describe("coerceChatHistoryTurns", () => {
  it("drops empty, fallback, and keeps middle-dot playlist titles", () => {
    const out = coerceChatHistoryTurns([
      { role: "user", content: "Play one of my playlists" },
      { role: "assistant", content: "Playing Jamz · My Artists · 2026-09-25." },
      { role: "assistant", content: "" },
      { role: "assistant", content: FRIENDLY_SPOTIFY_GUIDANCE },
      { role: "user", text: "legacy" },
    ]);
    expect(out).toEqual([
      { role: "user", content: "Play one of my playlists" },
      { role: "assistant", content: "Playing Jamz · My Artists · 2026-09-25." },
      { role: "user", content: "legacy" },
    ]);
  });

  it("trims to CHAT_HISTORY_MAX_TURNS", () => {
    const rows = Array.from({ length: CHAT_HISTORY_MAX_TURNS + 5 }, (_, i) => ({
      role: i % 2 === 0 ? "user" : "assistant",
      content: `turn ${i}`,
    })) as Array<{ role: "user" | "assistant"; content: string }>;
    expect(coerceChatHistoryTurns(rows).length).toBe(CHAT_HISTORY_MAX_TURNS);
  });
});

describe("buildChatStreamRequestBody", () => {
  it("matches documented JSON shape", () => {
    const body = buildChatStreamRequestBody(
      "What's playing?",
      [{ role: "user", content: "hi" }],
      "conv-1",
    );
    expect(body).toEqual({
      message: "What's playing?",
      history: [{ role: "user", content: "hi" }],
      conversation_id: "conv-1",
    });
  });
});
