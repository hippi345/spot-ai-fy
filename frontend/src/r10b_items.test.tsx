/** Round-10b: per-chat conversation_id in UI requests and persistence. */

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import * as api from "./lib/api";
import type { SetupStatus } from "./lib/api";
import {
  CHAT_SESSION_STORAGE_KEY,
  buildChatStreamRequestBody,
  createConversationId,
  loadChatSession,
  saveChatSession,
  startNewChatSession,
} from "./lib/chatSession";

function completeStatus(): SetupStatus {
  return {
    spotify_configured: true,
    spotify_signed_in: true,
    llm_ready: true,
    provider: "ollama",
    redirect_uri: "http://127.0.0.1:8765/callback",
    setup_complete: true,
  };
}

function sseResponse(text = "OK") {
  const body =
    `data: {"type":"final","text":"${text}"}\n\n` + `data: {"type":"done"}\n\n`;
  return new Response(body, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

describe("test_r10b_item1_conversation_id_in_chat_requests", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();
  });

  it("test_r10b_item1_request_body_includes_conversation_id", () => {
    const id = "11111111-1111-4111-8111-111111111111";
    const body = buildChatStreamRequestBody("hello", [], id);
    expect(body.conversation_id).toBe(id);
    expect(body.message).toBe("hello");
  });

  it("test_r10b_item1_same_id_across_two_messages_in_one_chat", async () => {
    const streamBodies: Array<{ conversation_id?: string; message?: string }> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/api/chat/stream")) {
          streamBodies.push(JSON.parse(String(init?.body ?? "{}")));
          return sseResponse();
        }
        if (url.includes("/api/session")) {
          return new Response(JSON.stringify({ signed_in: true, device_id: null }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          });
        }
        if (url.includes("/api/llm")) {
          return new Response(
            JSON.stringify({
              configured_model: "m",
              reachable: true,
              models: [],
              error: null,
            }),
            { status: 200, headers: { "Content-Type": "application/json" } },
          );
        }
        return new Response("{}", { status: 404 });
      }),
    );
    vi.spyOn(api, "fetchSetupStatus").mockResolvedValue(completeStatus());

    render(<App />);
    const chat = screen.getByPlaceholderText(/Play John Mayer/i);
    await waitFor(() => expect(chat).not.toBeDisabled());

    fireEvent.change(chat, { target: { value: "first message" } });
    fireEvent.click(screen.getByRole("button", { name: /^Send$/i }));
    await waitFor(() => expect(streamBodies.length).toBeGreaterThanOrEqual(1));

    fireEvent.change(chat, { target: { value: "second message" } });
    fireEvent.click(screen.getByRole("button", { name: /^Send$/i }));
    await waitFor(() => expect(streamBodies.length).toBeGreaterThanOrEqual(2));

    expect(streamBodies[0].conversation_id).toBeTruthy();
    expect(streamBodies[1].conversation_id).toBe(streamBodies[0].conversation_id);
  });

  it("test_r10b_item1_new_chat_rotates_conversation_id", () => {
    saveChatSession({
      conversationId: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
      messages: [{ role: "user", text: "hi" }],
    });
    const loaded = loadChatSession();
    const before = loaded.conversationId;
    const fresh = startNewChatSession();
    expect(fresh.conversationId).not.toBe(before);
    expect(loadChatSession().conversationId).toBe(fresh.conversationId);
    expect(loadChatSession().messages).toEqual([]);
  });

  it("test_r10b_item1_reload_restores_conversation_id_from_storage", () => {
    const id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";
    saveChatSession({
      conversationId: id,
      messages: [{ role: "user", text: "saved" }],
    });
    const restored = loadChatSession();
    expect(restored.conversationId).toBe(id);
    expect(restored.messages).toEqual([{ role: "user", text: "saved" }]);
    expect(localStorage.getItem(CHAT_SESSION_STORAGE_KEY)).toContain(id);
  });

  it("test_r10b_item1_createConversationId_fallback_when_uuid_missing", () => {
    const orig = globalThis.crypto;
    Object.defineProperty(globalThis, "crypto", {
      value: { randomUUID: undefined },
      configurable: true,
    });
    const id = createConversationId();
    expect(id.startsWith("conv-")).toBe(true);
    Object.defineProperty(globalThis, "crypto", { value: orig, configurable: true });
  });
});
