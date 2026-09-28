/** Per-chat conversation id + optional message persistence for undo across turns. */

import { isUnpersistedAssistantFallback } from "./chatMessages";

export const CHAT_SESSION_STORAGE_KEY = "spotaify.chatSession";

/** Must match backend spot_backend.chat_request.CHAT_HISTORY_MAX_TURNS */
export const CHAT_HISTORY_MAX_TURNS = 40;

export type PersistedChatMessage = {
  role: "user" | "assistant";
  text: string;
};

export type ChatSessionState = {
  conversationId: string;
  messages: PersistedChatMessage[];
};

export function createConversationId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `conv-${Date.now()}-${Math.random().toString(36).slice(2, 12)}`;
}

export function loadChatSession(): ChatSessionState {
  try {
    const raw = localStorage.getItem(CHAT_SESSION_STORAGE_KEY);
    if (!raw) {
      return { conversationId: createConversationId(), messages: [] };
    }
    const parsed = JSON.parse(raw) as Partial<ChatSessionState>;
    const conversationId =
      typeof parsed.conversationId === "string" && parsed.conversationId.trim()
        ? parsed.conversationId.trim()
        : createConversationId();
    const messages = Array.isArray(parsed.messages)
      ? parsed.messages.filter(
          (m): m is PersistedChatMessage =>
            Boolean(m) &&
            typeof m === "object" &&
            (m.role === "user" || m.role === "assistant") &&
            typeof m.text === "string",
        )
      : [];
    return { conversationId, messages };
  } catch {
    return { conversationId: createConversationId(), messages: [] };
  }
}

export function saveChatSession(state: ChatSessionState): void {
  try {
    localStorage.setItem(CHAT_SESSION_STORAGE_KEY, JSON.stringify(state));
  } catch {
    /* ignore quota / private mode */
  }
}

export function startNewChatSession(): ChatSessionState {
  const fresh = { conversationId: createConversationId(), messages: [] };
  saveChatSession(fresh);
  return fresh;
}

export type ChatHistoryTurn = { role: "user" | "assistant"; content: string };

type LegacyHistoryTurn = {
  role?: string;
  content?: string;
  text?: string;
};

/** Normalize UI/history rows to canonical `{ role, content }` for the API. */
export function coerceChatHistoryTurns(
  turns: Array<ChatHistoryTurn | LegacyHistoryTurn>,
): ChatHistoryTurn[] {
  const out: ChatHistoryTurn[] = [];
  for (const row of turns) {
    if (!row || typeof row !== "object") continue;
    const role = row.role;
    const legacyText = "text" in row && typeof row.text === "string" ? row.text : "";
    const raw = (row.content ?? legacyText).trim();
    if (role !== "user" && role !== "assistant") continue;
    if (!raw) continue;
    if (role === "assistant" && isUnpersistedAssistantFallback(raw)) continue;
    out.push({ role, content: raw });
  }
  if (out.length > CHAT_HISTORY_MAX_TURNS) {
    return out.slice(-CHAT_HISTORY_MAX_TURNS);
  }
  return out;
}

export function buildChatStreamRequestBody(
  message: string,
  history: Array<ChatHistoryTurn | LegacyHistoryTurn>,
  conversationId: string,
): { message: string; history: ChatHistoryTurn[]; conversation_id: string } {
  const trimmedMessage = message.trim();
  return {
    message: trimmedMessage,
    history: coerceChatHistoryTurns(history),
    conversation_id: conversationId.trim(),
  };
}
