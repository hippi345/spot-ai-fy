/** Per-chat conversation id + optional message persistence for undo across turns. */

export const CHAT_SESSION_STORAGE_KEY = "spotaify.chatSession";

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

export function buildChatStreamRequestBody(
  message: string,
  history: ChatHistoryTurn[],
  conversationId: string,
): { message: string; history: ChatHistoryTurn[]; conversation_id: string } {
  return {
    message,
    history,
    conversation_id: conversationId,
  };
}
