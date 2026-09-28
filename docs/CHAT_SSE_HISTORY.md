# Chat SSE request body (`POST /api/chat/stream`)

The UI sends one JSON object per turn. The **current** user text is always in `message`; prior turns are in `history`.

## Shape (canonical)

```json
{
  "message": "How many albums does Drake have?",
  "conversation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "history": [
    { "role": "user", "content": "Play something by The Weeknd" },
    { "role": "assistant", "content": "Playing The Weeknd — Blinding Lights." },
    { "role": "user", "content": "play his latest single" },
    { "role": "assistant", "content": "Playing the latest single …" },
    { "role": "user", "content": "Can you play podcasts via this interface?" },
    { "role": "assistant", "content": "No — podcast playback is not supported here." },
    { "role": "user", "content": "Play one of my playlists" },
    {
      "role": "assistant",
      "content": "Playing Jamz · My Artists · 2026-09-25."
    }
  ]
}
```

## Fields

| Field | Type | Required | Limits / notes |
|-------|------|----------|----------------|
| `message` | string | yes | Non-empty after trim; max 48 000 chars (backend truncates). |
| `conversation_id` | string | no | Stable per chat tab; UUID from `createConversationId()`; max 128 chars. |
| `history` | array | no | Up to **40** turns kept server-side (oldest dropped); see below. |

## History turn objects

- **Canonical:** `{ "role": "user" \| "assistant", "content": "<text>" }`
- **Legacy (still accepted):** `{ "role": "...", "text": "<text>" }` — `text` is mapped to `content`.
- Empty `content` / `text` rows are dropped.
- Assistant rows matching the empty-reply fallback copy (`FRIENDLY_SPOTIFY_GUIDANCE`, stock “no assistant text” hints, etc.) are dropped and not replayed to the model.
- Extra keys (e.g. `trace`) are ignored.

## Frontend trimming (`frontend/src/lib/chatSession.ts`)

`buildChatStreamRequestBody()`:

1. Takes in-memory messages as `{ role, content }`.
2. Drops assistant fallback rows (same rules as the UI history filter).
3. Drops empty content.
4. Keeps the last **40** turns (`slice(-40)`).
5. Sends the new user phrase only in `message`, not duplicated in `history`.

`App.tsx` builds `historyPayload` from `messages` with the same filters before calling `buildChatStreamRequestBody`.

## Validation failures

Malformed bodies that previously returned HTTP **422** from FastAPI now get a **plain-language** response:

- **`/api/chat/stream`:** HTTP **200** with SSE `error` + `done` events (UI shows the message).
- **`/api/chat`:** HTTP **400** with `{ "detail": "<string>" }`.
