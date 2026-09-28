"""Lenient parsing for /api/chat and /api/chat/stream request bodies."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from spot_backend.agent import _coerce_chat_history

logger = logging.getLogger(__name__)

# Keep in sync with frontend buildChatStreamRequestBody (slice -40).
CHAT_HISTORY_MAX_TURNS = 40
CHAT_MESSAGE_MAX_LEN = 48_000
CHAT_CONVERSATION_ID_MAX_LEN = 128


def trim_chat_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(history) <= CHAT_HISTORY_MAX_TURNS:
        return history
    trimmed = history[-CHAT_HISTORY_MAX_TURNS :]
    logger.warning(
        "chat_history_trimmed dropped=%s kept=%s",
        len(history) - len(trimmed),
        len(trimmed),
    )
    return trimmed


def format_validation_errors(errors: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for err in errors[:6]:
        loc = ".".join(str(x) for x in err.get("loc") or ())
        msg = str(err.get("msg") or "invalid value")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) or "Invalid chat request."


class ChatBody(BaseModel):
    """POST body for /api/chat and /api/chat/stream (lenient history; never reject on length)."""

    model_config = ConfigDict(extra="ignore")

    message: str = ""
    history: list[dict[str, str]] | None = None
    conversation_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_shapes(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return data

    @model_validator(mode="after")
    def _normalize(self) -> ChatBody:
        msg = (self.message or "").strip()
        if not msg:
            raise ValueError("message is required")
        if len(msg) > CHAT_MESSAGE_MAX_LEN:
            msg = msg[:CHAT_MESSAGE_MAX_LEN]
        self.message = msg

        coerced = _coerce_chat_history(self.history)
        if coerced:
            for row in coerced:
                text = row.get("content") or ""
                if len(text) > CHAT_MESSAGE_MAX_LEN:
                    row["content"] = text[:CHAT_MESSAGE_MAX_LEN]
        self.history = trim_chat_history(coerced) if coerced else None

        conv = (self.conversation_id or "").strip() or None
        if conv and len(conv) > CHAT_CONVERSATION_ID_MAX_LEN:
            conv = conv[:CHAT_CONVERSATION_ID_MAX_LEN]
        self.conversation_id = conv
        return self


def dump_chat_history(body: ChatBody) -> list[dict[str, str]] | None:
    if not body.history:
        return None
    return [{"role": t["role"], "content": t["content"]} for t in body.history]
