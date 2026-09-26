"""Wire deterministic shortcuts into chat event streams."""

from __future__ import annotations

from typing import Any, Callable, Iterator

from spot_backend.chat_messages import prepare_user_visible_reply
from spot_backend.chat_shortcuts import (
    try_deterministic_chat_reply,
    try_deterministic_recently_played_reply,
)
from spot_backend.deterministic_chat_types import DeterministicChatResult
from spot_backend.spotify_tools import SpotifyToolRunner


def resolve_deterministic_chat_outcome(
    user_text: str,
    runner: SpotifyToolRunner,
    *,
    conversation_id: str | None,
) -> DeterministicChatResult | None:
    outcome = try_deterministic_recently_played_reply(user_text, runner)
    if outcome is not None:
        return outcome
    return try_deterministic_chat_reply(
        user_text,
        runner,
        conversation_id=conversation_id,
    )


def iter_deterministic_shortcut_events(
    outcome: DeterministicChatResult,
) -> Iterator[dict[str, Any]]:
    for name, _, raw in outcome.tool_steps:
        preview = raw[:240] + ("…" if len(raw) > 240 else "")
        yield {"type": "tool_start", "name": name}
        yield {"type": "tool_done", "name": name, "preview": preview}
    yield {
        "type": "final",
        "text": prepare_user_visible_reply(outcome.reply, outcome.tool_raw_results()),
    }


def ollama_deterministic_shortcut_events(
    user_text: str,
    runner: SpotifyToolRunner,
    *,
    conversation_id: str | None,
) -> Iterator[dict[str, Any]] | None:
    outcome = resolve_deterministic_chat_outcome(
        user_text,
        runner,
        conversation_id=conversation_id,
    )
    if outcome is None:
        return None
    return iter_deterministic_shortcut_events(outcome)


def gemini_deterministic_shortcut_reply(
    user_text: str,
    runner: SpotifyToolRunner,
    *,
    conversation_id: str | None,
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> str | None:
    outcome = resolve_deterministic_chat_outcome(
        user_text,
        runner,
        conversation_id=conversation_id,
    )
    if outcome is None:
        return None
    if emit:
        for ev in iter_deterministic_shortcut_events(outcome):
            if ev.get("type") in ("tool_start", "tool_done"):
                emit(ev)
    return prepare_user_visible_reply(outcome.reply, outcome.tool_raw_results())
