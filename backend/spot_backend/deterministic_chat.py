"""Wire deterministic shortcuts into chat event streams."""

from __future__ import annotations

import os
from typing import Any, Callable, Iterator

from spot_backend.chat_messages import prepare_user_visible_reply
from spot_backend.config import Settings
from spot_backend.reply_tool_trace import persist_shortcut_tool_steps
from spot_backend.chat_shortcuts import (
    try_deterministic_chat_reply,
    try_deterministic_current_track_release_reply,
    try_deterministic_recently_played_reply,
)
from spot_backend.capability_replies import try_capability_question_reply
from spot_backend.deterministic_chat_types import DeterministicChatResult
from spot_backend.spotify_tools import SpotifyToolRunner


def deterministic_chat_shortcuts_disabled() -> bool:
    """When true, all chat turns go through the LLM tool loop (used by Ollama smoke tests)."""
    raw = os.environ.get("SPOT_AI_FY_DISABLE_DETERMINISTIC_CHAT", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def resolve_deterministic_chat_outcome(
    user_text: str,
    runner: SpotifyToolRunner,
    *,
    conversation_id: str | None,
) -> DeterministicChatResult | None:
    if deterministic_chat_shortcuts_disabled():
        return None
    cap = try_capability_question_reply(user_text)
    if cap:
        return DeterministicChatResult(cap, [])
    outcome = try_deterministic_current_track_release_reply(user_text, runner)
    if outcome is not None:
        return outcome
    outcome = try_deterministic_recently_played_reply(user_text, runner)
    if outcome is not None:
        return outcome
    return try_deterministic_chat_reply(
        user_text,
        runner,
        conversation_id=conversation_id,
    )


def persist_deterministic_shortcut_traces(
    settings: Settings,
    outcome: DeterministicChatResult,
    *,
    conversation_id: str | None,
    known_secrets: list[str] | None = None,
) -> None:
    persist_shortcut_tool_steps(
        settings.data_dir,
        outcome.tool_steps,
        conversation_id=conversation_id,
        known_secrets=known_secrets,
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
    settings: Settings | None = None,
) -> Iterator[dict[str, Any]] | None:
    outcome = resolve_deterministic_chat_outcome(
        user_text,
        runner,
        conversation_id=conversation_id,
    )
    if outcome is None:
        return None
    if settings is not None:
        persist_deterministic_shortcut_traces(
            settings,
            outcome,
            conversation_id=conversation_id,
        )
    return iter_deterministic_shortcut_events(outcome)


def gemini_deterministic_shortcut_reply(
    user_text: str,
    runner: SpotifyToolRunner,
    *,
    conversation_id: str | None,
    emit: Callable[[dict[str, Any]], None] | None = None,
    settings: Settings | None = None,
    known_secrets: list[str] | None = None,
) -> str | None:
    outcome = resolve_deterministic_chat_outcome(
        user_text,
        runner,
        conversation_id=conversation_id,
    )
    if outcome is None:
        return None
    if settings is not None:
        persist_deterministic_shortcut_traces(
            settings,
            outcome,
            conversation_id=conversation_id,
            known_secrets=known_secrets,
        )
    if emit:
        for ev in iter_deterministic_shortcut_events(outcome):
            if ev.get("type") in ("tool_start", "tool_done"):
                emit(ev)
    return prepare_user_visible_reply(outcome.reply, outcome.tool_raw_results())
