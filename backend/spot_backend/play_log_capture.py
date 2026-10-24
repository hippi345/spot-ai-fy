"""Turn-scoped context for play log capture (request text per agent turn)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

_lock = threading.Lock()


@dataclass
class TurnContext:
    request_text: str = ""
    conversation_id: str | None = None


_state = TurnContext()


def bind_turn_context(request_text: str, conversation_id: str | None = None) -> None:
    with _lock:
        global _state
        _state = TurnContext(
            request_text=(request_text or "").strip(),
            conversation_id=(conversation_id or "").strip() or None,
        )


def current_turn_context() -> TurnContext:
    with _lock:
        return _state


def clear_turn_context() -> None:
    bind_turn_context("", None)


def record_playback_started(
    runner: Any,
    *,
    track_uri: str,
    title: str,
    artist: str,
    album: str,
) -> int | None:
    """Insert a play_log row when playback through Spot-AI-fy succeeds."""
    from spot_backend.play_log import PlayLogStore, play_log_enabled

    if not play_log_enabled(runner.settings):
        return None
    ctx = current_turn_context()
    store = PlayLogStore(settings=runner.settings)
    return store.insert_play(
        track_uri=track_uri,
        title=title or track_uri,
        artist=artist or "",
        album=album or "",
        request_text=ctx.request_text or None,
        conversation_id=runner.conversation_id or ctx.conversation_id,
    )


def record_skip_for_uri(runner: Any, track_uri: str) -> None:
    from spot_backend.play_log import PlayLogStore, play_log_enabled

    if not play_log_enabled(runner.settings) or not track_uri:
        return
    PlayLogStore(settings=runner.settings).mark_latest_open_skipped(track_uri.strip())
