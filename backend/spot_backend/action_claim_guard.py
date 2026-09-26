"""Detect assistant replies that claim Spotify actions without a matching successful tool call."""

from __future__ import annotations

import json
import re
from typing import Any

_PLAYBACK_TOOLS = frozenset(
    {
        "spotify_start_resume_playback",
        "spotify_play_playlist",
        "spotify_play_next",
        "spotify_add_to_queue",
        "spotify_transfer_playback",
    }
)
_PAUSE_TOOLS = frozenset({"spotify_pause"})
_SKIP_TOOLS = frozenset({"spotify_skip_next", "spotify_skip_previous"})
_VOLUME_TOOLS = frozenset({"spotify_set_volume"})
_LIBRARY_SAVE_TOOLS = frozenset(
    {
        "spotify_save_tracks",
        "spotify_save_albums",
        "spotify_follow_artist",
        "spotify_follow_playlist",
        "spotify_add_tracks_to_playlist",
        "spotify_add_tracks_by_query",
    }
)
_LIBRARY_REMOVE_TOOLS = frozenset(
    {
        "spotify_unsave_tracks",
        "spotify_unsave_albums",
        "spotify_unfollow_artist",
        "spotify_unfollow_playlist",
        "spotify_remove_playlist_tracks",
    }
)

_CLAIM_RULES: list[tuple[re.Pattern[str], frozenset[str]]] = [
    (re.compile(r"\bnow playing\b", re.I), _PLAYBACK_TOOLS),
    (re.compile(r"\b(started|starting) (playing|playback)\b", re.I), _PLAYBACK_TOOLS),
    (re.compile(r"\bplaying\b.+\b(on|in) your\b", re.I), _PLAYBACK_TOOLS),
    (re.compile(r"\b(play(ing)? (music|songs|tracks) by)\b", re.I), _PLAYBACK_TOOLS),
    (re.compile(r"\b(paused|pause(d)? playback)\b", re.I), _PAUSE_TOOLS),
    (re.compile(r"\b(skipped|skipping)\b", re.I), _SKIP_TOOLS),
    (re.compile(r"\b(volume (is )?set|set (the )?volume)\b", re.I), _VOLUME_TOOLS),
    (re.compile(r"\b(liked|saved|added to (your )?library)\b", re.I), _LIBRARY_SAVE_TOOLS),
    (re.compile(r"\b(followed|now following)\b", re.I), _LIBRARY_SAVE_TOOLS | frozenset({"spotify_follow_artist"})),
    (re.compile(r"\b(unliked|removed from (your )?library|unfollowed)\b", re.I), _LIBRARY_REMOVE_TOOLS),
]

_ACTION_CLAIM_REPROMPT = (
    "Your last reply claimed a Spotify action (play, pause, save, follow, skip, volume, etc.) but no matching "
    "tool succeeded in this turn. Call the appropriate Spotify tool(s) now, then answer briefly with what "
    "actually happened — do not claim success without tool proof."
)

_HONEST_FALLBACK = (
    "I wasn't able to run the Spotify action that turn, so I can't confirm anything changed. "
    "Please try again or rephrase the request."
)


def tool_result_succeeded(tool_name: str, raw_result: str) -> bool:
    try:
        data = json.loads(raw_result)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    if data.get("error"):
        return False
    if tool_name in _PLAYBACK_TOOLS and data.get("ok") is False:
        return False
    if tool_name in _PAUSE_TOOLS and data.get("ok") is False:
        return False
    return True


def reply_claims_unbacked_action(text: str, successful_tools: set[str]) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    for pattern, required_any in _CLAIM_RULES:
        if not pattern.search(stripped):
            continue
        if not successful_tools.intersection(required_any):
            return True
    return False


def action_claim_reprompt() -> str:
    return _ACTION_CLAIM_REPROMPT


def action_claim_honest_fallback() -> str:
    return _HONEST_FALLBACK


def record_successful_tool(successful: set[str], tool_name: str, raw_result: str) -> None:
    if tool_result_succeeded(tool_name, raw_result):
        successful.add(tool_name)
