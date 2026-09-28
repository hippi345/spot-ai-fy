"""Detect assistant replies that claim Spotify actions without a matching successful tool call."""

from __future__ import annotations

import json
import re

from spot_backend.playback_reply import prompt_asks_whats_playing

_PLAYBACK_STATE_TOOLS = frozenset({"spotify_playback_state"})
_PLAYLIST_LIST_TOOLS = frozenset({"spotify_user_playlists"})
_PLAYBACK_TOOLS = frozenset(
    {
        "spotify_start_resume_playback",
        "spotify_play_playlist",
        "spotify_play_artist",
        "spotify_play_track",
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

# Only match first-person or direct assistant action claims — not how-to questions or
# informational replies that mention "liked" in passing.
_CLAIM_RULES: list[tuple[re.Pattern[str], frozenset[str]]] = [
    (re.compile(r"\b(?:I(?:'ve| have)?\s+)?now playing\b", re.I), _PLAYBACK_TOOLS),
    (re.compile(r"\bnow playing\b", re.I), _PLAYBACK_TOOLS),
    (re.compile(r"\bplaying\b.+\b(?:top tracks|radio)\b", re.I), _PLAYBACK_TOOLS),
    (re.compile(r"^playing\b", re.I), _PLAYBACK_TOOLS),
    (re.compile(r"\bplaying\s+['\"]", re.I), _PLAYBACK_TOOLS),
    (re.compile(r"\b(?:I(?:'m| am)\s+)playing\s+[^.?!\n]{2,}", re.I), _PLAYBACK_TOOLS),
    (re.compile(r"\b(?:I(?:'ve| have)?\s+)?(?:started|starting) (?:playing|playback)\b", re.I), _PLAYBACK_TOOLS),
    (re.compile(r"\b(?:I(?:'m| am)\s+)?playing\b.+\b(?:on|in) your\b", re.I), _PLAYBACK_TOOLS),
    (re.compile(r"\b(?:I(?:'ve| have)?\s+)?paused(?: playback)?\b", re.I), _PAUSE_TOOLS),
    (re.compile(r"^skipped\b", re.I), _SKIP_TOOLS),
    (re.compile(r"\b(?:I(?:'ve| have)?\s+)?(?:skipped|skipping)\b", re.I), _SKIP_TOOLS),
    (re.compile(r"\bskipped\b.+\b(?:song|track)\b", re.I), _SKIP_TOOLS),
    (re.compile(r"\b(?:I(?:'ve| have)?\s+)?set (?:the )?volume\b", re.I), _VOLUME_TOOLS),
    (re.compile(r"\bvolume (?:is )?now (?:at|set to)\b", re.I), _VOLUME_TOOLS),
    (re.compile(r"^saved\b", re.I), _LIBRARY_SAVE_TOOLS),
    (re.compile(r"\b(?:I(?:'ve| have)?\s+)?saved(?:\s+(?:that|the|this|it))?\b", re.I), _LIBRARY_SAVE_TOOLS),
    (re.compile(r"\b(?:I(?:'ve| have)?\s+)?saved\s+['\"]", re.I), _LIBRARY_SAVE_TOOLS),
    (re.compile(r"\bsaved\b.+\bto your (?:liked|library)\b", re.I), _LIBRARY_SAVE_TOOLS),
    (re.compile(r"^liked\b", re.I), _LIBRARY_SAVE_TOOLS),
    (re.compile(r"\b(?:I(?:'ve| have)?\s+)?liked (?:that|the|this|it)\b", re.I), _LIBRARY_SAVE_TOOLS),
    (re.compile(r"\b(?:I(?:'ve| have)?\s+)?liked\s+['\"]", re.I), _LIBRARY_SAVE_TOOLS),
    (re.compile(r"^added\b", re.I), _LIBRARY_SAVE_TOOLS),
    (re.compile(r"\b(?:I(?:'ve| have)?\s+)?added\b.+\bto (?:your )?(?:library|liked|playlist)\b", re.I), _LIBRARY_SAVE_TOOLS),
    (
        re.compile(r"\b(?:I(?:'ve| have)?\s+)?added (?:it )?to (?:your )?library\b", re.I),
        _LIBRARY_SAVE_TOOLS,
    ),
    (re.compile(r"\b(?:I(?:'ve| have)?\s+)?followed\b", re.I), _LIBRARY_SAVE_TOOLS),
    (re.compile(r"\b(?:I(?:'m| am) )?now following\b", re.I), _LIBRARY_SAVE_TOOLS),
    (
        re.compile(
            r"\b(?:I(?:'ve| have)?\s+)?(?:removed|unliked|unfollowed)\b",
            re.I,
        ),
        _LIBRARY_REMOVE_TOOLS,
    ),
    (
        re.compile(r"\bremoved .+ from (?:your )?(?:library|liked songs|saved)\b", re.I),
        _LIBRARY_REMOVE_TOOLS,
    ),
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

_TOOL_SUMMARIZE_REPROMPT_PREFIX = (
    "The Spotify tool call(s) for this turn already succeeded. Reply with one or two short sentences "
    "that summarize the tool result JSON for the user. Do not say you failed, could not run tools, "
    "or ask them to rephrase. Base your answer only on the tool output below.\n\n"
)


def is_failure_boilerplate(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped == _HONEST_FALLBACK.strip():
        return True
    low = stripped.lower()
    return "wasn't able to run the spotify action" in low or "can't confirm anything changed" in low


def tool_summarize_reprompt(tool_results: list[str], *, max_chars: int = 6000) -> str:
    chunks: list[str] = []
    used = 0
    for raw in tool_results:
        piece = raw if isinstance(raw, str) else str(raw)
        if used + len(piece) > max_chars:
            piece = piece[: max(0, max_chars - used)] + "…"
        chunks.append(piece)
        used += len(piece)
        if used >= max_chars:
            break
    body = "\n\n---\n\n".join(chunks) if chunks else "(empty tool results)"
    return _TOOL_SUMMARIZE_REPROMPT_PREFIX + body


def turn_tool_calls_all_succeeded(calls: list[tuple[str, str]]) -> bool:
    if not calls:
        return False
    return all(tool_result_succeeded(name, raw) for name, raw in calls)


def _prompt_requests_user_playlists(user_text: str) -> bool:
    low = (user_text or "").strip().lower()
    if not low:
        return False
    return "playlist" in low and any(w in low for w in ("my", "what are", "list", "show"))


def _playback_state_backs_reply(user_text: str, successful_tools: set[str], reply: str) -> bool:
    if not successful_tools.intersection(_PLAYBACK_STATE_TOOLS):
        return False
    if prompt_asks_whats_playing(user_text):
        return True
    low = reply.lower()
    return any(
        phrase in low
        for phrase in (
            "listening to",
            "nothing is playing",
            "not playing",
            "no track",
            "currently playing",
            "right now",
        )
    )


def _playlist_list_backs_reply(user_text: str, successful_tools: set[str]) -> bool:
    if not successful_tools.intersection(_PLAYLIST_LIST_TOOLS):
        return False
    return _prompt_requests_user_playlists(user_text)


def _reply_claims_playback_started(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    for pattern, required_any in _CLAIM_RULES:
        if required_any is not _PLAYBACK_TOOLS:
            continue
        if pattern.search(stripped):
            return True
    return False


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


def reply_claims_unbacked_action(
    text: str,
    successful_tools: set[str],
    *,
    user_text: str = "",
    turn_tool_calls: list[tuple[str, str]] | None = None,
) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if prompt_asks_whats_playing(user_text) and successful_tools.intersection(_PLAYBACK_STATE_TOOLS):
        return False
    if (
        _prompt_requests_user_playlists(user_text)
        and successful_tools.intersection(_PLAYLIST_LIST_TOOLS)
        and not _reply_claims_playback_started(stripped)
    ):
        return False
    for pattern, required_any in _CLAIM_RULES:
        if not pattern.search(stripped):
            continue
        if successful_tools.intersection(required_any):
            continue
        if turn_tool_calls:
            attempted = [name for name, raw in turn_tool_calls if name in required_any]
            if attempted and all(not tool_result_succeeded(name, raw) for name, raw in turn_tool_calls if name in required_any):
                return True
        if required_any is _PLAYBACK_TOOLS and _playback_state_backs_reply(
            user_text, successful_tools, stripped
        ):
            continue
        if required_any is _LIBRARY_SAVE_TOOLS and _playlist_list_backs_reply(
            user_text, successful_tools
        ):
            continue
        return True
    return False


def action_claim_reprompt() -> str:
    return _ACTION_CLAIM_REPROMPT


def action_claim_honest_fallback() -> str:
    return _HONEST_FALLBACK


def record_successful_tool(successful: set[str], tool_name: str, raw_result: str) -> None:
    if tool_result_succeeded(tool_name, raw_result):
        successful.add(tool_name)
