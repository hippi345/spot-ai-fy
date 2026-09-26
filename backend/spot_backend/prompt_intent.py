"""Classify user prompts (informational, multi-step) and Spotify tool exposure policy."""

from __future__ import annotations

import re
from typing import Any

from spot_backend.tool_registry import agent_tool_names

# Read-only catalog / library lookups — safe for informational turns that need live data.
SPOTIFY_READ_ONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "spotify_search",
        "spotify_search_playlists",
        "spotify_me",
        "spotify_user_playlists",
        "spotify_user_public_playlists",
        "spotify_playlist_tracks",
        "spotify_get_playlist",
        "spotify_get_album",
        "spotify_get_track",
        "spotify_get_artist",
        "spotify_artist_albums",
        "spotify_artist_latest_album",
        "spotify_artist_top_tracks",
        "spotify_user_saved_tracks",
        "spotify_recently_played",
        "spotify_saved_albums",
        "spotify_followed_artists",
        "spotify_top_artists",
        "spotify_top_tracks",
        "spotify_playlists_containing_track",
        "spotify_get_queue",
        "spotify_devices",
        "spotify_playback_state",
    }
)

SPOTIFY_MUTATING_TOOL_NAMES: frozenset[str] = frozenset(
    agent_tool_names() - SPOTIFY_READ_ONLY_TOOL_NAMES
)

# How-to / advice / explanation — informational unless a leading action imperative wins.
_HOW_TO_ADVICE_RE = re.compile(
    r"(?:"
    r"\bhow\s+(?:do|can|should|would)\s+i\b"
    r"|"
    r"\bhow\s+would\s+i\b"
    r"|"
    r"\bhow\s+(?:does|do)\s+\w"
    r"|"
    r"\bwhat(?:'s|s| is)\s+the\s+(?:best\s+)?way\s+to\b"
    r"|"
    r"\bwhat\s+is\s+the\s+way\s+to\b"
    r"|"
    r"\bwhat\s+(?:does|is|are)\b"
    r"|"
    r"\bwhere\s+(?:do|can)\s+i\s+(?:find|get|see)\b"
    r"|"
    r"\bcould\s+you\s+walk\s+me\s+through\b"
    r"|"
    r"\bwalk\s+me\s+through\b"
    r"|"
    r"\bhelp\s+me\s+understand\b"
    r"|"
    r"\bany\s+tips\s+for\b"
    r"|"
    r"\bis\s+there\s+a\s+way\s+to\b"
    r"|"
    r"\bcan\s+you\s+explain\s+how\b"
    r"|"
    r"\bcan\s+you\s+explain\b"
    r"|"
    r"\bexplain\s+how\b"
    r"|"
    r"\bexplain\b"
    r"|"
    r"\btell\s+me\s+(?:how|about)\b"
    r"|"
    r"\bwhat\s+do\s+i\s+say\s+to\b"
    r"|"
    r"\bwhat\s+should\s+i\s+(?:type|say|tell)\b"
    r")",
    re.I,
)

# Live catalog / library lookups — allow tools even when phrased as questions.
_CATALOG_DATA_QUESTION_RE = re.compile(
    r"(?:"
    r"\bwhat(?:'s|s| is)\s+(?:on|playing|in)\b"
    r"|"
    r"\bwhat\s+(?:songs?|tracks?|albums?|artists?|playlists?)\s+(?:are|is)\b"
    r"|"
    r"\bwhich\s+(?:songs?|tracks?|albums?|artists?|playlists?)\b"
    r"|"
    r"\bwhat\s+did\s+i\s+(?:just\s+)?play\b"
    r")",
    re.I,
)

# Direct commands — not informational even if they mention "how" elsewhere.
_ACTION_IMPERATIVE_RE = re.compile(
    r"^\s*(?:"
    r"play|pause|resume|skip|previous|shuffle|repeat|like|save|follow|unfollow|"
    r"add|delete|remove|make|turn|queue|stop"
    r")\b",
    re.I,
)

_MULTI_STEP_RE = re.compile(
    r"\b(?:then|and\s+then|after\s+that|next,)\b|(?:,\s*){2,}.+\b(?:then|and)\b",
    re.I,
)


def prompt_is_informational(user_text: str) -> bool:
    """How-to / advice / explanation — read-only tools only; no mutations."""
    t = (user_text or "").strip()
    if not t:
        return False
    if _ACTION_IMPERATIVE_RE.match(t):
        return False
    if _CATALOG_DATA_QUESTION_RE.search(t):
        return False
    return bool(_HOW_TO_ADVICE_RE.search(t))


def prompt_is_multi_step(user_text: str) -> bool:
    """Compound requests that need several tool rounds — do not force Gemini ANY on round 1."""
    t = (user_text or "").strip().lower()
    if not t:
        return False
    if _MULTI_STEP_RE.search(t):
        return True
    verbs = (
        "add",
        "create",
        "play",
        "verify",
        "save",
        "remove",
        "shuffle",
        "repeat",
        "queue",
        "pause",
        "skip",
    )
    hits = sum(1 for v in verbs if re.search(rf"\b{v}\b", t))
    return hits >= 2 and ("," in t or " and " in t or " then " in t)


def spotify_tool_is_mutating(name: str) -> bool:
    return name in SPOTIFY_MUTATING_TOOL_NAMES


def filter_ollama_tools_for_prompt(
    tools: list[dict[str, Any]],
    *,
    informational: bool,
    allow_read_only: bool = False,
) -> list[dict[str, Any]]:
    if not informational:
        return tools
    if not allow_read_only:
        return []
    allowed = SPOTIFY_READ_ONLY_TOOL_NAMES
    out: list[dict[str, Any]] = []
    for entry in tools:
        fn = entry.get("function") if isinstance(entry, dict) else None
        if isinstance(fn, dict) and fn.get("name") in allowed:
            out.append(entry)
    return out


def gemini_declarations_for_prompt(
    declarations: list[dict[str, Any]],
    *,
    informational: bool,
) -> list[dict[str, Any]]:
    if not informational:
        return declarations
    return []


def refused_mutating_tool_result(tool_name: str) -> str:
    import json

    return json.dumps(
        {
            "error": (
                f"Refused to run mutating tool {tool_name!r} for an informational/how-to question. "
                "Answer in plain language without changing the user's Spotify library or playback."
            ),
            "informational_refusal": True,
            "reconnect_spotify_unnecessary": True,
            "sign_out_not_recommended": True,
        },
        ensure_ascii=False,
    )


INFORMATIONAL_REPLY_SYSTEM_SUFFIX = """

INFORMATIONAL / HOW-TO TURN (no Spotify mutations):
- The user is asking how something works, not asking you to do it now.
- Do NOT call tools that create, edit, play, pause, queue, save, follow, shuffle, repeat, or otherwise change Spotify state.
- Answer in everyday language: what to type in this chat, or where to tap in the Spotify desktop/mobile app.
- To save a track to Liked Songs in the Spotify app: tap the heart icon, or use '+' / Add to Liked Songs. That action is Spotify's "like". Never tell the user liking is impossible or that the app lacks a heart / save control.
- Never mention internal tool names, function names, parameters, or code spans in your reply.
"""


def gemini_should_use_any_first_round(
    user_text: str,
    *,
    had_tool_results: bool,
    intent_tools: list[str] | None,
    wants_spotify: bool,
) -> bool:
    if had_tool_results:
        return False
    if prompt_is_informational(user_text) or prompt_is_multi_step(user_text):
        return False
    if intent_tools:
        return True
    return bool(wants_spotify)
