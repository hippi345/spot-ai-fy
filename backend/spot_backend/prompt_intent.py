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

# Longer tokens first so e.g. "unlike" wins over "like".
_ACTION_VERBS: tuple[str, ...] = (
    "unfollow",
    "unlike",
    "previous",
    "transfer",
    "shuffle",
    "resume",
    "repeat",
    "rename",
    "remove",
    "create",
    "delete",
    "follow",
    "switch",
    "queue",
    "start",
    "pause",
    "play",
    "save",
    "like",
    "skip",
    "next",
    "make",
    "turn",
    "undo",
    "stop",
    "add",
    "set",
    "put",
)

_ACTION_VERB_ALT = "|".join(_ACTION_VERBS)

_LEADING_ACTION_REQUEST_RE = re.compile(
    rf"^\s*(?:please\s+|hey\s+|ok(?:ay)?\s+)?(?:{_ACTION_VERB_ALT})\b",
    re.I,
)

_POLITE_ACTION_REQUEST_RE = re.compile(
    rf"(?:"
    rf"(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:{_ACTION_VERB_ALT})\b"
    rf"|"
    rf"^\s*please\s+(?:{_ACTION_VERB_ALT})\b"
    rf"|"
    rf"^\s*i\s+(?:want|would\s+like|'d\s+like)\s+(?:you\s+to\s+)?(?:{_ACTION_VERB_ALT})\b"
    rf"|"
    rf"^\s*let'?s\s+(?:{_ACTION_VERB_ALT})\b"
    rf")",
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

_QUESTION_START_RE = re.compile(
    r"^\s*(?:what|where|when|why|how|which|who|can|could|should|would|is|are|do|does|did|any)\b",
    re.I,
)

_ADVICE_EXPLANATION_RE = re.compile(
    r"(?:"
    r"\bexplain\b"
    r"|"
    r"\bhelp\s+me\s+understand\b"
    r"|"
    r"\btips\b"
    r"|"
    r"\bwalk\s+me\s+through\b"
    r"|"
    r"\btell\s+me\s+(?:how|about)\b"
    r"|"
    r"\bbest\s+way\b"
    r")",
    re.I,
)

_MULTI_STEP_RE = re.compile(
    r"\b(?:then|and\s+then|after\s+that|next,)\b|(?:,\s*){2,}.+\b(?:then|and)\b",
    re.I,
)


def _prompt_has_action_request(text: str) -> bool:
    """True when the user is asking the agent to perform a Spotify action now."""
    t = text.strip()
    if not t:
        return False
    if _LEADING_ACTION_REQUEST_RE.match(t):
        return True
    return bool(_POLITE_ACTION_REQUEST_RE.search(t))


def _prompt_is_question_form(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    return bool(_QUESTION_START_RE.match(t))


def _prompt_is_advice_or_explanation(text: str) -> bool:
    return bool(_ADVICE_EXPLANATION_RE.search(text))


def prompt_is_informational(user_text: str) -> bool:
    """Questions and advice without an action request → read-only / no mutations."""
    t = (user_text or "").strip()
    if not t:
        return False
    if _prompt_has_action_request(t):
        return False
    if _CATALOG_DATA_QUESTION_RE.search(t):
        return False
    return _prompt_is_question_form(t) or _prompt_is_advice_or_explanation(t)


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
