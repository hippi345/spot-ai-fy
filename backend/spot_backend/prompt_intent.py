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

_CAPABILITY_INTERFACE_RE = re.compile(
    r"(?:"
    r"\b(?:via|through|using|with)\s+(?:this\s+)?(?:interface|app|chat|spot-?ai-?fy|here)\b"
    r"|(?:does|can)\s+(?:this|the)\s+(?:app|interface|chat)\b"
    r"|(?:are\s+you\s+able|is\s+it\s+possible)\s+to\b"
    r"|(?:what\s+can\s+you|what\s+do\s+you)\s+(?:do|support)\b"
    r")",
    re.I,
)


def _prompt_is_question_form(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    return bool(_QUESTION_START_RE.match(t))


def prompt_is_capability_question(user_text: str) -> bool:
    """Whether the user asks what the app can do (not a command to act now)."""
    t = (user_text or "").strip()
    if not t or not _prompt_is_question_form(t):
        return False
    low = t.lower()
    if re.search(r"\bwhat\s+can\s+you\s+do\b", low):
        return True
    if (
        re.search(r"\bcan\s+(?:you|this)\b", t, re.I)
        and re.search(r"\b(?:make|create)\b", low)
        and "playlist" in low
    ):
        if re.search(r"\bplaylist\s+\S", t) and not re.search(
            r"\b(?:make|create)\s+(?:me\s+)?(?:a\s+)?playlists?\s*\??\s*$",
            t,
            re.I,
        ):
            return False
        if re.search(r"\b(?:make|create)\s+(?:me\s+)?(?:a\s+)?playlists?\s*\??\s*$", t, re.I):
            return True
        if re.search(r"\bmake\s+me\s+a\s+playlists?\s*\??\s*$", t, re.I):
            return True
    if _CAPABILITY_INTERFACE_RE.search(t):
        return True
    low = t.lower()
    if re.search(r"\bcan\s+(?:you|this)\b", t, re.I) and re.search(
        r"\b(?:podcasts?|episodes?)\b", low
    ):
        if not re.search(
            r"\bplay\s+(?:the\s+)?(?:episode|podcast)\s+[\"']?\w",
            t,
            re.I,
        ):
            return True
    return False


def _prompt_has_action_request(text: str) -> bool:
    """True when the user is asking the agent to perform a Spotify action now."""
    t = text.strip()
    if not t:
        return False
    if prompt_is_capability_question(t):
        return False
    if _LEADING_ACTION_REQUEST_RE.match(t):
        return True
    return bool(_POLITE_ACTION_REQUEST_RE.search(t))


def _prompt_is_advice_or_explanation(text: str) -> bool:
    return bool(_ADVICE_EXPLANATION_RE.search(text))


def _prompt_asks_for_user_spotify_data(text: str) -> bool:
    """Structural: user-specific library / listening questions (may use read-only tools)."""
    t = text.strip().lower()
    if not t:
        return False
    if re.search(r"\bmy\b", t):
        return True
    if re.search(r"\bwhat(?:'s|s| is)\s+playing\b", t):
        return True
    if re.search(r"\bwhat\s+did\s+i\b", t):
        return True
    if re.search(r"\bhow\s+many\b", t) and re.search(r"\b(?:i|my|me)\b", t):
        return True
    return False


def prompt_is_pure_how_to(user_text: str) -> bool:
    """App how-to / advice only — informational but should not call lookup tools."""
    if not prompt_is_informational(user_text):
        return False
    if _prompt_asks_for_user_spotify_data(user_text):
        return False
    t = user_text.strip()
    if re.search(r"\bhow\s+(?:do|can|should|would)\s+i\b", t, re.I):
        return True
    if re.search(r"\bwhere\s+(?:do|can)\s+i\b", t, re.I):
        return True
    return _prompt_is_advice_or_explanation(t)


_RECENT_LISTENING_TIME_RE = re.compile(
    r"\b(?:lately|recently|last few days|these days)\b",
    re.I,
)
_RECENT_LISTENING_ACTIVITY_RE = re.compile(
    r"\b(?:listening|played|heard|been\s+listening)\b",
    re.I,
)


def prompt_requests_recent_listening_history(user_text: str) -> bool:
    """True when the user wants recently played history (not top-tracks aggregates)."""
    t = (user_text or "").strip()
    if not t:
        return False
    low = t.lower()
    if "what have i been listening" in low:
        return True
    if not _RECENT_LISTENING_ACTIVITY_RE.search(t):
        return False
    return bool(_RECENT_LISTENING_TIME_RE.search(t))


_CURRENT_TRACK_REF_RE = re.compile(
    r"\b(?:"
    r"this\s+song|this\s+track|this\b|"
    r"current\s+song|current\s+track|"
    r"what(?:'s|\s+is)\s+playing|now\s+playing"
    r")\b",
    re.I,
)
_RELEASE_DATE_QUESTION_RE = re.compile(
    r"\b(?:when\s+did|release\s+date|come\s+out|came\s+out|what\s+year)\b",
    re.I,
)


def prompt_requests_current_track_release(user_text: str) -> bool:
    t = (user_text or "").strip()
    if not t:
        return False
    if not _RELEASE_DATE_QUESTION_RE.search(t):
        return False
    return bool(_CURRENT_TRACK_REF_RE.search(t))


def prompt_is_informational(user_text: str) -> bool:
    """Questions and advice without an action request → read-only tools only, no mutations."""
    t = (user_text or "").strip()
    if not t:
        return False
    if _prompt_has_action_request(t):
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
    user_text: str = "",
) -> list[dict[str, Any]]:
    if informational and prompt_is_capability_question(user_text):
        return []
    if not informational:
        return tools
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
    user_text: str = "",
) -> list[dict[str, Any]]:
    if informational and prompt_is_capability_question(user_text):
        return []
    if not informational:
        return declarations
    allowed = SPOTIFY_READ_ONLY_TOOL_NAMES
    return [d for d in declarations if isinstance(d, dict) and d.get("name") in allowed]


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


CAPABILITY_QUESTION_SUFFIX = """

APP CAPABILITY QUESTION:
- Answer whether the feature is supported in plain language (yes/no). Do NOT call playback, queue, or library-mutation tools on this turn.
- Podcast episodes are not supported through these Spotify Web API tools — only music (tracks, albums, artists) and the user's playlists/library controls.
"""

INFORMATIONAL_REPLY_SYSTEM_SUFFIX = """

INFORMATIONAL TURN (read-only Spotify data allowed; no mutations):
- Do NOT call tools that create, edit, play, pause, queue, save, follow, shuffle, repeat, or otherwise change Spotify state.
- You MAY use read-only lookup tools when the user asks about their library, playlists, listening history, playback, or catalog facts.
- Answer in everyday language where possible; use tools when live Spotify data is needed.
- To save a track to Liked Songs in the Spotify app: tap the heart icon, or use '+' / Add to Liked Songs. That action is Spotify's "like". Never tell the user liking is impossible or that the app lacks a heart / save control.
- Never mention internal tool names, function names, parameters, or code spans in your reply.
"""

PURE_HOW_TO_NO_LOOKUP_SUFFIX = """

PURE HOW-TO (app instructions only):
- The user only wants to know how to do something in the Spotify app or what to type here later — not a live report of their library.
- Do NOT call Spotify lookup tools on this turn; explain steps from general Spotify knowledge.
- Spotify app facts you may cite: crossfade is available under Settings → Playback (adjust crossfade duration); collaborative playlists are enabled in playlist settings (Make collaborative); deleting a playlist is done from the playlist menu (⋯) → Delete.
- When you mention song or artist names in examples, keep them generic unless they came from tool results in this chat.
"""


def informational_system_suffix(user_text: str) -> str:
    base = INFORMATIONAL_REPLY_SYSTEM_SUFFIX
    if prompt_is_capability_question(user_text):
        return base + CAPABILITY_QUESTION_SUFFIX
    if prompt_is_pure_how_to(user_text):
        return base + PURE_HOW_TO_NO_LOOKUP_SUFFIX
    return base


_VAGUE_PLAYLIST_PLAY_RE = re.compile(
    r"\bplay\s+(?:one\s+of\s+)?(?:my|a)\s+playlists?\b",
    re.I,
)

_ALTERNATE_OWNED_PLAYLIST_RE = re.compile(
    r"(?:"
    r"\b(?:actually,?\s+)?play\s+(?:a\s+)?different\s+playlist\b"
    r"|"
    r"\banother\s+playlist\b"
    r"|"
    r"\ba\s+different\s+one\b"
    r"|"
    r"\bsomething\s+else\s+from\s+my\s+playlists?\b"
    r")",
    re.I,
)

_OLLAMA_PLAYBACK_AFTER_LIST_TOOLS = frozenset(
    {
        "spotify_play_playlist",
        "spotify_start_resume_playback",
        "spotify_play_track",
        "spotify_play_artist",
        "spotify_play_artist_latest_release",
        "spotify_play_artist_popular_track",
    }
)

OLLAMA_VAGUE_PLAYLIST_PLAY_NUDGE = (
    "You listed the user's playlists but did not start playback. Pick one playlist id from "
    "the spotify_user_playlists result and call spotify_play_playlist now. Then reply in one "
    "short sentence (e.g. which playlist you started)."
)


_SURPRISE_ME_RE = re.compile(
    r"^\s*(?:surprise\s+me|play\s+something\s+random|something\s+random|pick\s+something\s+for\s+me)\s*\.?\s*$",
    re.I,
)


def prompt_is_surprise_me_request(user_text: str) -> bool:
    return bool(_SURPRISE_ME_RE.match((user_text or "").strip()))


def prompt_is_vague_playlist_play_request(user_text: str) -> bool:
    """True for requests like 'play one of my playlists' without naming a specific list."""
    t = (user_text or "").strip()
    if not t:
        return False
    if prompt_is_alternate_owned_playlist_play_request(t):
        return False
    return bool(_VAGUE_PLAYLIST_PLAY_RE.search(t))


def prompt_is_alternate_owned_playlist_play_request(user_text: str) -> bool:
    """True when the user wants another owned playlist (not the one now playing)."""
    t = (user_text or "").strip()
    if not t:
        return False
    return bool(_ALTERNATE_OWNED_PLAYLIST_RE.search(t))


def turn_needs_vague_playlist_play_nudge(turn_tool_calls: list[tuple[str, str]]) -> bool:
    """True when user_playlists ran this turn but no playback tool did."""
    names = [name for name, _ in turn_tool_calls]
    if "spotify_user_playlists" not in names:
        return False
    return not any(name in _OLLAMA_PLAYBACK_AFTER_LIST_TOOLS for name in names)


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
