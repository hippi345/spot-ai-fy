"""When deterministic play/queue shortcuts must defer to the LLM."""

from __future__ import annotations

import re

from spot_backend.prompt_intent import (
    prompt_is_capability_question,
    prompt_is_informational,
    prompt_is_multi_step,
)

_CAN_YOU_RE = re.compile(r"\b(?:can|could|will|would)\s+you\b", re.I)

_VAGUE_PLAY_FRAGMENT_RE = re.compile(
    r"\b(?:"
    r"his|her|their|its|my|our|your|"
    r"something|anything|stuff|"
    r"one\s+of|some\s+of|"
    r"latest|newest|recent|"
    r"single|singles|"
    r"podcast|podcasts|episode|episodes|"
    r"interface|"
    r"this|that|it"
    r")\b",
    re.I,
)

_GENERIC_TRACK_TITLES = frozenset(
    {
        "something",
        "anything",
        "a song",
        "a track",
        "some music",
        "music",
        "that",
        "this",
        "it",
    }
)


def prompt_blocks_deterministic_play_shortcuts(user_text: str) -> bool:
    """True when play/track/queue shortcuts must not run (LLM handles the turn)."""
    t = (user_text or "").strip()
    if not t:
        return True
    if prompt_is_informational(t) or prompt_is_capability_question(t):
        return True
    if prompt_is_multi_step(t):
        return True
    if "?" in t:
        return True
    if _CAN_YOU_RE.search(t):
        return True
    if _VAGUE_PLAY_FRAGMENT_RE.search(t):
        return True
    return False


def play_track_title_too_vague_for_shortcut(track_title: str) -> bool:
    title = (track_title or "").strip()
    if not title:
        return True
    low = title.lower()
    if low in _GENERIC_TRACK_TITLES:
        return True
    if _VAGUE_PLAY_FRAGMENT_RE.search(title):
        return True
    return False


def play_artist_name_too_vague_for_shortcut(artist_name: str) -> bool:
    name = (artist_name or "").strip()
    if not name:
        return True
    if name.lower() in {"something", "anything", "stuff", "music"}:
        return True
    if _VAGUE_PLAY_FRAGMENT_RE.search(name):
        return True
    return False
