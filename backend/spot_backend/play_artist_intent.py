"""Detect natural-language 'play <artist>' requests."""

from __future__ import annotations

import re

from spot_backend.chat_shortcut_policy import play_artist_name_too_vague_for_shortcut

_PLAY_ARTIST_RE = re.compile(
    r"^\s*(?:please\s+)?play\s+(.+?)\s*[.!?]*\s*$",
    re.I,
)

_EXCLUDE_PLAY_TARGET_RE = re.compile(
    r"\b(?:playlists?|albums?|songs?|tracks?|episodes?|podcasts?|music|my\s+\w+\s+playlist)\b",
    re.I,
)

_GENERIC_PLAY_TARGETS = frozenset(
    {
        "something",
        "anything",
        "that",
        "this",
        "it",
        "stuff",
        "some music",
        "a song",
        "a track",
    }
)


def extract_play_artist_name(user_text: str) -> str | None:
    """Return artist name for 'play Radiohead' / 'can you play Radiohead?' or None."""
    t = (user_text or "").strip()
    if not t:
        return None
    m = _PLAY_ARTIST_RE.match(t)
    if not m:
        return None
    name = m.group(1).strip()
    if not name or _EXCLUDE_PLAY_TARGET_RE.search(name):
        return None
    if re.search(
        r"\b(?:most\s+popular|biggest\s+hit|best[- ]?known)\s+(?:song|track|single)\b",
        name,
        re.I,
    ):
        return None
    if re.search(r"\s+by\s+", name, re.I):
        return None
    if re.match(r"^artist\s+", name, re.I):
        return None
    if name.lower() in _GENERIC_PLAY_TARGETS:
        return None
    if play_artist_name_too_vague_for_shortcut(name):
        return None
    return name
