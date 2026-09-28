"""Detect bare 'play <X>' and 'play songs by <artist>' phrasing."""

from __future__ import annotations

import re

from spot_backend.chat_shortcut_policy import (
    play_artist_name_too_vague_for_shortcut,
    play_track_title_too_vague_for_shortcut,
)

_PLAY_BARE_RE = re.compile(
    r"^\s*(?:please\s+)?play\s+(.+?)\s*[.!?]*\s*$",
    re.I,
)

_EXCLUDE_PLAY_TARGET_RE = re.compile(
    r"\b(?:playlists?|albums?|episodes?|podcasts?|music|my\s+\w+\s+playlist)\b",
    re.I,
)

# "play songs by X" — generic leading word, not a track title.
_PLAY_MUSIC_BY_ARTIST_RE = re.compile(
    r"^\s*(?:please\s+)?play\s+(?:songs?|tracks?|music|something)\s+by\s+(.+?)\s*[.!?]*\s*$",
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


def extract_play_music_by_artist(user_text: str) -> str | None:
    """Return artist for 'play songs/music/tracks/something by Radiohead'."""
    t = (user_text or "").strip()
    if not t:
        return None
    m = _PLAY_MUSIC_BY_ARTIST_RE.match(t)
    if not m:
        return None
    artist = m.group(1).strip()
    if not artist or play_artist_name_too_vague_for_shortcut(artist):
        return None
    return artist


def extract_bare_play_target(user_text: str) -> str | None:
    """Return X for 'play X' when X is not 'X by Y' and not excluded."""
    t = (user_text or "").strip()
    if not t:
        return None
    by_artist = extract_play_music_by_artist(t)
    if by_artist:
        return None
    m = _PLAY_BARE_RE.match(t)
    if not m:
        return None
    name = m.group(1).strip()
    if not name or _EXCLUDE_PLAY_TARGET_RE.search(name):
        return None
    if re.search(r"\s+by\s+", name, re.I):
        return None
    if re.search(
        r"\b(?:most\s+popular|biggest\s+hit|best[- ]?known)\s+(?:song|track|single)\b",
        name,
        re.I,
    ):
        return None
    if re.match(r"^artist\s+", name, re.I):
        return None
    if name.lower() in _GENERIC_PLAY_TARGETS:
        return None
    if play_artist_name_too_vague_for_shortcut(name):
        return None
    if play_track_title_too_vague_for_shortcut(name):
        return None
    return name
