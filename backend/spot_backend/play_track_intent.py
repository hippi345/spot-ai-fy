"""Detect natural-language 'play <track> by <artist>' requests."""

from __future__ import annotations

import re

_PLAY_TRACK_BY_RE = re.compile(
    r"^\s*(?:(?:can|could|will|would)\s+you\s+)?(?:please\s+)?play\s+(.+?)\s+by\s+(.+?)\s*[.!?]*\s*$",
    re.I,
)

_EXCLUDE_PLAY_TRACK_RE = re.compile(
    r"\b(?:playlist|album|episode|podcast)\b",
    re.I,
)


def extract_play_track_request(user_text: str) -> tuple[str, str] | None:
    """Return (track_title, artist_name) for 'play Blinding Lights by The Weeknd' or None."""
    t = (user_text or "").strip()
    if not t:
        return None
    m = _PLAY_TRACK_BY_RE.match(t)
    if not m:
        return None
    title = m.group(1).strip()
    artist = m.group(2).strip()
    if not title or not artist:
        return None
    if _EXCLUDE_PLAY_TRACK_RE.search(title):
        return None
    if re.match(r"^(?:some\s+)?music\b", title, re.I):
        return None
    return title, artist
