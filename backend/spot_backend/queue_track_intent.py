"""Detect explicit 'queue <track> by <artist>' requests."""

from __future__ import annotations

import re

_QUEUE_RE = re.compile(
    r"^\s*(?:(?:can|could|will|would)\s+you\s+)?(?:please\s+)?queue\s+(.+?)\s*[.!?]*\s*$",
    re.I,
)
_BY_ARTIST_RE = re.compile(r"^(.+?)\s+by\s+(.+)$", re.I)


def extract_queue_track_request(user_text: str) -> tuple[str, str] | None:
    """Return (track_title, artist_name) for explicit queue phrasing, or None."""
    t = (user_text or "").strip()
    if not t:
        return None
    m = _QUEUE_RE.match(t)
    if not m:
        return None
    tail = m.group(1).strip()
    if not tail:
        return None
    by = _BY_ARTIST_RE.match(tail)
    if by:
        track = by.group(1).strip()
        artist = by.group(2).strip()
        if track and artist:
            return track, artist
    return tail, ""
