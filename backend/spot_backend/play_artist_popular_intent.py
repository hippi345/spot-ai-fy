"""Detect requests to play an artist's most popular / best-known track."""

from __future__ import annotations

import re

_POPULAR_TRACK_RE = re.compile(
    r"(?:"
    r"(?:play\s+)?(?:their|his|her)\s+(?:most\s+popular|biggest\s+hit|best[- ]?known)\s+(?:song|track|single)\b"
    r"|"
    r"\b(?:most\s+popular|biggest\s+hit|best[- ]?known)\s+(?:song|track|single)\s+by\s+"
    r"|"
    r"\bplay\s+(?:the\s+)?(?:most\s+popular|biggest\s+hit|best[- ]?known)\s+(?:song|track|single)\s+by\s+"
    r"|"
    r"\bplay\s+.+\s+(?:most\s+popular|biggest\s+hit|best[- ]?known)\s+(?:song|track|single)\b"
    r")",
    re.I,
)


def prompt_requests_play_artist_popular_track(user_text: str) -> bool:
    t = (user_text or "").strip()
    if not t:
        return False
    return bool(_POPULAR_TRACK_RE.search(t))
