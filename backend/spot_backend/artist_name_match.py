"""Normalize and compare artist names for play-by-artist requests."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# Tune with tests: 'the weeknd' vs 'The Weeknd', 'beyonce' vs 'Beyoncé' match;
# 'Zxqvbrt Plonk' vs 'ZXVREN' does not.
ARTIST_NAME_MATCH_MIN_RATIO = 0.85


def normalize_artist_name_for_match(name: str) -> str:
    """Casefold, strip accents/punctuation, drop a leading 'the'."""
    s = (name or "").strip().casefold()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    if s.startswith("the "):
        s = s[4:].strip()
    return s


def artist_name_similarity(requested: str, candidate: str) -> float:
    a = normalize_artist_name_for_match(requested)
    b = normalize_artist_name_for_match(candidate)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def artist_names_match(
    requested: str,
    candidate: str,
    *,
    min_ratio: float = ARTIST_NAME_MATCH_MIN_RATIO,
) -> bool:
    return artist_name_similarity(requested, candidate) >= min_ratio
