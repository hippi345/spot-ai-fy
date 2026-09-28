"""Normalize and compare artist names for play-by-artist requests."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# Tune with tests: 'the weeknd' vs 'The Weeknd', 'beyonce' vs 'Beyoncé' match;
# 'Zxqvbrt Plonk' vs 'ZXVREN' does not.
ARTIST_NAME_MATCH_MIN_RATIO = 0.85

# Mononym queries that may resolve to a different Spotify display name (e.g. Ye / Kanye West).
_ARTIST_MONONYM_ALIASES: dict[str, frozenset[str]] = {
    "ye": frozenset({"ye", "kanye west"}),
}

BARE_ARTIST_SHORT_QUERY_MAX_LEN = 3


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


def artist_query_matches_candidate_name(query: str, candidate_name: str) -> bool:
    """True when query matches candidate by fuzzy name or known mononym alias."""
    if artist_names_match(query, candidate_name):
        return True
    qn = normalize_artist_name_for_match(query)
    cn = normalize_artist_name_for_match(candidate_name)
    if not qn or not cn:
        return False
    alias_group = _ARTIST_MONONYM_ALIASES.get(qn)
    if alias_group and cn in alias_group:
        return True
    return False


def resolve_artist_from_search_items(
    query: str,
    items: list[dict[str, object]],
    *,
    allow_short_query_top_popular: bool = True,
) -> tuple[str, str, int, str] | None:
    """Pick an artist row from Spotify search items.

    Returns (id, name, popularity, match_kind) where match_kind is one of:
    exact_name, alias, short_query_top.
    """
    qn = normalize_artist_name_for_match(query)
    if not qn:
        return None
    candidates: list[tuple[str, str, int, str]] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        aid = row.get("id")
        name = row.get("name")
        if not isinstance(aid, str) or not aid.strip():
            continue
        if not isinstance(name, str) or not name.strip():
            continue
        pop = int(row.get("popularity") or 0)
        cn = normalize_artist_name_for_match(name)
        if cn == qn:
            candidates.append((aid.strip(), name.strip(), pop, "exact_name"))
        elif artist_query_matches_candidate_name(query, name):
            candidates.append((aid.strip(), name.strip(), pop, "alias"))
    if candidates:
        exact = [c for c in candidates if c[3] == "exact_name"]
        if exact:
            best = max(exact, key=lambda c: c[2])
            return best
        alias = [c for c in candidates if c[3] == "alias"]
        if alias:
            best = max(alias, key=lambda c: c[2])
            return best
    if not allow_short_query_top_popular or len(qn) > BARE_ARTIST_SHORT_QUERY_MAX_LEN:
        return None
    scored: list[tuple[str, str, int, str]] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        aid = row.get("id")
        name = row.get("name")
        if not isinstance(aid, str) or not aid.strip():
            continue
        if not isinstance(name, str) or not name.strip():
            continue
        pop = int(row.get("popularity") or 0)
        scored.append((aid.strip(), name.strip(), pop, "short_query_top"))
    if not scored:
        return None
    return max(scored, key=lambda c: c[2])
