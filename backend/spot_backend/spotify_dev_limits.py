"""Spotify Web API page-size limits for dev-mode (non-Extended-Quota) apps."""

from __future__ import annotations

from typing import Any

# Feb 2026 migration: many list/search endpoints reject limit > 10 in dev mode.
SPOTIFY_DEV_MAX_PAGE = 10

# Search endpoints default to 5 in dev mode; hard cap 10.
SPOTIFY_SEARCH_DEFAULT_LIMIT = 5

# Max pages when scanning a user's library for owned playlists, etc.
SPOTIFY_DEV_MAX_PAGINATION_PAGES = 10


def clamp_spotify_page_limit(value: Any, default: int = SPOTIFY_DEV_MAX_PAGE) -> int:
    """Clamp a requested `limit` query param to Spotify dev-mode maximum."""
    try:
        x = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(1, min(SPOTIFY_DEV_MAX_PAGE, x))


def assert_spotify_page_limit(limit: int, *, context: str = "") -> None:
    """Test helper: raise if a limit exceeds dev-mode cap."""
    if limit > SPOTIFY_DEV_MAX_PAGE:
        msg = f"Spotify page limit {limit} exceeds {SPOTIFY_DEV_MAX_PAGE}"
        if context:
            msg = f"{msg} ({context})"
        raise AssertionError(msg)
