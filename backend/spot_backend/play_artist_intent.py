"""Detect natural-language 'play <artist>' requests."""

from __future__ import annotations

from spot_backend.play_bare_intent import (
    extract_bare_play_target,
    extract_play_music_by_artist,
)


def extract_play_artist_name(user_text: str) -> str | None:
    """Return artist name for 'play Radiohead' or 'play songs by X' — not bare track titles."""
    by_artist = extract_play_music_by_artist(user_text)
    if by_artist:
        return by_artist
    target = extract_bare_play_target(user_text)
    if not target:
        return None
    return target
