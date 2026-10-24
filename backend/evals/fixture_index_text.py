"""Shared library-index document text for eval fixture tracks (generator + runtime index)."""

from __future__ import annotations

from typing import Any


def library_index_document_text(track: dict[str, Any], *, added_at: str | None = None) -> str:
    artists = ", ".join(
        a.get("name", "") for a in (track.get("artists") or []) if isinstance(a, dict)
    )
    album = track.get("album") if isinstance(track.get("album"), dict) else {}
    year = (album.get("release_date") or "")[:4]
    moods = track.get("eval_moods") or track.get("moods") or []
    tempo = track.get("eval_tempo") or track.get("tempo") or ""
    tags = track.get("eval_tags") or track.get("tags") or []
    genre = track.get("eval_genre") or track.get("genre") or ""
    play_count = track.get("eval_play_count", track.get("play_count"))
    liked = added_at or track.get("eval_liked_at") or track.get("liked_at") or ""
    parts = [
        f"{track.get('name', '')} — {artists}",
        f"album {album.get('name', '')}",
        year,
    ]
    if genre:
        parts.append(f"genre {genre}")
    if moods:
        parts.append("moods " + ", ".join(str(m) for m in moods))
    if tempo:
        parts.append(f"tempo {tempo}")
    if tags:
        parts.append("tags " + ", ".join(str(t) for t in tags))
    if play_count is not None:
        parts.append(f"loved plays {play_count}")
    if liked:
        parts.append(f"liked {liked[:10]}")
    return " · ".join(p for p in parts if p)
