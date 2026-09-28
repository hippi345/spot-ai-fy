"""Pick user-owned playlists safe to play in dev-mode Spotify apps."""

from __future__ import annotations

from typing import Any

_SPOTIFY_CURATED_PREFIX = "37i9dQZF1"


def playlist_id_is_spotify_curated(playlist_id: str) -> bool:
    pid = (playlist_id or "").strip()
    return pid.startswith(_SPOTIFY_CURATED_PREFIX)


def playlist_row_playable_owned(row: dict[str, Any], me_id: str) -> bool:
    if not isinstance(row, dict):
        return False
    pid = row.get("id")
    if not isinstance(pid, str) or not pid.strip():
        return False
    if playlist_id_is_spotify_curated(pid):
        return False
    owner_id = row.get("owner_id")
    if not isinstance(owner_id, str) or not owner_id.strip():
        owner = row.get("owner")
        if isinstance(owner, dict):
            oid = owner.get("id")
            owner_id = oid if isinstance(oid, str) else ""
    if (owner_id or "").strip().lower() == "spotify":
        return False
    me = (me_id or "").strip()
    if not me:
        return False
    return owner_id == me


def owned_playlist_candidates(
    items: list[Any],
    me_id: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in items:
        if isinstance(row, dict) and playlist_row_playable_owned(row, me_id):
            out.append(row)
    return out
