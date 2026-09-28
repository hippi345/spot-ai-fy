"""Pick user-owned playlists safe to play in dev-mode Spotify apps."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from spot_backend.spotify_dev_limits import SPOTIFY_DEV_MAX_PAGE, SPOTIFY_DEV_MAX_PAGINATION_PAGES

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


def fetch_owned_playlist_candidates_paginated(
    run_user_playlists: Callable[[dict[str, int]], str],
    me_id: str,
    *,
    max_pages: int = SPOTIFY_DEV_MAX_PAGINATION_PAGES,
    page_size: int = SPOTIFY_DEV_MAX_PAGE,
) -> tuple[list[dict[str, Any]], list[tuple[str, dict[str, int], str]]]:
    """Page GET /me/playlists with dev-mode limit until enough owned rows or cap."""
    owned: list[dict[str, Any]] = []
    steps: list[tuple[str, dict[str, int], str]] = []
    offset = 0
    for _ in range(max_pages):
        args = {"limit": page_size, "offset": offset}
        raw = run_user_playlists(args)
        steps.append(("spotify_user_playlists", args, raw))
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            break
        if not isinstance(data, dict):
            break
        if data.get("error") and not data.get("items"):
            break
        items = data.get("items") if isinstance(data.get("items"), list) else []
        owned.extend(owned_playlist_candidates(items, me_id))
        if owned:
            return owned, steps
        total = data.get("total")
        if not items:
            break
        offset += len(items)
        if isinstance(total, int) and offset >= total:
            break
        if not isinstance(total, int) and len(items) < page_size:
            break
    return owned, steps
