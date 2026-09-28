"""Pick user-owned playlists safe to play in dev-mode Spotify apps."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from spot_backend.spotify_dev_limits import SPOTIFY_DEV_MAX_PAGE, SPOTIFY_DEV_MAX_PAGINATION_PAGES

_SPOTIFY_EDITORIAL_PREFIX = "37i9"


def playlist_id_is_spotify_curated(playlist_id: str) -> bool:
    """True for Spotify-owned editorial playlist ids (inaccessible in dev-mode apps)."""
    pid = (playlist_id or "").strip()
    return pid.startswith(_SPOTIFY_EDITORIAL_PREFIX)


def playlist_row_track_total(row: dict[str, Any]) -> int | None:
    """Track count from /me/playlists row; None when the API omitted counts."""
    direct = row.get("tracks_total")
    if isinstance(direct, int) and direct >= 0:
        return direct
    tracks = row.get("tracks")
    if isinstance(tracks, dict):
        total = tracks.get("total")
        if isinstance(total, int) and total >= 0:
            return total
    items = row.get("items")
    if isinstance(items, dict):
        total = items.get("total")
        if isinstance(total, int) and total >= 0:
            return total
    return None


def playlist_row_track_total_or_zero(row: dict[str, Any]) -> int:
    total = playlist_row_track_total(row)
    return total if isinstance(total, int) else 0


def _probe_playlist_has_tracks(
    playlist_id: str,
    probe_cache: dict[str, bool],
    run_probe: Callable[[str], str],
) -> bool:
    """GET /playlists/{id}/items?limit=1 — cached per request."""
    pid = (playlist_id or "").strip()
    if not pid:
        return False
    if pid in probe_cache:
        return probe_cache[pid]
    raw = run_probe(pid)
    has_tracks = False
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        probe_cache[pid] = False
        return False
    if isinstance(data, dict):
        if data.get("error"):
            probe_cache[pid] = False
            return False
        total = data.get("total")
        if isinstance(total, int):
            has_tracks = total > 0
        else:
            items = data.get("items")
            has_tracks = isinstance(items, list) and any(
                isinstance(row, dict) for row in items
            )
    probe_cache[pid] = has_tracks
    return has_tracks


def playlist_row_has_tracks(
    row: dict[str, Any],
    *,
    probe_cache: dict[str, bool] | None = None,
    run_probe: Callable[[str], str] | None = None,
) -> bool:
    total = playlist_row_track_total(row)
    if total is not None:
        return total > 0
    pid = row.get("id")
    if (
        isinstance(pid, str)
        and pid.strip()
        and probe_cache is not None
        and run_probe is not None
    ):
        return _probe_playlist_has_tracks(pid, probe_cache, run_probe)
    return False


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
    *,
    exclude_ids: set[str] | frozenset[str] | None = None,
    probe_cache: dict[str, bool] | None = None,
    run_probe: Callable[[str], str] | None = None,
) -> list[dict[str, Any]]:
    skip = {x.strip() for x in (exclude_ids or frozenset()) if str(x).strip()}
    out: list[dict[str, Any]] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        pid = row.get("id")
        if isinstance(pid, str) and pid.strip() in skip:
            continue
        if (
            playlist_row_playable_owned(row, me_id)
            and playlist_row_has_tracks(row, probe_cache=probe_cache, run_probe=run_probe)
        ):
            out.append(row)
    return out


def fetch_owned_playlist_candidates_paginated(
    run_user_playlists: Callable[[dict[str, int]], str],
    me_id: str,
    *,
    max_pages: int = SPOTIFY_DEV_MAX_PAGINATION_PAGES,
    page_size: int = SPOTIFY_DEV_MAX_PAGE,
    exclude_ids: set[str] | frozenset[str] | None = None,
    probe_cache: dict[str, bool] | None = None,
    run_probe: Callable[[str], str] | None = None,
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
        owned.extend(
            owned_playlist_candidates(
                items,
                me_id,
                exclude_ids=exclude_ids,
                probe_cache=probe_cache,
                run_probe=run_probe,
            )
        )
        if owned:
            return owned, steps
        # Keep paging when the only owned rows on this page were empty playlists.
        total = data.get("total")
        if not items:
            break
        offset += len(items)
        if isinstance(total, int) and offset >= total:
            break
        if not isinstance(total, int) and len(items) < page_size:
            break
    return owned, steps
