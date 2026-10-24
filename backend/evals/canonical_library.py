"""Load canonical eval library fixture and scoring helpers."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "canonical_library.json"


@lru_cache(maxsize=1)
def load_fixture() -> dict[str, Any]:
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("tracks"), list):
        raise ValueError(f"Invalid canonical library fixture: {FIXTURE_PATH}")
    return data


def tracks() -> list[dict[str, Any]]:
    return list(load_fixture()["tracks"])


def track_by_id(track_id: str) -> dict[str, Any] | None:
    for t in tracks():
        if t.get("id") == track_id:
            return t
    return None


def all_track_uris() -> set[str]:
    return {str(t["uri"]) for t in tracks() if t.get("uri")}


def all_track_ids() -> set[str]:
    return {str(t["id"]) for t in tracks() if t.get("id")}


def top_loved_uris(fraction: float = 0.25) -> set[str]:
    """URIs in the top fraction by play_count (loved / heavy rotation)."""
    ranked = sorted(tracks(), key=lambda t: int(t.get("play_count") or 0), reverse=True)
    n = max(1, int(len(ranked) * fraction))
    return {str(t["uri"]) for t in ranked[:n]}


def loved_pick_share(acted_uris: list[str], *, fraction: float = 0.25) -> float:
    if not acted_uris:
        return 0.0
    top = top_loved_uris(fraction)
    hits = sum(1 for u in acted_uris if u in top)
    return hits / len(acted_uris)


def spotify_saved_tracks_page(offset: int, limit: int) -> dict[str, Any]:
    rows = tracks()
    total = len(rows)
    chunk = rows[offset : offset + limit]
    items = []
    for t in chunk:
        items.append(
            {
                "added_at": t.get("liked_at"),
                "track": {
                    "id": t["id"],
                    "uri": t["uri"],
                    "name": t["name"],
                    "artists": t.get("artists") or [],
                    "album": t.get("album") or {},
                    "popularity": min(100, int(t.get("play_count", 1)) // 5),
                },
            }
        )
    return {"items": items, "total": total, "limit": limit, "offset": offset}


def spotify_top_tracks(limit: int = 50) -> dict[str, Any]:
    ranked = sorted(tracks(), key=lambda t: int(t.get("play_count") or 0), reverse=True)[:limit]
    return {
        "items": [
            {
                "id": t["id"],
                "uri": t["uri"],
                "name": t["name"],
                "artists": t.get("artists") or [],
                "album": t.get("album") or {},
            }
            for t in ranked
        ],
        "total": len(ranked),
    }
