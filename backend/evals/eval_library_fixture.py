"""SAMPLE_DATA — shared stubbed Spotify library for evals, local dev seed, and live sunset cases."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "eval_library.json"
EMBEDDINGS_PATH = Path(__file__).with_name("fixtures") / "eval_library_embeddings.npz"
PRIMARY_USER = "eval_user_a"

# Back-compat path name
LEGACY_PATH = Path(__file__).with_name("fixtures") / "canonical_library.json"


@lru_cache(maxsize=1)
def load_fixture() -> dict[str, Any]:
    path = FIXTURE_PATH if FIXTURE_PATH.is_file() else LEGACY_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("tracks"), list):
        raise ValueError(f"Invalid eval library fixture: {path}")
    return data


def tracks(user_id: str = PRIMARY_USER) -> list[dict[str, Any]]:
    rows = load_fixture()["tracks"]
    return [t for t in rows if (t.get("owner_id") or PRIMARY_USER) == user_id]


def decoy_tracks_user_b() -> list[dict[str, Any]]:
    dec = load_fixture().get("decoy_tracks_user_b")
    return list(dec) if isinstance(dec, list) else []


def planted_sunset_ids() -> set[str]:
    meta = load_fixture().get("meta") or {}
    planted = (meta.get("planted_sets") or {}).get("sunset_loved") or []
    return {str(x) for x in planted}


def planted_sunset_uris() -> set[str]:
    ids = planted_sunset_ids()
    return {f"spotify:track:{tid}" for tid in ids}


def all_track_uris(user_id: str = PRIMARY_USER) -> set[str]:
    return {str(t["uri"]) for t in tracks(user_id) if t.get("uri")}


def top_loved_uris(fraction: float = 0.25, user_id: str = PRIMARY_USER) -> set[str]:
    ranked = sorted(tracks(user_id), key=lambda t: int(t.get("play_count") or 0), reverse=True)
    n = max(1, int(len(ranked) * fraction))
    return {str(t["uri"]) for t in ranked[:n]}


def loved_pick_share(acted_uris: list[str], *, fraction: float = 0.25) -> float:
    if not acted_uris:
        return 0.0
    top = top_loved_uris(fraction)
    return sum(1 for u in acted_uris if u in top) / len(acted_uris)


def _track_to_spotify_item(t: dict[str, Any]) -> dict[str, Any]:
    return {
        "added_at": t.get("liked_at"),
        "track": {
            "id": t["id"],
            "uri": t["uri"],
            "name": t["name"],
            "artists": t.get("artists") or [],
            "album": t.get("album") or {},
            "popularity": min(100, int(t.get("play_count", 1)) // 5),
            "eval_play_count": t.get("eval_play_count", t.get("play_count")),
            "eval_liked_at": t.get("eval_liked_at", t.get("liked_at")),
            "eval_moods": t.get("eval_moods") or [],
            "eval_tempo": t.get("eval_tempo") or "",
            "eval_tags": t.get("eval_tags") or [],
            "eval_genre": t.get("eval_genre") or t.get("genre") or "",
        },
    }


def spotify_saved_tracks_page(offset: int, limit: int, user_id: str = PRIMARY_USER) -> dict[str, Any]:
    rows = tracks(user_id)
    total = len(rows)
    chunk = rows[offset : offset + limit]
    return {
        "items": [_track_to_spotify_item(t) for t in chunk],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def spotify_top_tracks(limit: int = 50, user_id: str = PRIMARY_USER) -> dict[str, Any]:
    ranked = sorted(tracks(user_id), key=lambda t: int(t.get("play_count") or 0), reverse=True)[:limit]
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
