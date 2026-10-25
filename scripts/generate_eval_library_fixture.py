#!/usr/bin/env python3
"""Deterministic eval library (3k–5k tracks) + planted sunset set + committed embeddings."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from evals.fixture_index_text import library_index_document_text
from spot_backend.ollama_embed import DOC_PREFIX, HashingEmbedder

SEED = 0xC0FFEE17
TRACK_COUNT = 3600
PLANTED_SUNSET = 50
DECOY_USER_B = 120
OUT_JSON = ROOT / "backend" / "evals" / "fixtures" / "eval_library.json"
OUT_NPZ = ROOT / "backend" / "evals" / "fixtures" / "eval_library_embeddings.npz"
EMBED_DIM = 64

GENRES = ["indie rock", "neo-soul", "ambient", "alt-pop", "garage", "folktronica", "deep house", "jazz rap"]
MOODS = ["sunset", "late night", "sunday morning", "workout", "focus", "rainy", "golden hour", "nostalgic"]
TEMPOS = ["slow", "mid", "upbeat", "downtempo"]

ADJECTIVES = ["amber", "quiet", "velvet", "slow", "coastal", "hollow", "silver", "warm", "drifting", "low"]
NOUNS = ["highway", "porch", "signal", "tide", "hallway", "garden", "static", "horizon", "meadow", "circuit"]
ARTIST_PREFIX = ["The", "Lake", "Metro", "Young", "Blue", "North", "Echo", "Vesper", "Neon", "Coast"]


def _spotify_id(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[:22]


def _base_track(
    rng: random.Random,
    i: int,
    *,
    owner_id: str,
    moods: list[str],
    tags: list[str],
    play_count: int,
    liked: datetime,
    title: str | None = None,
) -> dict:
    genre = GENRES[i % len(GENRES)]
    title = title or f"{rng.choice(ADJECTIVES).title()} {rng.choice(NOUNS).title()}"
    artist = f"{rng.choice(ARTIST_PREFIX)} {rng.choice(NOUNS).title()}"
    year = 2014 + (i % 11)
    tid = _spotify_id(f"eval-lib-{owner_id}-{i}-{artist}-{title}")
    tempo = rng.choice(TEMPOS)
    row = {
        "id": tid,
        "uri": f"spotify:track:{tid}",
        "name": title,
        "artists": [{"id": _spotify_id(f"artist-{artist}"), "name": artist}],
        "album": {"name": f"{genre.title()} Sessions", "release_date": f"{year}-06-15"},
        "owner_id": owner_id,
        "eval_genre": genre,
        "eval_moods": moods,
        "eval_tempo": tempo,
        "eval_tags": tags,
        "eval_play_count": play_count,
        "eval_liked_at": liked.isoformat().replace("+00:00", "Z"),
        "play_count": play_count,
        "liked_at": liked.isoformat().replace("+00:00", "Z"),
        "tags": tags,
        "genre": genre,
    }
    return row


def main() -> None:
    rng = random.Random(SEED)
    base = datetime(2018, 1, 1, tzinfo=timezone.utc)
    tracks: list[dict] = []

    for i in range(PLANTED_SUNSET):
        liked = base + timedelta(days=rng.randint(200, 2500))
        tracks.append(
            _base_track(
                rng,
                i,
                owner_id="eval_user_a",
                moods=["sunset", "golden hour"],
                tags=["loved", "sunset_vibes", "planted_sunset"],
                play_count=rng.randint(180, 520),
                liked=liked,
                title=f"Sunset Memory {i + 1}",
            )
        )

    for i in range(PLANTED_SUNSET, TRACK_COUNT):
        liked = base + timedelta(days=rng.randint(0, 2800))
        moods = [rng.choice(MOODS)]
        tags: list[str] = []
        if "sunset" in moods[0] and rng.random() < 0.15:
            tags.append("sunset_vibes")
        play = max(1, int(rng.lognormvariate(2.0, 0.85)))
        tracks.append(
            _base_track(
                rng,
                i,
                owner_id="eval_user_a",
                moods=moods,
                tags=tags,
                play_count=play,
                liked=liked,
            )
        )

    decoys: list[dict] = []
    for j in range(DECOY_USER_B):
        liked = base + timedelta(days=rng.randint(0, 900))
        decoys.append(
            _base_track(
                rng,
                90000 + j,
                owner_id="eval_user_b",
                moods=["sunset", "golden hour"],
                tags=["loved", "sunset_vibes", "decoy_user_b"],
                play_count=rng.randint(300, 600),
                liked=liked,
                title=f"Sunset Decoy B {j + 1}",
            )
        )

    tracks.sort(key=lambda t: t["liked_at"])
    planted_ids = [t["id"] for t in tracks if "planted_sunset" in (t.get("eval_tags") or [])]

    embedder = HashingEmbedder(dims=EMBED_DIM)
    texts = [library_index_document_text(t, added_at=t["liked_at"]) for t in tracks]
    vectors = np.array(embedder.embed_documents(texts), dtype=np.float32)
    text_hashes = np.array([hashlib.sha256((DOC_PREFIX + t).encode()).hexdigest() for t in texts])

    meta = {
        "version": 2,
        "seed": SEED,
        "track_count": len(tracks),
        "decoy_user_b_count": len(decoys),
        "planted_sets": {"sunset_loved": planted_ids},
        "primary_user_id": "eval_user_a",
        "generated_at": "2026-10-24",
        "embedding": {"model": embedder.model_name, "dimensions": EMBED_DIM},
    }
    payload = {"meta": meta, "tracks": tracks, "decoy_tracks_user_b": decoys}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    np.savez_compressed(
        OUT_NPZ,
        track_ids=np.array([t["id"] for t in tracks]),
        text_hashes=text_hashes,
        vectors=vectors,
    )
    print(f"Wrote {len(tracks)} tracks -> {OUT_JSON} ({OUT_JSON.stat().st_size} bytes)")
    print(f"Planted sunset loved: {len(planted_ids)}")
    print(f"Wrote embeddings -> {OUT_NPZ} ({OUT_NPZ.stat().st_size} bytes)")
    print("Note: for all-MiniLM-L6-v2 doc+query vectors run scripts/regenerate_eval_embeddings.py")


if __name__ == "__main__":
    main()
