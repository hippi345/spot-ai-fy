#!/usr/bin/env python3
"""Deterministic play history fixture (~3k plays) + committed embeddings for CI."""

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

from spot_backend.ollama_embed import HashingEmbedder
from spot_backend.play_log import play_index_text

SEED = 0x50A90123
CATALOG_SIZE = 2000
PLAY_COUNT = 3000
OUT_JSON = ROOT / "backend" / "evals" / "fixtures" / "play_history_seed.json"
OUT_NPZ = ROOT / "backend" / "evals" / "fixtures" / "play_history_embeddings.npz"
EMBED_DIM = 64

MOODS = ["sunset", "workout", "focus", "party", "rainy", "chill", "golden hour", "dusk"]
GENRES = ["indie", "synth", "lofi", "house", "folk", "rnb", "ambient"]


def _tid(i: int) -> str:
    return hashlib.sha256(f"play-fix-{i}".encode()).hexdigest()[:22]


def main() -> None:
    rng = random.Random(SEED)
    now = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
    catalog: list[dict] = []
    for i in range(CATALOG_SIZE):
        mood = MOODS[i % len(MOODS)]
        genre = GENRES[i % len(GENRES)]
        title = f"{mood.title()} Track {i % 97}"
        artist = f"Fixture Artist {i % 41}"
        album = f"{genre.title()} Sessions"
        catalog.append(
            {
                "track_uri": f"spotify:track:{_tid(i)}",
                "title": title,
                "artist": artist,
                "album": album,
                "mood_tags": f"{mood} {genre}",
            }
        )

    plays: list[dict] = []
    planted_sunset_this_week: list[str] = []
    planted_sunset_old: list[str] = []
    planted_sunset_skipped: list[str] = []
    prune_90_ids: list[int] = []
    prune_skip_ids: list[int] = []

    # ~15 sunset this week (non-skipped)
    for j in range(15):
        c = catalog[j % 40]
        if "sunset" not in c["mood_tags"]:
            c = {**c, "mood_tags": "sunset golden hour", "title": f"Sunset Glow {j}"}
        played = now - timedelta(days=rng.randint(0, 3), hours=rng.randint(1, 10))
        row = {
            **c,
            "played_at": played.isoformat().replace("+00:00", "Z"),
            "request_text": "play some golden hour chill",
            "skipped": False,
            "listened_ms": 180_000,
            "plant": "sunset_this_week",
        }
        plays.append(row)
        planted_sunset_this_week.append(c["track_uri"])

    # decoy sunset older than a week
    for j in range(12):
        c = catalog[50 + j]
        played = now - timedelta(days=10 + j)
        row = {
            **c,
            "mood_tags": "sunset dusk",
            "played_at": played.isoformat().replace("+00:00", "Z"),
            "request_text": "sunset vibes",
            "skipped": False,
            "listened_ms": 120_000,
            "plant": "sunset_old",
        }
        plays.append(row)
        planted_sunset_old.append(c["track_uri"])

    # skipped sunset this week
    for j in range(4):
        c = catalog[100 + j]
        played = now - timedelta(days=1)
        row = {
            **c,
            "mood_tags": "sunset",
            "played_at": played.isoformat().replace("+00:00", "Z"),
            "request_text": "sunset skip test",
            "skipped": True,
            "listened_ms": 5_000,
            "plant": "sunset_skipped",
        }
        plays.append(row)
        planted_sunset_skipped.append(c["track_uri"])

    # fill to PLAY_COUNT
    while len(plays) < PLAY_COUNT - 20:
        i = len(plays)
        c = dict(catalog[i % CATALOG_SIZE])
        age_days = rng.randint(8, 85)
        if age_days < 7 and "sunset" in c.get("mood_tags", ""):
            c["mood_tags"] = c["mood_tags"].replace("sunset", "focus")
        played = now - timedelta(days=age_days, hours=rng.randint(0, 23))
        skipped = rng.random() < 0.08 and age_days < 5
        plant = None
        row = {
            **c,
            "played_at": played.isoformat().replace("+00:00", "Z"),
            "request_text": rng.choice(["play chill", "workout mix", "focus time", "party tracks", ""]),
            "skipped": skipped,
            "listened_ms": rng.randint(30_000, 240_000),
            "plant": plant,
        }
        plays.append(row)

    # Explicit prune fixtures (row ids assigned after append order below)
    for j in range(10):
        c = catalog[500 + j]
        played = now - timedelta(days=95 + j)
        plays.append(
            {
                **c,
                "played_at": played.isoformat().replace("+00:00", "Z"),
                "request_text": "old play",
                "skipped": False,
                "listened_ms": 60_000,
                "plant": "prune_90",
            }
        )
    for j in range(8):
        c = catalog[600 + j]
        played = now - timedelta(days=12 + j)
        plays.append(
            {
                **c,
                "played_at": played.isoformat().replace("+00:00", "Z"),
                "request_text": "skipped old",
                "skipped": True,
                "listened_ms": 2_000,
                "plant": "prune_skip",
            }
        )

    embedder = HashingEmbedder(dims=EMBED_DIM)
    ids: list[int] = []
    vectors: list[np.ndarray] = []
    for idx, row in enumerate(plays, start=1):
        if row.get("plant") == "prune_90":
            prune_90_ids.append(idx)
        if row.get("plant") == "prune_skip":
            prune_skip_ids.append(idx)
        itext = play_index_text(
            row["title"],
            row["artist"],
            row["album"],
            row.get("request_text"),
            mood_tags=row.get("mood_tags", ""),
        )
        vec = np.array(embedder.embed_documents([itext])[0], dtype=np.float32)
        ids.append(idx)
        vectors.append(vec)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "reference_now": now.isoformat().replace("+00:00", "Z"),
        "plays": plays,
        "planted": {
            "sunset_this_week": planted_sunset_this_week,
            "sunset_old": planted_sunset_old,
            "sunset_skipped": planted_sunset_skipped,
            "prune_90_row_ids": prune_90_ids,
            "prune_skip_row_ids": prune_skip_ids,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(OUT_NPZ, ids=np.array(ids, dtype=np.int32), vectors=np.vstack(vectors))
    print(f"Wrote {OUT_JSON} ({len(plays)} plays)")
    print(f"Wrote {OUT_NPZ}")


if __name__ == "__main__":
    main()
