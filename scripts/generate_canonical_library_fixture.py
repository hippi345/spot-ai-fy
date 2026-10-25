#!/usr/bin/env python3
"""Deterministic ~2k-track eval library (no personal data). Commit output to backend/evals/fixtures/."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEED = 0xC0FFEE17
TRACK_COUNT = 2200
OUT = Path(__file__).resolve().parents[1] / "backend" / "evals" / "fixtures" / "canonical_library.json"

GENRES = [
    "indie rock",
    "neo-soul",
    "ambient",
    "alt-pop",
    "garage",
    "folktronica",
    "deep house",
    "jazz rap",
    "shoegaze",
    "synthwave",
]

# Hidden taste cluster: high play-count skew toward "neon pastoral" (not obvious from titles alone).
TASTE_TAG = "neon_pastoral"
SUMMER_TAG = "summer_anchor"

ADJECTIVES = [
    "amber",
    "quiet",
    "velvet",
    "slow",
    "coastal",
    "hollow",
    "silver",
    "warm",
    "drifting",
    "low",
]
NOUNS = [
    "highway",
    "porch",
    "signal",
    "tide",
    "hallway",
    "garden",
    "static",
    "horizon",
    "meadow",
    "circuit",
]
ARTIST_PREFIX = ["The", "DJ", "Lake", "Metro", "Young", "Old", "Blue", "North", "Echo", "Vesper"]


def _spotify_id(seed: str) -> str:
    h = hashlib.sha256(seed.encode()).hexdigest()
    return h[:22]


def _track_row(rng: random.Random, i: int) -> dict:
    genre = GENRES[i % len(GENRES)]
    title = f"{rng.choice(ADJECTIVES).title()} {rng.choice(NOUNS).title()}"
    artist = f"{rng.choice(ARTIST_PREFIX)} {rng.choice(NOUNS).title()}"
    year = 2014 + (i % 12)
    month = 1 + (i % 12)
    day = 1 + (i % 28)
    liked = datetime(year, month, day, tzinfo=timezone.utc) + timedelta(days=rng.randint(0, 400))
    play_count = max(1, int(rng.lognormvariate(2.2, 0.9)))
    tags: list[str] = []
    if rng.random() < 0.08:
        tags.append(TASTE_TAG)
    if liked.month in (6, 7, 8) and liked.year == 2024 and rng.random() < 0.35:
        tags.append(SUMMER_TAG)
    if TASTE_TAG in tags and rng.random() < 0.65:
        play_count = int(play_count * rng.uniform(2.5, 6.0))
    tid = _spotify_id(f"canonical-track-{i}-{artist}-{title}")
    album = f"{genre.title()} Years, Vol. {(i % 7) + 1}"
    return {
        "id": tid,
        "uri": f"spotify:track:{tid}",
        "name": title,
        "artists": [{"id": _spotify_id(f"artist-{artist}"), "name": artist}],
        "album": {"name": album, "release_date": f"{year}-{month:02d}-{day:02d}"},
        "play_count": play_count,
        "liked_at": liked.isoformat().replace("+00:00", "Z"),
        "tags": tags,
        "genre": genre,
    }


def main() -> None:
    rng = random.Random(SEED)
    tracks = [_track_row(rng, i) for i in range(TRACK_COUNT)]
    tracks.sort(key=lambda t: t["liked_at"])
    meta = {
        "version": 1,
        "seed": SEED,
        "track_count": len(tracks),
        "taste_tag": TASTE_TAG,
        "taste_note": (
            "Top-played likes skew toward tracks tagged neon_pastoral in fixture metadata "
            "(not exposed in Spotify API fields the agent sees)."
        ),
        "generated_at": "2026-10-24",
    }
    payload = {"meta": meta, "tracks": tracks}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(tracks)} tracks to {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
