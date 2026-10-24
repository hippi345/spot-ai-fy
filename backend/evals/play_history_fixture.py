"""Load committed play-history fixture + embeddings for deterministic tests."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from spot_backend.ollama_embed import HashingEmbedder
from spot_backend.play_log import PlayLogStore, play_index_text

FIXTURE_JSON = Path(__file__).resolve().parent / "fixtures" / "play_history_seed.json"
FIXTURE_NPZ = Path(__file__).resolve().parent / "fixtures" / "play_history_embeddings.npz"


class CommittedPlayHistoryEmbedder:
    """Query vectors via hashing; document vectors from committed npz when seeding."""

    def __init__(self, dims: int = 64) -> None:
        self._hash = HashingEmbedder(dims=dims)
        self._by_row: dict[int, list[float]] = {}
        if FIXTURE_NPZ.is_file():
            data = np.load(FIXTURE_NPZ)
            ids = data["ids"]
            vectors = data["vectors"]
            for i, rid in enumerate(ids):
                self._by_row[int(rid)] = vectors[i].astype(np.float32).tolist()

    @property
    def model_name(self) -> str:
        return "play-history-fixture"

    @property
    def dimensions(self) -> int:
        return self._hash.dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._hash.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._hash.embed_query(text)

    def vector_for_row(self, row_id: int) -> list[float] | None:
        return self._by_row.get(row_id)


def load_fixture_metadata() -> dict:
    return json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))


def seed_play_log_db(store: PlayLogStore, *, reference_now: datetime | None = None) -> dict:
    """Populate play_log.sqlite from committed JSON + npz."""
    meta = load_fixture_metadata()
    ref = reference_now
    if ref is None and meta.get("reference_now"):
        ref = datetime.fromisoformat(str(meta["reference_now"]).replace("Z", "+00:00"))
    embedder = CommittedPlayHistoryEmbedder()
    for idx, row in enumerate(meta.get("plays") or [], start=1):
        played = datetime.fromisoformat(str(row["played_at"]).replace("Z", "+00:00"))
        vec = embedder.vector_for_row(idx)
        store.insert_play(
            track_uri=str(row["track_uri"]),
            title=str(row["title"]),
            artist=str(row["artist"]),
            album=str(row["album"]),
            request_text=str(row.get("request_text") or "") or None,
            conversation_id="fixture",
            played_at=played,
            mood_tags=str(row.get("mood_tags") or ""),
            skipped=bool(row.get("skipped")),
            listened_ms=int(row.get("listened_ms") or 0),
            vec=vec,
        )
    return meta.get("planted") or {}
