"""Load committed play-history fixture + embeddings for deterministic tests."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from evals.committed_npz import load_vector_maps, lookup_document, lookup_query
from evals.text_hash import sha256_text
from spot_backend.play_log import PlayLogStore, play_index_text

FIXTURE_JSON = Path(__file__).resolve().parent / "fixtures" / "play_history_seed.json"
FIXTURE_NPZ = Path(__file__).resolve().parent / "fixtures" / "play_history_embeddings.npz"


class CommittedPlayHistoryEmbedder:
    """Document + query vectors from committed npz (all-MiniLM-L6-v2)."""

    def __init__(self) -> None:
        doc_by_hash, query_by_hash, model, dim = load_vector_maps(FIXTURE_NPZ)
        self._doc_by_hash = doc_by_hash
        self._query_by_hash = query_by_hash
        self._model_name = model
        self._dims = dim
        self._by_row: dict[int, list[float]] = {}
        data = np.load(FIXTURE_NPZ)
        ids = data["ids"]
        vectors = data["vectors"]
        for i, rid in enumerate(ids):
            self._by_row[int(rid)] = vectors[i].astype(np.float32).tolist()
        self._index_hash_to_row: dict[str, int] = {}
        meta = load_fixture_metadata()
        for idx, row in enumerate(meta.get("plays") or [], start=1):
            itext = play_index_text(
                row["title"],
                row["artist"],
                row["album"],
                row.get("request_text"),
                mood_tags=row.get("mood_tags", ""),
            )
            self._index_hash_to_row[sha256_text(itext)] = idx

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [lookup_document(self._doc_by_hash, t or "") for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return lookup_query(self._query_by_hash, text or "")

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
