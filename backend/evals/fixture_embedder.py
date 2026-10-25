"""Load committed fixture embeddings for deterministic vector search tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from evals.eval_library_fixture import EMBEDDINGS_PATH
from evals.fixture_index_text import library_index_document_text
from spot_backend.ollama_embed import DOC_PREFIX, HashingEmbedder


class CommittedFixtureEmbedder:
    """Embeds using precomputed vectors keyed by document text hash."""

    def __init__(self, npz_path: Path | None = None, *, dims: int = 64) -> None:
        path = npz_path or EMBEDDINGS_PATH
        data = np.load(path)
        self._dims = dims
        self._by_hash: dict[str, np.ndarray] = {}
        for h, vec in zip(data["text_hashes"], data["vectors"]):
            self._by_hash[str(h)] = np.asarray(vec, dtype=np.float32)
        self._fallback = HashingEmbedder(dims=dims)

    @property
    def model_name(self) -> str:
        return "eval-fixture-committed"

    @property
    def dimensions(self) -> int:
        return self._dims

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _vec(self, prefixed: str) -> list[float]:
        h = self._hash(prefixed)
        row = self._by_hash.get(h)
        if row is not None:
            return row.tolist()
        return self._fallback._vec(prefixed)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(DOC_PREFIX + (t or "")) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._fallback.embed_query(text)


def document_text_for_track(track: dict) -> str:
    return library_index_document_text(track, added_at=track.get("liked_at"))
