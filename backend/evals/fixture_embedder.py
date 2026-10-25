"""Load committed fixture embeddings for deterministic vector search tests."""

from __future__ import annotations

from pathlib import Path

from evals.committed_npz import load_vector_maps, lookup_document, lookup_query
from evals.eval_library_fixture import EMBEDDINGS_PATH
from evals.fixture_index_text import library_index_document_text


class CommittedFixtureEmbedder:
    """Precomputed all-MiniLM-L6-v2 vectors keyed by index-text / query hash."""

    def __init__(self, npz_path: Path | None = None) -> None:
        path = npz_path or EMBEDDINGS_PATH
        doc_by_hash, query_by_hash, model, dim = load_vector_maps(path)
        self._doc_by_hash = doc_by_hash
        self._query_by_hash = query_by_hash
        self._model_name = model
        self._dims = dim

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


def document_text_for_track(track: dict) -> str:
    return library_index_document_text(track, added_at=track.get("liked_at"))
