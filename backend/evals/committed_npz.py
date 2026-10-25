from __future__ import annotations

from pathlib import Path

import numpy as np

from evals.text_hash import sha256_text


class MissingCommittedQueryEmbeddingError(RuntimeError):
    pass


class MissingCommittedDocumentEmbeddingError(RuntimeError):
    pass


def load_vector_maps(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str, int]:
    data = np.load(path, allow_pickle=False)
    doc_by_hash: dict[str, np.ndarray] = {}
    for h, vec in zip(data["text_hashes"], data["vectors"]):
        doc_by_hash[str(h)] = np.asarray(vec, dtype=np.float32)
    query_by_hash: dict[str, np.ndarray] = {}
    if "query_hashes" in data and "query_vectors" in data:
        for h, vec in zip(data["query_hashes"], data["query_vectors"]):
            query_by_hash[str(h)] = np.asarray(vec, dtype=np.float32)
    model = "unknown"
    if "embed_model" in data:
        raw = data["embed_model"]
        model = str(raw.item() if hasattr(raw, "item") else raw)
    dim = int(next(iter(doc_by_hash.values())).shape[0]) if doc_by_hash else 0
    return doc_by_hash, query_by_hash, model, dim


def lookup_document(doc_by_hash: dict[str, np.ndarray], text: str) -> list[float]:
    h = sha256_text(text)
    row = doc_by_hash.get(h)
    if row is None:
        raise MissingCommittedDocumentEmbeddingError(
            f"No committed document embedding for text hash {h[:12]}… "
            "(run scripts/regenerate_eval_embeddings.py)"
        )
    return row.tolist()


def lookup_query(query_by_hash: dict[str, np.ndarray], text: str) -> list[float]:
    h = sha256_text(text)
    row = query_by_hash.get(h)
    if row is None:
        raise MissingCommittedQueryEmbeddingError(
            f"No committed query embedding for text hash {h[:12]}… "
            "(run scripts/regenerate_eval_embeddings.py)"
        )
    return row.tolist()
