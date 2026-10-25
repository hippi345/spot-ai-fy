#!/usr/bin/env python3
"""Regenerate eval library + play-history doc/query embeddings (offline MiniLM).

Model: sentence-transformers/all-MiniLM-L6-v2 (Spot offline retrieval gate).
No Spotify network calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from evals.embedding_constants import EVAL_EMBED_NORMALIZE, EVAL_SENTENCE_TRANSFORMER_MODEL
from evals.eval_library_fixture import FIXTURE_PATH as LIB_JSON
from evals.fixture_index_text import library_index_document_text
from evals.play_history_fixture import FIXTURE_JSON as PLAY_JSON
from evals.play_history_fixture import FIXTURE_NPZ as PLAY_NPZ
from evals.retrieval_queries import LIBRARY_COMMITTED_QUERIES, PLAY_HISTORY_COMMITTED_QUERIES
from evals.text_hash import sha256_text
from spot_backend.play_log import play_index_text

LIB_NPZ = ROOT / "backend" / "evals" / "fixtures" / "eval_library_embeddings.npz"


def _encode(model, texts: list[str]) -> np.ndarray:
    vecs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=EVAL_EMBED_NORMALIZE,
    )
    return np.asarray(vecs, dtype=np.float32)


def _write_npz(
    path: Path,
    *,
    text_hashes: list[str],
    vectors: np.ndarray,
    query_hashes: list[str],
    query_vectors: np.ndarray,
    extra: dict[str, np.ndarray] | None = None,
) -> None:
    payload: dict[str, np.ndarray] = {
        "text_hashes": np.array(text_hashes),
        "vectors": vectors,
        "query_hashes": np.array(query_hashes),
        "query_vectors": query_vectors,
        "embed_model": np.array(EVAL_SENTENCE_TRANSFORMER_MODEL),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"Wrote {path} ({path.stat().st_size} bytes)")


def regenerate_library(model) -> None:
    data = json.loads(LIB_JSON.read_text(encoding="utf-8"))
    tracks = data["tracks"]
    texts = [library_index_document_text(t, added_at=t.get("liked_at")) for t in tracks]
    vectors = _encode(model, texts)
    text_hashes = [sha256_text(t) for t in texts]
    q_vecs = _encode(model, list(LIBRARY_COMMITTED_QUERIES))
    q_hashes = [sha256_text(q) for q in LIBRARY_COMMITTED_QUERIES]
    meta = data.get("meta") or {}
    meta["embedding"] = {
        "model": EVAL_SENTENCE_TRANSFORMER_MODEL,
        "dimensions": int(vectors.shape[1]),
        "normalize": EVAL_EMBED_NORMALIZE,
        "regenerated_by": "scripts/regenerate_eval_embeddings.py",
    }
    data["meta"] = meta
    LIB_JSON.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    _write_npz(
        LIB_NPZ,
        text_hashes=text_hashes,
        vectors=vectors,
        query_hashes=q_hashes,
        query_vectors=q_vecs,
        extra={"track_ids": np.array([t["id"] for t in tracks])},
    )
    print(f"Updated {LIB_JSON}")


def regenerate_play_history(model) -> None:
    meta = json.loads(PLAY_JSON.read_text(encoding="utf-8"))
    plays = meta.get("plays") or []
    texts: list[str] = []
    ids: list[int] = []
    for idx, row in enumerate(plays, start=1):
        itext = play_index_text(
            row["title"],
            row["artist"],
            row["album"],
            row.get("request_text"),
            mood_tags=row.get("mood_tags", ""),
        )
        texts.append(itext)
        ids.append(idx)
    vectors = _encode(model, texts)
    text_hashes = [sha256_text(t) for t in texts]
    q_vecs = _encode(model, list(PLAY_HISTORY_COMMITTED_QUERIES))
    q_hashes = [sha256_text(q) for q in PLAY_HISTORY_COMMITTED_QUERIES]
    meta["embedding"] = {
        "model": EVAL_SENTENCE_TRANSFORMER_MODEL,
        "dimensions": int(vectors.shape[1]),
        "normalize": EVAL_EMBED_NORMALIZE,
        "regenerated_by": "scripts/regenerate_eval_embeddings.py",
    }
    PLAY_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_npz(
        PLAY_NPZ,
        text_hashes=text_hashes,
        vectors=vectors,
        query_hashes=q_hashes,
        query_vectors=q_vecs,
        extra={"ids": np.array(ids, dtype=np.int32)},
    )
    print(f"Updated {PLAY_JSON}")


def main() -> None:
    from sentence_transformers import SentenceTransformer

    print(f"Loading {EVAL_SENTENCE_TRANSFORMER_MODEL} (offline cache after first run)…")
    model = SentenceTransformer(EVAL_SENTENCE_TRANSFORMER_MODEL)
    regenerate_library(model)
    regenerate_play_history(model)
    print("Done.")


if __name__ == "__main__":
    main()
