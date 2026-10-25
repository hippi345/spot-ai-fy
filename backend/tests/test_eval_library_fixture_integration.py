"""SAMPLE library fixture — deterministic vector search integration (mock CI, no LLM)."""

from __future__ import annotations

import math

import pytest

from evals.eval_library_fixture import decoy_tracks_user_b, planted_sunset_uris, tracks
from evals.fixture_embedder import CommittedFixtureEmbedder
from evals.fixture_index_text import library_index_document_text
from evals.library_retrieval_metrics import bench_search, precision_at_k, recall_at_k
from evals.retrieval_queries import LIBRARY_SUNSET_LITERAL
from spot_backend.config import Settings
from spot_backend.library_index import LibraryIndex

RECALL_AT_20_MIN = 0.12
PRECISION_AT_20_MIN = 0.08
P95_MS_MAX = 100.0
QUERY_SUNSET = LIBRARY_SUNSET_LITERAL


def _docs_for_user(user_id: str = "eval_user_a") -> list[dict]:
    docs: list[dict] = []
    for t in tracks(user_id):
        text = library_index_document_text(t, added_at=t.get("liked_at"))
        play = int(t.get("eval_play_count") or t.get("play_count") or 1)
        boost = 1.0 + math.log1p(max(0, play)) * 0.06
        docs.append(
            {
                "id": f"track:{t['id']}",
                "kind": "track",
                "uri": t["uri"],
                "text": text,
                "boost": boost,
            }
        )
    return docs


@pytest.fixture
def indexed_library(data_dir) -> LibraryIndex:
    embedder = CommittedFixtureEmbedder()
    settings = Settings(data_dir=data_dir)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    idx = LibraryIndex(settings=settings, embedder=embedder)
    idx.sync_from_documents(_docs_for_user())
    return idx


def test_sunset_query_recall_and_precision(indexed_library: LibraryIndex) -> None:
    relevant = planted_sunset_uris()
    sem = indexed_library.search(QUERY_SUNSET, limit=20)
    assert sem.get("mode") == "semantic"
    sem_uris = [r["uri"] for r in sem.get("results") or []]
    sem_recall = recall_at_k(relevant, sem_uris, 20)
    sem_precision = precision_at_k(relevant, sem_uris, 20)
    assert sem_recall >= RECALL_AT_20_MIN, f"semantic recall@20 {sem_recall}"
    assert sem_precision >= PRECISION_AT_20_MIN, f"semantic precision@20 {sem_precision}"


def test_user_scoping_excludes_decoy_catalog(indexed_library: LibraryIndex) -> None:
    decoy_uris = {t["uri"] for t in decoy_tracks_user_b()}
    assert decoy_uris
    out = indexed_library.search(QUERY_SUNSET, limit=20)
    hits = {r["uri"] for r in out.get("results") or []}
    assert not hits.intersection(decoy_uris)


def test_play_boost_honored(indexed_library: LibraryIndex) -> None:
    relevant = planted_sunset_uris()
    out = indexed_library.search(QUERY_SUNSET, limit=5)
    top = [r["uri"] for r in out.get("results") or []]
    assert top
    assert len(set(top) & relevant) >= 1


def test_search_p95_latency_under_100ms(indexed_library: LibraryIndex) -> None:
    p95, _ = bench_search(lambda: indexed_library.search(QUERY_SUNSET, limit=20), n=100)
    assert p95 < P95_MS_MAX, f"p95 latency {p95:.1f}ms"
