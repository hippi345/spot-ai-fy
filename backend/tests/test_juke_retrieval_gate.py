"""Offline Juke-style retrieval gate: vector must beat BM25 on paraphrase queries."""

from __future__ import annotations

from datetime import datetime

import pytest

from evals.eval_library_fixture import planted_sunset_uris, tracks
from evals.fixture_embedder import CommittedFixtureEmbedder
from evals.fixture_index_text import library_index_document_text
from evals.library_retrieval_metrics import ndcg_at_k, recall_at_k
from evals.play_history_fixture import load_fixture_metadata, seed_play_log_db
from evals.retrieval_gate import (
    GateArmMetrics,
    GateCaseResult,
    assert_zero_token_overlap,
    format_gate_report,
)
from evals.retrieval_queries import JUKE_GATE_CASES, LIBRARY_JUKE, PLAY_HISTORY_JUKE
from spot_backend.config import Settings
from spot_backend.library_index import LibraryIndex
from spot_backend.play_log import PlayLogStore

GATE_K = 20


def _library_docs() -> list[dict]:
    docs: list[dict] = []
    for t in tracks("eval_user_a"):
        text = library_index_document_text(t, added_at=t.get("liked_at"))
        play = int(t.get("eval_play_count") or t.get("play_count") or 1)
        import math

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


def _library_planted_texts() -> list[str]:
    ids = {tid.replace("spotify:track:", "") for tid in planted_sunset_uris()}
    texts: list[str] = []
    for t in tracks("eval_user_a"):
        if t["id"] in ids:
            texts.append(f"{t.get('name') or ''} {t.get('artists', [{}])[0].get('name', '')}")
    return texts


def _play_planted_texts(meta: dict) -> list[str]:
    uris = set(meta["planted"]["sunset_this_week"])
    texts: list[str] = []
    for row in meta.get("plays") or []:
        if row["track_uri"] in uris:
            texts.append(f"{row.get('title') or ''} {row.get('artist') or ''}")
    return texts


@pytest.fixture
def library_index(data_dir) -> LibraryIndex:
    embedder = CommittedFixtureEmbedder()
    settings = Settings(data_dir=data_dir)
    idx = LibraryIndex(settings=settings, embedder=embedder)
    idx.sync_from_documents(_library_docs())
    return idx


@pytest.fixture
def play_log_store(tmp_path) -> PlayLogStore:
    s = Settings(data_dir=tmp_path / "data", spot_play_log="on", user_timezone="UTC")
    from evals.play_history_fixture import CommittedPlayHistoryEmbedder

    store = PlayLogStore(settings=s, embedder=CommittedPlayHistoryEmbedder())
    seed_play_log_db(store)
    return store


def test_juke_retrieval_gate_vector_beats_bm25(library_index: LibraryIndex, play_log_store: PlayLogStore) -> None:
    results: list[GateCaseResult] = []
    meta = load_fixture_metadata()
    ref = datetime.fromisoformat(meta["reference_now"].replace("Z", "+00:00"))

    for case in JUKE_GATE_CASES:
        if case.corpus == "library":
            planted_texts = _library_planted_texts()
            assert_zero_token_overlap(case.query, planted_texts)
            relevant = planted_sunset_uris()
            sem = library_index.search(case.query, limit=GATE_K)
            sem_uris = [r["uri"] for r in sem.get("results") or []]
            library_index._keyword_only = True
            kw = library_index.search(case.query, limit=GATE_K)
            library_index._keyword_only = False
            kw_uris = [r["uri"] for r in kw.get("results") or []]
        else:
            planted_texts = _play_planted_texts(meta)
            assert_zero_token_overlap(case.query, planted_texts)
            relevant = set(meta["planted"]["sunset_this_week"])
            sem = play_log_store.search(
                case.query, window=case.window, limit=GATE_K, now=ref
            )
            sem_uris = [r["track_uri"] for r in sem["results"]]
            from evals.play_history_fixture import CommittedPlayHistoryEmbedder

            store_kw = PlayLogStore(
                settings=play_log_store.settings,
                path=play_log_store.path,
                embedder=type(
                    "NoVec",
                    (),
                    {
                        "embed_query": lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no vec")),
                        "embed_documents": lambda *_a, **_k: [],
                        "model_name": "keyword-only",
                        "dimensions": 384,
                    },
                )(),
            )
            kw = store_kw.search(case.query, window=case.window, limit=GATE_K, now=ref)
            kw_uris = [r["track_uri"] for r in kw["results"]]

        v_metrics = GateArmMetrics(
            recall_at_k=recall_at_k(relevant, sem_uris, GATE_K),
            ndcg_at_k=ndcg_at_k(relevant, sem_uris, GATE_K),
        )
        b_metrics = GateArmMetrics(
            recall_at_k=recall_at_k(relevant, kw_uris, GATE_K),
            ndcg_at_k=ndcg_at_k(relevant, kw_uris, GATE_K),
        )
        margin_r = v_metrics.recall_at_k - b_metrics.recall_at_k
        margin_n = v_metrics.ndcg_at_k - b_metrics.ndcg_at_k
        wins = v_metrics.ndcg_at_k > b_metrics.ndcg_at_k or (
            v_metrics.ndcg_at_k == b_metrics.ndcg_at_k and margin_r > 0
        )
        results.append(
            GateCaseResult(
                name=case.name,
                query=case.query,
                k=GATE_K,
                vector=v_metrics,
                bm25=b_metrics,
                margin_recall=margin_r,
                margin_ndcg=margin_n,
                vector_wins=wins,
            )
        )

    report = format_gate_report(results)
    print(report)
    failures = [r for r in results if not r.vector_wins]
    if failures:
        pytest.fail(
            "Vector retrieval did not beat BM25/FTS on paraphrase gate:\n" + report
        )
