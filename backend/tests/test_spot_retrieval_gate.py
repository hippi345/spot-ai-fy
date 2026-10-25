"""Offline Spot retrieval gate: vector must beat BM25 on paraphrase queries."""

from __future__ import annotations

from datetime import datetime

import pytest

from evals.eval_library_fixture import planted_sunset_uris, tracks
from evals.fixture_embedder import CommittedFixtureEmbedder
from evals.fixture_index_text import library_index_document_text
from evals.library_retrieval_metrics import ndcg_at_k, precision_at_k, recall_at_k
from evals.play_history_fixture import CommittedPlayHistoryEmbedder, load_fixture_metadata, seed_play_log_db
from evals.retrieval_gate import (
    GateArmMetrics,
    GateCaseResult,
    assert_zero_token_overlap,
    format_gate_report,
    summarize_arm,
)
from evals.retrieval_gate_config import (
    LIBRARY_GATE_K,
    LIBRARY_MEAN_NDCG_AT_K_FLOOR,
    LIBRARY_MEAN_PRECISION_AT_K_FLOOR,
    LIBRARY_PLANTED_RELEVANT_COUNT,
    MIN_PARAPHRASE_QUERIES_PER_ARM,
    MIN_VECTOR_WINS_PER_ARM,
    PLAY_HISTORY_GATE_K,
    PLAY_HISTORY_MEAN_RECALL_AT_K_FLOOR,
    PLAY_HISTORY_THIS_WEEK_RELEVANT_COUNT,
)
from evals.retrieval_queries import library_gate_cases, play_history_gate_cases
from spot_backend.config import Settings
from spot_backend.library_index import LibraryIndex
from spot_backend.play_log import PlayLogStore

SPOT_GATE_CASES = library_gate_cases() + play_history_gate_cases()


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
    return [
        library_index_document_text(t, added_at=t.get("liked_at"))
        for t in tracks("eval_user_a")
        if t["id"] in ids
    ]


def _play_planted_texts(meta: dict) -> list[str]:
    from spot_backend.play_log import play_index_text

    uris = set(meta["planted"]["sunset_this_week"])
    texts: list[str] = []
    for row in meta.get("plays") or []:
        if row["track_uri"] in uris:
            texts.append(
                play_index_text(
                    row["title"],
                    row["artist"],
                    row["album"],
                    row.get("request_text"),
                    mood_tags=row.get("mood_tags", ""),
                )
            )
    return texts


def _metrics(relevant: set[str], uris: list[str], k: int) -> GateArmMetrics:
    return GateArmMetrics(
        recall_at_k=recall_at_k(relevant, uris, k),
        precision_at_k=precision_at_k(relevant, uris, k),
        ndcg_at_k=ndcg_at_k(relevant, uris, k),
    )


def _vector_wins(vector: GateArmMetrics, bm25: GateArmMetrics) -> bool:
    if vector.ndcg_at_k > bm25.ndcg_at_k:
        return True
    if vector.ndcg_at_k < bm25.ndcg_at_k:
        return False
    return vector.recall_at_k > bm25.recall_at_k


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
    store = PlayLogStore(settings=s, embedder=CommittedPlayHistoryEmbedder())
    seed_play_log_db(store)
    return store


def test_spot_retrieval_gate_paraphrase_vector_beats_bm25(
    library_index: LibraryIndex, play_log_store: PlayLogStore
) -> None:
    lib_cases = library_gate_cases()
    play_cases = play_history_gate_cases()
    assert len(lib_cases) >= MIN_PARAPHRASE_QUERIES_PER_ARM
    assert len(play_cases) >= MIN_PARAPHRASE_QUERIES_PER_ARM

    results: list[GateCaseResult] = []
    meta = load_fixture_metadata()
    ref = datetime.fromisoformat(meta["reference_now"].replace("Z", "+00:00"))
    library_planted_texts = _library_planted_texts()
    play_planted_texts = _play_planted_texts(meta)

    for case in SPOT_GATE_CASES:
        if case.corpus == "library":
            k = LIBRARY_GATE_K
            assert_zero_token_overlap(case.query, library_planted_texts)
            relevant = planted_sunset_uris()
            relevant_count = LIBRARY_PLANTED_RELEVANT_COUNT
            sem = library_index.search(case.query, limit=k)
            sem_uris = [r["uri"] for r in sem.get("results") or []]
            library_index._keyword_only = True
            kw = library_index.search(case.query, limit=k)
            library_index._keyword_only = False
            kw_uris = [r["uri"] for r in kw.get("results") or []]
        else:
            k = PLAY_HISTORY_GATE_K
            assert_zero_token_overlap(case.query, play_planted_texts)
            relevant = set(meta["planted"]["sunset_this_week"])
            relevant_count = PLAY_HISTORY_THIS_WEEK_RELEVANT_COUNT
            sem = play_log_store.search(case.query, window=case.window, limit=k, now=ref)
            sem_uris = [r["track_uri"] for r in sem["results"]]
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
            kw = store_kw.search(case.query, window=case.window, limit=k, now=ref)
            kw_uris = [r["track_uri"] for r in kw["results"]]

        v_metrics = _metrics(relevant, sem_uris, k)
        b_metrics = _metrics(relevant, kw_uris, k)
        wins = _vector_wins(v_metrics, b_metrics)
        results.append(
            GateCaseResult(
                name=case.name,
                query=case.query,
                corpus=case.corpus,
                k=k,
                relevant_count=relevant_count,
                vector=v_metrics,
                bm25=b_metrics,
                margin_recall=v_metrics.recall_at_k - b_metrics.recall_at_k,
                margin_ndcg=v_metrics.ndcg_at_k - b_metrics.ndcg_at_k,
                vector_wins=wins,
            )
        )

    report = format_gate_report(
        results,
        library_relevant=LIBRARY_PLANTED_RELEVANT_COUNT,
        play_relevant=PLAY_HISTORY_THIS_WEEK_RELEVANT_COUNT,
    )
    print(report)

    lib = summarize_arm(results, "library")
    play = summarize_arm(results, "play_history")

    failures: list[str] = []
    if lib.vector_wins < MIN_VECTOR_WINS_PER_ARM:
        failures.append(
            f"library wins {lib.vector_wins}/{lib.query_count} "
            f"(need >={MIN_VECTOR_WINS_PER_ARM})"
        )
    if play.vector_wins < MIN_VECTOR_WINS_PER_ARM:
        failures.append(
            f"play_history wins {play.vector_wins}/{play.query_count} "
            f"(need >={MIN_VECTOR_WINS_PER_ARM})"
        )

    # Library |relevant|=50 with k=20 caps recall@20 at 0.40; gate on precision + nDCG.
    if lib.mean_vector_precision < LIBRARY_MEAN_PRECISION_AT_K_FLOOR:
        failures.append(
            f"library mean vector precision@{LIBRARY_GATE_K}={lib.mean_vector_precision:.4f} "
            f"< floor {LIBRARY_MEAN_PRECISION_AT_K_FLOOR}"
        )
    if lib.mean_vector_ndcg < LIBRARY_MEAN_NDCG_AT_K_FLOOR:
        failures.append(
            f"library mean vector nDCG@{LIBRARY_GATE_K}={lib.mean_vector_ndcg:.4f} "
            f"< floor {LIBRARY_MEAN_NDCG_AT_K_FLOOR}"
        )
    if play.mean_vector_recall < PLAY_HISTORY_MEAN_RECALL_AT_K_FLOOR:
        failures.append(
            f"play_history mean vector recall@{PLAY_HISTORY_GATE_K}={play.mean_vector_recall:.4f} "
            f"< floor {PLAY_HISTORY_MEAN_RECALL_AT_K_FLOOR}"
        )

    if failures:
        pytest.fail(
            "Spot retrieval gate failed:\n"
            + "\n".join(failures)
            + f"\n\nThresholds: wins>={MIN_VECTOR_WINS_PER_ARM}/{lib.query_count} per arm; "
            f"library mean prec>={LIBRARY_MEAN_PRECISION_AT_K_FLOOR}, "
            f"mean nDCG>={LIBRARY_MEAN_NDCG_AT_K_FLOOR} "
            f"(|relevant|={LIBRARY_PLANTED_RELEVANT_COUNT}, k={LIBRARY_GATE_K}); "
            f"play mean recall>={PLAY_HISTORY_MEAN_RECALL_AT_K_FLOOR} "
            f"(|relevant|={PLAY_HISTORY_THIS_WEEK_RELEVANT_COUNT}, k={PLAY_HISTORY_GATE_K})\n\n"
            + report
        )
