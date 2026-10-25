"""Offline Spot retrieval gate (vector vs BM25/FTS, same tokenizer)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from evals.library_retrieval_metrics import precision_at_k


def fts_query_tokens(user_query: str) -> set[str]:
    """Same tokenization as SQLite FTS MATCH (format_fts_match_query)."""
    text = (user_query or "").strip().replace("/", " ")
    tokens = re.findall(r"[\w']+", text, flags=re.UNICODE)
    return {t.lower().replace("'", "") for t in tokens if t.strip()}


def planted_corpus_tokens(*texts: str) -> set[str]:
    out: set[str] = set()
    for blob in texts:
        out |= fts_query_tokens(blob)
    return out


def assert_zero_token_overlap(query: str, planted_texts: list[str]) -> None:
    """No FTS token overlap with planted indexed doc text (same tokenizer as BM25)."""
    q = fts_query_tokens(query)
    p = planted_corpus_tokens(*planted_texts)
    overlap = q & p
    if overlap:
        raise AssertionError(
            f"query/planted indexed-doc token overlap not allowed for {query!r}: {sorted(overlap)}"
        )


@dataclass
class GateArmMetrics:
    recall_at_k: float
    precision_at_k: float
    ndcg_at_k: float


@dataclass
class GateCaseResult:
    name: str
    query: str
    corpus: str
    k: int
    relevant_count: int
    vector: GateArmMetrics
    bm25: GateArmMetrics
    margin_recall: float
    margin_ndcg: float
    vector_wins: bool
    reported_only: bool = False


@dataclass
class GateArmSummary:
    corpus: str
    query_count: int
    vector_wins: int
    mean_vector_recall: float
    mean_vector_precision: float
    mean_vector_ndcg: float
    mean_bm25_recall: float
    mean_bm25_precision: float
    mean_bm25_ndcg: float


def summarize_arm(results: list[GateCaseResult], corpus: str) -> GateArmSummary:
    arm = [r for r in results if r.corpus == corpus]
    if not arm:
        raise ValueError(f"no gate results for corpus {corpus}")
    wins = sum(1 for r in arm if r.vector_wins)
    return GateArmSummary(
        corpus=corpus,
        query_count=len(arm),
        vector_wins=wins,
        mean_vector_recall=sum(r.vector.recall_at_k for r in arm) / len(arm),
        mean_vector_precision=sum(r.vector.precision_at_k for r in arm) / len(arm),
        mean_vector_ndcg=sum(r.vector.ndcg_at_k for r in arm) / len(arm),
        mean_bm25_recall=sum(r.bm25.recall_at_k for r in arm) / len(arm),
        mean_bm25_precision=sum(r.bm25.precision_at_k for r in arm) / len(arm),
        mean_bm25_ndcg=sum(r.bm25.ndcg_at_k for r in arm) / len(arm),
    )


def _format_gate_case_line(r: GateCaseResult) -> str:
    tag = " reported-only" if r.reported_only else ""
    return (
        f"  [{r.corpus}{tag}] {r.name}: {r.query!r}\n"
        f"    vector  recall@{r.k}={r.vector.recall_at_k:.4f} "
        f"prec@{r.k}={r.vector.precision_at_k:.4f} nDCG@{r.k}={r.vector.ndcg_at_k:.4f} "
        f"win={r.vector_wins}\n"
        f"    bm25    recall@{r.k}={r.bm25.recall_at_k:.4f} "
        f"prec@{r.k}={r.bm25.precision_at_k:.4f} nDCG@{r.k}={r.bm25.ndcg_at_k:.4f}"
    )


def format_gate_report(
    results: list[GateCaseResult],
    *,
    library_relevant: int,
    play_relevant: int,
    gating_results: list[GateCaseResult] | None = None,
) -> str:
    gating = gating_results if gating_results is not None else [r for r in results if not r.reported_only]
    k = gating[0].k if gating else (results[0].k if results else "?")
    lines = [
        "Spot retrieval gate (paraphrase queries; vector vs BM25/FTS):",
        f"  library |relevant|={library_relevant} (recall@k = hits/|relevant|; k={k})",
        f"  play_history this_week |relevant|={play_relevant}",
        "",
    ]
    for r in results:
        if not r.reported_only:
            lines.append(_format_gate_case_line(r))
    reported = [r for r in results if r.reported_only]
    if reported:
        lines.append("\n  Library reported-only (not gating):")
        for r in reported:
            lines.append(_format_gate_case_line(r))
    for corpus in ("library", "play_history"):
        arm = [r for r in gating if r.corpus == corpus]
        if not arm:
            continue
        try:
            s = summarize_arm(arm, corpus)
        except ValueError:
            continue
        lines.append(
            f"\n  {corpus} gating summary: wins={s.vector_wins}/{s.query_count} "
            f"mean_vector recall={s.mean_vector_recall:.4f} prec={s.mean_vector_precision:.4f} "
            f"nDCG={s.mean_vector_ndcg:.4f}"
        )
    return "\n".join(lines)
