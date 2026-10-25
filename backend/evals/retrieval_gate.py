"""Juke-style offline retrieval gate (vector vs BM25/FTS, same tokenizer)."""

from __future__ import annotations

import re
from dataclasses import dataclass

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
    q = fts_query_tokens(query)
    p = planted_corpus_tokens(*planted_texts)
    overlap = q & p
    if overlap:
        raise AssertionError(f"query/planted token overlap not allowed: {sorted(overlap)}")


@dataclass
class GateArmMetrics:
    recall_at_k: float
    ndcg_at_k: float


@dataclass
class GateCaseResult:
    name: str
    query: str
    k: int
    vector: GateArmMetrics
    bm25: GateArmMetrics
    margin_recall: float
    margin_ndcg: float
    vector_wins: bool


def format_gate_report(results: list[GateCaseResult]) -> str:
    lines = ["Juke retrieval gate (vector vs BM25/FTS, k={}):".format(results[0].k if results else "?")]
    for r in results:
        lines.append(
            f"  {r.name}: query={r.query!r}\n"
            f"    vector  recall@{r.k}={r.vector.recall_at_k:.4f} nDCG@{r.k}={r.vector.ndcg_at_k:.4f}\n"
            f"    bm25    recall@{r.k}={r.bm25.recall_at_k:.4f} nDCG@{r.k}={r.bm25.ndcg_at_k:.4f}\n"
            f"    margin  recall={r.margin_recall:+.4f} nDCG={r.margin_ndcg:+.4f} wins={r.vector_wins}"
        )
    agg_ndcg = sum(r.margin_ndcg for r in results) / max(1, len(results))
    agg_recall = sum(r.margin_recall for r in results) / max(1, len(results))
    lines.append(f"  aggregate margin: recall={agg_recall:+.4f} nDCG={agg_ndcg:+.4f}")
    return "\n".join(lines)
