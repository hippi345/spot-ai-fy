"""Retrieval metrics for SAMPLE library fixture integration tests."""

from __future__ import annotations

import math
import time
from typing import Callable


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return len(set(top) & relevant) / len(relevant)


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    return len(set(top) & relevant) / k


def ndcg_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    top = retrieved[:k]
    dcg = 0.0
    for i, item in enumerate(top):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def percentile_ms(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
    return ordered[idx] * 1000.0


def bench_search(fn: Callable[[], dict], *, n: int = 100) -> tuple[float, list[float]]:
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return percentile_ms(times, 95), times
