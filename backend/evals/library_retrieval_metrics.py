"""Retrieval metrics for SAMPLE library fixture integration tests."""

from __future__ import annotations

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
