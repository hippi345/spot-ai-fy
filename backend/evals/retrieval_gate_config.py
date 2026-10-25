"""Thresholds for offline Spot retrieval gate (paraphrase vs BM25/FTS)."""

from __future__ import annotations

# Library fixture: 50 planted sunset loved tracks (see eval_library.json meta).
LIBRARY_PLANTED_RELEVANT_COUNT = 50
LIBRARY_GATE_K = 20  # library_index.search caps at 25

# Play fixture: 15 URIs in planted.sunset_this_week (see play_history_seed.json).
PLAY_HISTORY_THIS_WEEK_RELEVANT_COUNT = 15
PLAY_HISTORY_GATE_K = 20

# Per-arm paraphrase query count (8–10) and majority win rule.
MIN_PARAPHRASE_QUERIES_PER_ARM = 8
MIN_VECTOR_WINS_PER_ARM = 6

# Library: |relevant|=50 with k=20 ⇒ recall@20 is hits/50 (max 0.40). Gate precision + nDCG.
# Floors ≈ half of observed gate means (~0.30 precision, ~0.30 nDCG on 9 paraphrases).
LIBRARY_MEAN_PRECISION_AT_K_FLOOR = 0.15
LIBRARY_MEAN_NDCG_AT_K_FLOOR = 0.15

# Play history: 15 relevant URIs in this_week window (recall@20 is meaningful).
PLAY_HISTORY_MEAN_RECALL_AT_K_FLOOR = 0.33
