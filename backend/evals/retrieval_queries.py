"""Eval / test queries with committed query embeddings (see scripts/regenerate_eval_embeddings.py)."""

from __future__ import annotations

from dataclasses import dataclass

# Library index (SAMPLE eval_library fixture) — literal + integration tests
LIBRARY_SUNSET_LITERAL = "sunset vibes I've loved"

# Play history — literals / latency tests
PLAY_SUNSET_THIS_WEEK_LITERAL = "sunset"
PLAY_GOLDEN_HOUR = "golden hour"
PLAY_DUSK_DRIVE = "dusk drive"

# Paraphrase gate queries: no FTS token overlap with planted title/artist text (see gate test).
LIBRARY_PARAPHRASE_QUERIES: tuple[str, ...] = (
    "warm coastal evening music from my collection",
    "mellow amber twilight songs I have saved",
    "warm amber twilight listening from my library",
    "soft evening glow tracks I have saved",
    "coastal twilight calm I have enjoyed",
    "quiet twilight warmth from my saved music",
    "tender sundown warmth from songs I saved",
    "coral twilight ambience in my music library",
    "hazy evening glow listening I have kept",
)

# Original shoreline/ocean phrasing — scored in gate test output only (not gating).
LIBRARY_REPORTED_ONLY_PARAPHRASE_QUERIES: tuple[str, ...] = (
    "relaxed shoreline melodies I have enjoyed",
    "quiet shoreline pieces from my saved collection",
    "balmy ocean breeze music from my collection",
)

PLAY_HISTORY_THIS_WEEK_PARAPHRASE_QUERIES: tuple[str, ...] = (
    "twilight warmth I replayed in recent days",
    "mellow amber twilight spins from the last few days",
    "quiet shoreline listening this week",
    "balmy evening playback from my recent history",
    "hazy horizon songs I heard lately this week",
    "coral sky ambience from recent playback",
    "twilight ambience I replayed in recent days",
    "ocean breeze listens from the last seven days",
    "amber twilight warmth I replayed in recent days",
)

LIBRARY_COMMITTED_QUERIES: tuple[str, ...] = (
    LIBRARY_SUNSET_LITERAL,
    *LIBRARY_PARAPHRASE_QUERIES,
    *LIBRARY_REPORTED_ONLY_PARAPHRASE_QUERIES,
)

PLAY_HISTORY_COMMITTED_QUERIES: tuple[str, ...] = (
    PLAY_SUNSET_THIS_WEEK_LITERAL,
    *PLAY_HISTORY_THIS_WEEK_PARAPHRASE_QUERIES,
    PLAY_GOLDEN_HOUR,
    PLAY_DUSK_DRIVE,
)


@dataclass(frozen=True)
class RetrievalGateCase:
    name: str
    query: str
    corpus: str  # "library" | "play_history"
    window: str = "this_week"
    reported_only: bool = False


def library_gate_cases() -> tuple[RetrievalGateCase, ...]:
    return tuple(
        RetrievalGateCase(
            name=f"library_paraphrase_{i + 1:02d}",
            query=q,
            corpus="library",
        )
        for i, q in enumerate(LIBRARY_PARAPHRASE_QUERIES)
    )


def play_history_gate_cases() -> tuple[RetrievalGateCase, ...]:
    return tuple(
        RetrievalGateCase(
            name=f"play_this_week_paraphrase_{i + 1:02d}",
            query=q,
            corpus="play_history",
            window="this_week",
        )
        for i, q in enumerate(PLAY_HISTORY_THIS_WEEK_PARAPHRASE_QUERIES)
    )


def library_reported_only_gate_cases() -> tuple[RetrievalGateCase, ...]:
    return tuple(
        RetrievalGateCase(
            name=f"library_reported_shoreline_{i + 1:02d}",
            query=q,
            corpus="library",
            reported_only=True,
        )
        for i, q in enumerate(LIBRARY_REPORTED_ONLY_PARAPHRASE_QUERIES)
    )
