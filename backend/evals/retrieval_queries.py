"""Eval / test queries with committed query embeddings (see scripts/regenerate_eval_embeddings.py)."""

from __future__ import annotations

from dataclasses import dataclass

# Library index (SAMPLE eval_library fixture)
LIBRARY_SUNSET_LITERAL = "sunset vibes I've loved"
LIBRARY_SUNSET_PARAPHRASE = "warm coastal evening music from my collection"

# Play history fixture — paraphrase must not share FTS tokens with planted sunset-this-week titles.
PLAY_SUNSET_THIS_WEEK_LITERAL = "sunset"
PLAY_SUNSET_THIS_WEEK_PARAPHRASE = "twilight warmth I replayed in recent days"

# Other committed queries used in tests
PLAY_GOLDEN_HOUR = "golden hour"
PLAY_DUSK_DRIVE = "dusk drive"


@dataclass(frozen=True)
class JukeGateCase:
    name: str
    query: str
    corpus: str  # "library" | "play_history"
    window: str = "this_week"


LIBRARY_JUKE = JukeGateCase(
    name="library_sunset_paraphrase",
    query=LIBRARY_SUNSET_PARAPHRASE,
    corpus="library",
)

PLAY_HISTORY_JUKE = JukeGateCase(
    name="play_history_sunset_this_week_paraphrase",
    query=PLAY_SUNSET_THIS_WEEK_PARAPHRASE,
    corpus="play_history",
    window="this_week",
)

JUKE_GATE_CASES: tuple[JukeGateCase, ...] = (LIBRARY_JUKE, PLAY_HISTORY_JUKE)

LIBRARY_COMMITTED_QUERIES: tuple[str, ...] = (
    LIBRARY_SUNSET_LITERAL,
    LIBRARY_SUNSET_PARAPHRASE,
)

PLAY_HISTORY_COMMITTED_QUERIES: tuple[str, ...] = (
    PLAY_SUNSET_THIS_WEEK_LITERAL,
    PLAY_SUNSET_THIS_WEEK_PARAPHRASE,
    PLAY_GOLDEN_HOUR,
    PLAY_DUSK_DRIVE,
)
