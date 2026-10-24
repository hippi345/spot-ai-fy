from __future__ import annotations

from evals.evaluators import (
    HitAt1,
    ReplyGrounding,
    ShortcutRouting,
    ToolSequenceMatch,
)
from evals.types import AgentTurnResult, EvalCaseMetadata, ToolCallRecord
from pydantic_evals.evaluators import EvaluatorContext


def _ctx(
    output: AgentTurnResult,
    metadata: dict,
    *,
    inputs: dict | None = None,
) -> EvaluatorContext:
    return EvaluatorContext(
        name="t",
        inputs=inputs or {"prompt": "x"},
        metadata=metadata,
        expected_output=None,
        output=output,
        duration=output.duration_s,
        _span_tree=None,
        attributes={},
        metrics={},
    )


def test_tool_sequence_match() -> None:
    out = AgentTurnResult(
        tool_calls=[ToolCallRecord(name="spotify_pause", arguments={}, result="{}")],
    )
    ev = ToolSequenceMatch()
    assert ev.evaluate(_ctx(out, {"expected_tools": ["spotify_pause"]}))
    assert not ev.evaluate(_ctx(out, {"expected_tools": ["spotify_skip_next"]}))


def test_shortcut_routing() -> None:
    out = AgentTurnResult(shortcut=True, llm_invoked=False, tool_calls=[ToolCallRecord(name="spotify_pause")])
    assert ShortcutRouting().evaluate(_ctx(out, {"expect_shortcut": True}))


def test_hit_at_1() -> None:
    out = AgentTurnResult(acted_uris=["spotify:track:abc"])
    meta = {"accepted_uris": ["spotify:track:abc", "spotify:track:def"]}
    assert HitAt1().evaluate(_ctx(out, meta))


def test_reply_grounding_passes_when_grounded() -> None:
    raw = '{"artists": [{"name": "Jay Sean"}], "tracks": {"items": [{"name": "Down", "artists": [{"name": "Jay Sean"}]}]}}'
    out = AgentTurnResult(
        reply='"Down" by Jay Sean is now playing.',
        tool_calls=[ToolCallRecord(name="spotify_play_track", result=raw)],
    )
    assert ReplyGrounding().evaluate(_ctx(out, {"expect_grounded": True}))
