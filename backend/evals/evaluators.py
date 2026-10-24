"""Deterministic evaluators for the agent eval suite."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from evals.types import AgentTurnResult, EvalCaseMetadata
from spot_backend.reply_grounding import (
    reply_mentions_artists_outside_tool_data,
    reply_mentions_catalog_outside_tool_data,
)


def _meta(ctx: EvaluatorContext) -> EvalCaseMetadata:
    raw = ctx.metadata
    if isinstance(raw, dict):
        return EvalCaseMetadata.model_validate(raw)
    return EvalCaseMetadata(group="unknown")


def _output(ctx: EvaluatorContext) -> AgentTurnResult:
    if isinstance(ctx.output, AgentTurnResult):
        return ctx.output
    if isinstance(ctx.output, dict):
        return AgentTurnResult.model_validate(ctx.output)
    return AgentTurnResult(reply=str(ctx.output or ""))


@dataclass
class ToolSequenceMatch(Evaluator):
    """Tool names (in order) must match metadata.expected_tools when set."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        meta = _meta(ctx)
        if not meta.expected_tools:
            return True
        out = _output(ctx)
        names = [tc.name for tc in out.tool_calls]
        return names == meta.expected_tools


@dataclass
class ToolSequencePrefix(Evaluator):
    """First N tools must match expected_tool_prefix."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        meta = _meta(ctx)
        if not meta.expected_tool_prefix:
            return True
        out = _output(ctx)
        names = [tc.name for tc in out.tool_calls]
        pref = meta.expected_tool_prefix
        if len(names) < len(pref):
            return False
        return names[: len(pref)] == pref


@dataclass
class ForbiddenTools(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        meta = _meta(ctx)
        if not meta.forbidden_tools:
            return True
        out = _output(ctx)
        used = {tc.name for tc in out.tool_calls}
        return not used.intersection(meta.forbidden_tools)


@dataclass
class ShortcutRouting(Evaluator):
    """Deterministic shortcuts must not invoke the LLM."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        meta = _meta(ctx)
        if not meta.expect_shortcut:
            return True
        out = _output(ctx)
        return out.shortcut and not out.llm_invoked


@dataclass
class HitAt1(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        meta = _meta(ctx)
        if not meta.accepted_uris:
            return True
        out = _output(ctx)
        if not out.acted_uris:
            return False
        first = out.acted_uris[0]
        return first in meta.accepted_uris


@dataclass
class CaseMaxLatency(Evaluator):
    """Per-case latency ceiling from metadata.max_latency_s."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        meta = _meta(ctx)
        if meta.max_latency_s is None:
            return True
        out = _output(ctx)
        return out.duration_s <= float(meta.max_latency_s)


@dataclass
class ReplyGrounding(Evaluator):
    """No quoted title-by-artist credits outside tool JSON."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        meta = _meta(ctx)
        if not meta.expect_grounded:
            return True
        out = _output(ctx)
        results = [tc.result for tc in out.tool_calls if tc.result]
        bad = reply_mentions_artists_outside_tool_data(out.reply, results)
        bad.extend(reply_mentions_catalog_outside_tool_data(out.reply, results))
        return len(bad) == 0


@dataclass
class ToolErrorHandled(Evaluator):
    """When expect_tool_error, reply should mention failure without claiming success."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        meta = _meta(ctx)
        if not meta.expect_tool_error:
            return True
        out = _output(ctx)
        low = (out.reply or "").lower()
        if not low:
            return bool(out.tool_calls)
        success_claims = ("now playing", "added to", "queued", "playing ")
        if any(s in low for s in success_claims):
            return False
        return any(w in low for w in ("rate", "limit", "error", "couldn't", "could not", "unable", "device"))


@dataclass
class MoodIntentJudge(Evaluator):
    """LLM mood rubric only for cases with metadata.mood_request=true (live+judge runs)."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        meta = _meta(ctx)
        if not meta.mood_request:
            return True
        # Mock CI and judge-disabled runs skip qualitative scoring.
        if not (os.environ.get("EVAL_JUDGE_MODEL") or os.environ.get("EVAL_MOOD_JUDGE") == "1"):
            return True
        import asyncio

        from evals.judges import mood_intent_judge

        inner = mood_intent_judge()
        coro = inner.evaluate(ctx)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return bool(asyncio.run(coro))
        return True


@dataclass
class MockScriptRan(Evaluator):
    """Mock cases with mock_script must execute at least one tool."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        inp = ctx.inputs
        script_len = 0
        if isinstance(inp, dict):
            script_len = len(inp.get("mock_script") or [])
        out = _output(ctx)
        if script_len == 0:
            return True
        mode = inp.get("mode") if isinstance(inp, dict) else None
        if mode == "live":
            return True
        return len(out.tool_calls) >= 1


CUSTOM_EVALUATOR_TYPES: list[type[Evaluator]] = [
    ToolSequenceMatch,
    ToolSequencePrefix,
    ForbiddenTools,
    ShortcutRouting,
    HitAt1,
    CaseMaxLatency,
    ReplyGrounding,
    ToolErrorHandled,
    MockScriptRan,
    MoodIntentJudge,
]


def dataset_evaluators(include_judges: bool = True) -> list[Any]:
    from pydantic_evals.evaluators import MaxDuration

    from evals.judges import qualitative_judges

    evals: list[Any] = [
        MaxDuration(seconds=120.0),
        ToolSequenceMatch(),
        ToolSequencePrefix(),
        ForbiddenTools(),
        ShortcutRouting(),
        HitAt1(),
        CaseMaxLatency(),
        ReplyGrounding(),
        ToolErrorHandled(),
        MockScriptRan(),
        MoodIntentJudge(),
    ]
    if include_judges:
        evals.extend(qualitative_judges())
    return evals
