"""Deterministic evaluators for the agent eval suite."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

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
        parts = [out.reply or ""]
        parts.extend(tc.result or "" for tc in out.tool_calls)
        low = "\n".join(parts).lower()
        if not low.strip():
            return bool(out.tool_calls)
        success_claims = ("now playing", "added to", "queued", "playing ")
        if any(s in low for s in success_claims):
            return False
        return any(
            w in low
            for w in (
                "rate",
                "limit",
                "error",
                "couldn't",
                "could not",
                "unable",
                "device",
                "429",
                "forbidden",
                "must be set",
            )
        )


@dataclass
class ToolJsonValid(Evaluator):
    """Tool results should be JSON objects when present (schema sanity)."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        out = _output(ctx)
        for tc in out.tool_calls:
            raw = (tc.result or "").strip()
            if not raw:
                continue
            if raw[0] not in "{[":
                # Live agent events store truncated previews, not full JSON payloads.
                continue
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                if os.environ.get("EVAL_JUDGE_MODEL"):
                    continue
                return False
            if not isinstance(data, dict):
                return False
        return True


@dataclass
class MoodIntentJudge(Evaluator):
    """LLM mood rubric only for cases with metadata.mood_request=true (live+judge runs)."""

    def evaluate(self, ctx: EvaluatorContext) -> bool | EvaluationReason:
        meta = _meta(ctx)
        if not meta.mood_request:
            return True
        if not (os.environ.get("EVAL_JUDGE_MODEL") or os.environ.get("EVAL_MOOD_JUDGE") == "1"):
            return EvaluationReason(
                value="skipped",
                reason="no EVAL_JUDGE_MODEL configured",
            )
        from evals.judges import _judge_passed, _run_judge_eval_sync, mood_intent_judge

        return _judge_passed(_run_judge_eval_sync(mood_intent_judge(), ctx))


@dataclass
class BlankTrackTitleRejected(Evaluator):
    """Playback tools must reject blank track or artist titles (no silent catalog guesses)."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        out = _output(ctx)
        for tc in out.tool_calls:
            if tc.name not in (
                "spotify_play_track",
                "spotify_add_to_queue",
                "spotify_play_next",
            ):
                continue
            args = tc.arguments or {}
            title_keys = ("track_name", "title", "name")
            artist_keys = ("artist_name", "artist")
            if not any(k in args for k in title_keys) and not any(k in args for k in artist_keys):
                # Live eval records tool_done previews only (arguments={}); cannot judge blanks.
                continue
            title = str(args.get("track_name") or args.get("title") or args.get("name") or "").strip()
            artist = str(args.get("artist_name") or args.get("artist") or "").strip()
            raw = tc.result or ""
            raw_l = raw.lower()
            if title == "" and artist == "":
                if "is required" not in raw_l and "track_name is required" not in raw_l:
                    return False
                continue
            if title == "":
                if "is required" in raw_l:
                    continue
                compact = raw.replace(" ", "")
                if '"ok":true' in compact or '"ok": true' in raw:
                    return False
                if '"error"' in raw and "is required" not in raw_l:
                    return False
                continue
            if artist == "":
                # Bare-track play: title required; artist may be omitted when search can resolve it.
                continue
        return True


@dataclass
class LibraryOnlyActs(Evaluator):
    """Canonical SAMPLE cases: played URIs must exist in the committed fixture library."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        meta = _meta(ctx)
        if not meta.canonical_library and not meta.library_only:
            return True
        from evals.eval_library_fixture import all_track_uris

        allowed = all_track_uris()
        out = _output(ctx)
        if not out.acted_uris:
            return True
        return all(u in allowed for u in out.acted_uris)


@dataclass
class LovedPickShare(Evaluator):
    """Share of acted URIs from top-loved tier (canonical SAMPLE sunset suite)."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        meta = _meta(ctx)
        if not meta.canonical_library:
            return True
        from evals.eval_library_fixture import loved_pick_share

        out = _output(ctx)
        share = loved_pick_share(out.acted_uris)
        return share >= 0.15


@dataclass
class CanonicalVibeJudge(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        meta = _meta(ctx)
        if not meta.canonical_library:
            return True
        if not (os.environ.get("EVAL_JUDGE_MODEL") or os.environ.get("EVAL_MOOD_JUDGE") == "1"):
            return True
        from evals.judges import _judge_passed, _run_judge_eval_sync, canonical_fits_vibe_judge

        return _judge_passed(_run_judge_eval_sync(canonical_fits_vibe_judge(), ctx))


@dataclass
class CanonicalTasteJudge(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        meta = _meta(ctx)
        if not meta.canonical_library:
            return True
        if not (os.environ.get("EVAL_JUDGE_MODEL") or os.environ.get("EVAL_MOOD_JUDGE") == "1"):
            return True
        from evals.judges import _judge_passed, _run_judge_eval_sync, canonical_fits_taste_judge

        return _judge_passed(_run_judge_eval_sync(canonical_fits_taste_judge(), ctx))


@dataclass
class NoMockHttpLeakInReply(Evaluator):
    """Fail when respx AllMockedAssertionError text leaks into the user-visible reply."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        out = _output(ctx)
        blob = (out.reply or "") + "\n".join((tc.result or "") for tc in out.tool_calls)
        if "AllMockedAssertionError" in blob:
            return False
        if "RESPX:" in blob and "not mocked" in blob:
            return False
        return True


@dataclass
class PlayHistoryWindowMatch(Evaluator):
    """search_play_history must use the expected window argument when set."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        meta = _meta(ctx)
        want = (meta.expect_play_history_window or "").strip()
        if not want:
            return True
        out = _output(ctx)
        for tc in out.tool_calls:
            if tc.name != "search_play_history":
                continue
            got = str(tc.arguments.get("window") or "").strip().lower()
            if got == want.lower():
                return True
        return False


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
    BlankTrackTitleRejected,
    ToolJsonValid,
    LibraryOnlyActs,
    LovedPickShare,
    CanonicalVibeJudge,
    CanonicalTasteJudge,
    NoMockHttpLeakInReply,
    PlayHistoryWindowMatch,
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
        BlankTrackTitleRejected(),
        ToolJsonValid(),
        LibraryOnlyActs(),
        LovedPickShare(),
        CanonicalVibeJudge(),
        CanonicalTasteJudge(),
        NoMockHttpLeakInReply(),
        PlayHistoryWindowMatch(),
    ]
    if include_judges:
        evals.extend(qualitative_judges())
    return evals
