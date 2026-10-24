"""Unit tests for library index dual-arm live eval helpers."""

from __future__ import annotations

from types import SimpleNamespace

from evals.library_dual_arm import (
    base_case_name,
    compare_library_arms,
    library_hit_share,
    with_library_arm,
)
from evals.baseline import _aggregate_report
from pydantic_evals import Case, Dataset


def test_with_library_arm_renames_and_forces_env_input() -> None:
    ds = Dataset(
        name="t",
        cases=[
            Case(
                name="mood_sad_acoustic",
                inputs={"prompt": "x", "mode": "live", "library_index_arm": "on"},
                metadata={"live": True},
            )
        ],
        evaluators=[],
    )
    off = with_library_arm(ds, "off")
    assert off.cases[0].name == "mood_sad_acoustic@lib_off"
    assert off.cases[0].inputs["library_index_arm"] == "off"
    on = with_library_arm(ds, "on")
    assert base_case_name(on.cases[0].name) == "mood_sad_acoustic"


def _fake_report(cases: list[tuple[str, list[str], dict[str, float], dict[str, bool]]]):
    """Build minimal report-like object for aggregate + comparison tests."""

    class Res:
        def __init__(self, value):
            self.value = value

    report_cases = []
    for name, tools, scores, assertions in cases:
        report_cases.append(
            SimpleNamespace(
                name=name,
                output={"reply": "ok", "tool_calls": [{"name": t, "arguments": {}, "result": ""} for t in tools]},
                scores={k: Res(v) for k, v in scores.items()},
                assertions={k: Res(v) for k, v in assertions.items()},
            )
        )
    return SimpleNamespace(cases=report_cases)


def test_library_hit_share_counts_search_my_library() -> None:
    rep = _fake_report(
        [
            ("a@lib_on", ["search_my_library"], {}, {}),
            ("b@lib_on", ["spotify_search"], {}, {}),
        ]
    )
    assert library_hit_share(rep) == 0.5


def test_compare_verdict_helps_on_rubric_gain() -> None:
    off = _fake_report(
        [
            (
                "c@lib_off",
                [],
                {"accurate_summary_score": 0.4, "concise_friendly_score": 0.8, "no_false_claims_score": 0.7},
                {"ReplyGrounding": True, "ToolJsonValid": True},
            )
        ]
    )
    on = _fake_report(
        [
            (
                "c@lib_on",
                ["search_my_library"],
                {"accurate_summary_score": 0.6, "concise_friendly_score": 0.85, "no_false_claims_score": 0.75},
                {"ReplyGrounding": True, "ToolJsonValid": True},
            )
        ]
    )
    # Wrap so _aggregate_report works — it expects EvaluationReport cases; use same shape
    from pydantic_evals.reporting import EvaluationReport

    def to_report(fake):
        # EvaluationReport is a dataclass; build via evaluate is heavy — use compare on fakes
        # by patching: compare uses _aggregate_report which only needs .cases with scores/assertions
        return fake

    comp = compare_library_arms(to_report(off), to_report(on))
    assert comp.verdict == "helps"
    assert comp.library_hit_share_on == 1.0
    assert comp.library_hit_share_off == 0.0


def test_compare_verdict_hurts_on_grounding_regression() -> None:
    off = _fake_report(
        [
            (
                "c@lib_off",
                [],
                {},
                {"ReplyGrounding": True, "ToolJsonValid": True},
            ),
            (
                "d@lib_off",
                [],
                {},
                {"ReplyGrounding": True, "ToolJsonValid": True},
            ),
        ]
    )
    on = _fake_report(
        [
            (
                "c@lib_on",
                [],
                {},
                {"ReplyGrounding": False, "ToolJsonValid": True},
            ),
            (
                "d@lib_on",
                [],
                {},
                {"ReplyGrounding": True, "ToolJsonValid": True},
            ),
        ]
    )
    comp = compare_library_arms(off, on)
    assert comp.verdict == "hurts"
    assert "ReplyGrounding" in comp.verdict_reason
