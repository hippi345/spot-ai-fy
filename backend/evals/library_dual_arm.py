"""Live eval library index ON/OFF dual-arm runs and comparison reporting."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_evals import Case, Dataset
from pydantic_evals.reporting import EvaluationReport

from evals.baseline import _aggregate_report
from evals.types import AgentTurnResult

_ARM_SUFFIX_RE = re.compile(r"@lib_(on|off)$")
_LIBRARY_TOOL = "search_my_library"
_RUBRIC_KEYS = (
    "accurate_summary_score",
    "concise_friendly_score",
    "no_false_claims_score",
)
_HARD_COMPARE_KEYS = ("ReplyGrounding", "ToolJsonValid")


def base_case_name(name: str) -> str:
    return _ARM_SUFFIX_RE.sub("", name)


def with_library_arm(dataset: Dataset, arm: str) -> Dataset:
    """Clone live cases with a forced library index arm (on|off)."""
    a = arm.strip().lower()
    if a not in ("on", "off"):
        raise ValueError(f"library arm must be on|off, got {arm!r}")
    cases: list[Case] = []
    for c in dataset.cases:
        inputs = copy.deepcopy(c.inputs)
        if isinstance(inputs, dict):
            inputs["library_index_arm"] = a
        meta = copy.deepcopy(c.metadata) if isinstance(c.metadata, dict) else {}
        if isinstance(meta, dict):
            meta["library_index_arm"] = a
        cases.append(
            Case(
                name=f"{base_case_name(c.name)}@lib_{a}",
                inputs=inputs,
                metadata=meta,
            )
        )
    return Dataset(
        name=f"{dataset.name}_lib_{a}",
        cases=cases,
        evaluators=list(dataset.evaluators),
        report_evaluators=list(getattr(dataset, "report_evaluators", []) or []),
    )


def _parse_output(case: Any) -> AgentTurnResult | None:
    out = getattr(case, "output", None)
    if out is None:
        return None
    if isinstance(out, AgentTurnResult):
        return out
    try:
        return AgentTurnResult.model_validate(out)
    except Exception:
        return None


def case_used_library_tool(case: Any) -> bool:
    out = _parse_output(case)
    if not out:
        return False
    return any((tc.name or "").strip() == _LIBRARY_TOOL for tc in out.tool_calls)


def library_hit_share(report: EvaluationReport) -> float:
    if not report.cases:
        return 0.0
    hits = sum(1 for c in report.cases if case_used_library_tool(c))
    return hits / len(report.cases)


def _assertion_rate(report: EvaluationReport, name: str) -> float | None:
    agg = _aggregate_report(report)
    val = (agg.get("assertion_pass_rate") or {}).get(name)
    if val is None:
        return None
    return float(val)


def _score_mean(report: EvaluationReport, name: str) -> float | None:
    agg = _aggregate_report(report)
    val = (agg.get("score_means") or {}).get(name)
    if val is None:
        return None
    return float(val)


def _paired_cases(off_report: EvaluationReport, on_report: EvaluationReport) -> list[tuple[str, Any, Any]]:
    off_by = {base_case_name(c.name): c for c in off_report.cases}
    on_by = {base_case_name(c.name): c for c in on_report.cases}
    names = sorted(set(off_by) & set(on_by))
    return [(n, off_by[n], on_by[n]) for n in names]


Verdict = Literal["helps", "neutral", "hurts"]


@dataclass
class RubricDelta:
    name: str
    off_mean: float | None
    on_mean: float | None
    delta: float | None


@dataclass
class LibraryArmComparison:
    paired_case_count: int
    library_hit_share_off: float
    library_hit_share_on: float
    rubric_deltas: list[RubricDelta] = field(default_factory=list)
    hard_metric_deltas: dict[str, dict[str, float | None]] = field(default_factory=dict)
    verdict: Verdict = "neutral"
    verdict_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "paired_case_count": self.paired_case_count,
            "library_hit_share": {"off": self.library_hit_share_off, "on": self.library_hit_share_on},
            "rubric_deltas": [
                {
                    "name": r.name,
                    "off": r.off_mean,
                    "on": r.on_mean,
                    "delta_on_minus_off": r.delta,
                }
                for r in self.rubric_deltas
            ],
            "hard_metrics": self.hard_metric_deltas,
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
        }


def compare_library_arms(
    off_report: EvaluationReport,
    on_report: EvaluationReport,
    *,
    rubric_help_delta: float = 0.05,
    rubric_hurt_delta: float = -0.10,
    hard_regression_tolerance: float = 0.05,
) -> LibraryArmComparison:
    paired = _paired_cases(off_report, on_report)
    comp = LibraryArmComparison(
        paired_case_count=len(paired),
        library_hit_share_off=library_hit_share(off_report),
        library_hit_share_on=library_hit_share(on_report),
    )

    for key in _RUBRIC_KEYS:
        off_m = _score_mean(off_report, key)
        on_m = _score_mean(on_report, key)
        delta = None
        if off_m is not None and on_m is not None:
            delta = on_m - off_m
        comp.rubric_deltas.append(RubricDelta(name=key, off_mean=off_m, on_mean=on_m, delta=delta))

    for key in _HARD_COMPARE_KEYS:
        off_r = _assertion_rate(off_report, key)
        on_r = _assertion_rate(on_report, key)
        comp.hard_metric_deltas[key] = {
            "off": off_r,
            "on": on_r,
            "delta_on_minus_off": (on_r - off_r) if off_r is not None and on_r is not None else None,
        }

    rubric_deltas = [r.delta for r in comp.rubric_deltas if r.delta is not None]
    mean_rubric_delta = sum(rubric_deltas) / len(rubric_deltas) if rubric_deltas else 0.0

    grounding = comp.hard_metric_deltas.get("ReplyGrounding", {})
    schema = comp.hard_metric_deltas.get("ToolJsonValid", {})
    g_off, g_on = grounding.get("off"), grounding.get("on")
    s_off, s_on = schema.get("off"), schema.get("on")

    hurts_reasons: list[str] = []
    if g_off is not None and g_on is not None and g_on + 1e-9 < g_off - hard_regression_tolerance:
        hurts_reasons.append(
            f"ReplyGrounding regressed ({g_on:.3f} vs off {g_off:.3f}, tol {hard_regression_tolerance})"
        )
    if s_off is not None and s_on is not None and s_on + 1e-9 < s_off - hard_regression_tolerance:
        hurts_reasons.append(
            f"ToolJsonValid regressed ({s_on:.3f} vs off {s_off:.3f}, tol {hard_regression_tolerance})"
        )
    if rubric_deltas and mean_rubric_delta <= rubric_hurt_delta:
        hurts_reasons.append(f"mean rubric delta {mean_rubric_delta:.3f} <= {rubric_hurt_delta}")

    if hurts_reasons:
        comp.verdict = "hurts"
        comp.verdict_reason = "; ".join(hurts_reasons)
        return comp

    helps_reasons: list[str] = []
    if rubric_deltas and mean_rubric_delta >= rubric_help_delta:
        helps_reasons.append(f"mean rubric delta {mean_rubric_delta:.3f} >= {rubric_help_delta}")
    if comp.library_hit_share_on > comp.library_hit_share_off + 0.01:
        helps_reasons.append(
            f"library-hit share up ({comp.library_hit_share_on:.2%} vs {comp.library_hit_share_off:.2%})"
        )

    if helps_reasons and not hurts_reasons:
        comp.verdict = "helps"
        comp.verdict_reason = "; ".join(helps_reasons)
    else:
        comp.verdict = "neutral"
        comp.verdict_reason = "No meaningful rubric gain and no grounding/schema regression."

    return comp


def format_library_arm_comparison_markdown(comp: LibraryArmComparison) -> str:
    lines = [
        "# Library index ON vs OFF (live eval)",
        "",
        f"**Verdict:** index **{comp.verdict}** — {comp.verdict_reason}",
        "",
        f"Paired cases: {comp.paired_case_count}",
        "",
        "## Library-hit share (`search_my_library` used)",
        "",
        f"| Arm | Share |",
        f"|-----|-------|",
        f"| OFF (default) | {comp.library_hit_share_off:.1%} |",
        f"| ON | {comp.library_hit_share_on:.1%} |",
        "",
        "## Per-rubric means (ON − OFF)",
        "",
        "| Rubric | OFF | ON | Δ |",
        "|--------|-----|----|---|",
    ]
    for r in comp.rubric_deltas:
        off_s = f"{r.off_mean:.3f}" if r.off_mean is not None else "n/a"
        on_s = f"{r.on_mean:.3f}" if r.on_mean is not None else "n/a"
        d_s = f"{r.delta:+.3f}" if r.delta is not None else "n/a"
        lines.append(f"| {r.name} | {off_s} | {on_s} | {d_s} |")

    lines.extend(["", "## Grounding / schema pass rates", "", "| Metric | OFF | ON | Δ |", "|--------|-----|----|---|"])
    for key, vals in comp.hard_metric_deltas.items():
        off_v = vals.get("off")
        on_v = vals.get("on")
        d_v = vals.get("delta_on_minus_off")
        off_s = f"{off_v:.3f}" if off_v is not None else "n/a"
        on_s = f"{on_v:.3f}" if on_v is not None else "n/a"
        d_s = f"{d_v:+.3f}" if d_v is not None else "n/a"
        lines.append(f"| {key} | {off_s} | {on_s} | {d_s} |")

    lines.extend(
        [
            "",
            "_Threshold policy:_ OFF arm is fully gated by `thresholds.yaml` (`live_off`). "
            "ON arm must complete without crashes; rubric deltas are informational unless "
            "grounding or schema regresses vs OFF.",
            "",
        ]
    )
    return "\n".join(lines)


def check_on_arm_regressions(
    comp: LibraryArmComparison,
    *,
    hard_regression_tolerance: float = 0.05,
) -> list[str]:
    failures: list[str] = []
    for key in _HARD_COMPARE_KEYS:
        vals = comp.hard_metric_deltas.get(key, {})
        off_v, on_v = vals.get("off"), vals.get("on")
        if off_v is None or on_v is None:
            continue
        if on_v + 1e-9 < float(off_v) - hard_regression_tolerance:
            failures.append(
                f"ON arm {key} regressed: {on_v:.3f} < OFF {off_v:.3f} - {hard_regression_tolerance}"
            )
    return failures


def write_comparison_artifacts(
    comp: LibraryArmComparison,
    *,
    json_path: Any,
    md_path: Any,
) -> None:
    json_path.write_text(json.dumps(comp.to_dict(), indent=2) + "\n", encoding="utf-8")
    md_path.write_text(format_library_arm_comparison_markdown(comp), encoding="utf-8")
