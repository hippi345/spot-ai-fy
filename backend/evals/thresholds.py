"""Load thresholds.yaml and gate eval reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic_evals.reporting import EvaluationReport

from evals.baseline import BASELINE_LIVE_PATH, _aggregate_report, regression_diff
from evals.evaluators import EvalCaseMetadata
from evals.key_leak_scan import find_key_leaks

THRESHOLDS_PATH = Path(__file__).with_name("thresholds.yaml")


def load_thresholds(path: Path | None = None) -> dict[str, Any]:
    p = path or THRESHOLDS_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid thresholds: {p}")
    return data


@dataclass
class ThresholdRow:
    check: str
    target: str
    actual: str
    ok: bool


@dataclass
class ThresholdResult:
    ok: bool
    rows: list[ThresholdRow] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def _meta(case: Any) -> EvalCaseMetadata:
    raw = getattr(case, "metadata", None)
    if isinstance(raw, dict):
        return EvalCaseMetadata.model_validate(raw)
    return EvalCaseMetadata(group="unknown")


def _score_value(scores: dict[str, Any], key: str) -> float | None:
    raw = scores.get(key)
    if raw is None:
        return None
    if hasattr(raw, "value"):
        try:
            return float(raw.value)
        except (TypeError, ValueError):
            return None
    if isinstance(raw, dict) and "value" in raw:
        try:
            return float(raw["value"])
        except (TypeError, ValueError):
            return None
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _add_row(result: ThresholdResult, check: str, target: str, actual: Any, ok: bool) -> None:
    result.rows.append(ThresholdRow(check=check, target=str(target), actual=str(actual), ok=ok))
    if not ok:
        result.failures.append(f"{check}: actual {actual} vs target {target}")


def _threshold_section(thresholds: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "mock":
        section = thresholds.get("mock_ci")
    elif mode in ("live", "live_off"):
        section = thresholds.get("live_off") or thresholds.get("live")
    elif mode == "live_on":
        section = thresholds.get("live_on")
    else:
        section = thresholds.get(mode)
    if not isinstance(section, dict):
        raise ValueError(f"No thresholds for mode={mode!r}")
    return section


def check_thresholds(
    report: EvaluationReport,
    thresholds: dict[str, Any],
    *,
    mode: str,
    report_text: str = "",
    extra_rows: list[ThresholdRow] | None = None,
    extra_failures: list[str] | None = None,
) -> ThresholdResult:
    section = _threshold_section(thresholds, mode)
    result = ThresholdResult(ok=True)
    agg = _aggregate_report(report)
    expected = section.get("expected_case_count")
    if expected is not None:
        actual = agg.get("case_count", 0)
        _add_row(result, "case_count", expected, actual, int(actual) == int(expected))
    for name, target in (section.get("hard_gates") or {}).get("assertion_pass_rate_min", {}).items():
        actual = (agg.get("assertion_pass_rate") or {}).get(name, 0.0)
        _add_row(result, f"hard:{name}", f">= {target}", f"{actual:.3f}", float(actual) + 1e-9 >= float(target))
    if (section.get("hard_gates") or {}).get("key_leak_scan") and report_text:
        leaks = find_key_leaks(report_text)
        _add_row(result, "key_leak_scan", "0 leaks", len(leaks), len(leaks) == 0)
    qual = (section.get("qualitative") or {}).get("rubric_mean_min") or {}
    if qual and mode != "live_on":
        for name, target in qual.items():
            actual = (agg.get("score_means") or {}).get(name)
            if actual is None:
                _add_row(result, f"rubric_mean:{name}", f">= {target}", "n/a", False)
            else:
                _add_row(
                    result,
                    f"rubric_mean:{name}",
                    f">= {target}",
                    f"{actual:.3f}",
                    float(actual) + 1e-9 >= float(target),
                )
    reg = section.get("regression_vs_live_baseline") or {}
    if reg.get("enabled") and BASELINE_LIVE_PATH.is_file():
        diff = regression_diff(report, BASELINE_LIVE_PATH, score_tolerance=float(reg.get("score_tolerance", 0.08)))
        regressions = [r for r in (diff.get("regressions") or []) if not str(r).startswith("case_count mismatch")]
        _add_row(result, "live_baseline_regression", f"tol={reg.get('score_tolerance')}", regressions or "(none)", not regressions)
    if extra_rows:
        result.rows.extend(extra_rows)
    if extra_failures:
        for msg in extra_failures:
            _add_row(result, "on_arm_regression", "none", msg, False)
    result.ok = all(r.ok for r in result.rows)
    return result


def format_threshold_table(result: ThresholdResult) -> str:
    lines = ["# Eval thresholds", "", f"**Overall:** {'PASS' if result.ok else 'FAIL'}", "", "| Check | Target | Actual | Status |", "|-------|--------|--------|--------|"]
    for row in result.rows:
        lines.append(f"| {row.check} | {row.target} | {row.actual} | {'pass' if row.ok else '**FAIL**'} |")
    return "\n".join(lines) + "\n"


def check_thresholds_file(
    report: EvaluationReport,
    *,
    mode: str,
    thresholds_path: Path | None = None,
    report_json_path: Path | None = None,
    report_md_path: Path | None = None,
) -> ThresholdResult:
    thresholds = load_thresholds(thresholds_path)
    chunks: list[str] = []
    if report_json_path and report_json_path.is_file():
        chunks.append(report_json_path.read_text(encoding="utf-8", errors="replace"))
    if report_md_path and report_md_path.is_file():
        chunks.append(report_md_path.read_text(encoding="utf-8", errors="replace"))
    return check_thresholds(report, thresholds, mode=mode, report_text="\n".join(chunks))
