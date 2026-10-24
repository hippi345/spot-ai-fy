"""Baseline capture and regression comparison for eval reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic_evals.reporting import EvaluationReport

BASELINE_PATH = Path(__file__).with_name("baseline.json")
BASELINE_CI_PATH = Path(__file__).with_name("baseline-ci.json")
BASELINE_LIVE_PATH = Path(__file__).with_name("baseline-live.json")


def _aggregate_report(report: EvaluationReport) -> dict[str, Any]:
    assertion_rates: dict[str, list[bool]] = {}
    score_vals: dict[str, list[float]] = {}
    for case in report.cases:
        for name, res in (case.assertions or {}).items():
            if isinstance(res.value, str) and res.value == "skipped":
                continue
            if isinstance(res.value, bool):
                assertion_rates.setdefault(name, []).append(res.value)
            else:
                assertion_rates.setdefault(name, []).append(bool(res.value))
        for name, res in (case.scores or {}).items():
            try:
                score_vals.setdefault(name, []).append(float(res.value))
            except (TypeError, ValueError):
                continue
    assertions_mean = {
        k: (sum(1 for v in vs if v) / len(vs) if vs else 0.0) for k, vs in assertion_rates.items()
    }
    scores_mean = {k: (sum(vs) / len(vs) if vs else 0.0) for k, vs in score_vals.items()}
    return {
        "case_count": len(report.cases),
        "assertion_pass_rate": assertions_mean,
        "score_means": scores_mean,
    }


def save_baseline(
    report: EvaluationReport,
    path: Path | None = None,
    *,
    scope: str = "full",
) -> Path:
    path = path or (BASELINE_CI_PATH if scope == "ci" else BASELINE_PATH)
    payload = {
        "version": 1,
        "suite": "spot_agent_evals",
        "scope": scope,
        "metrics": _aggregate_report(report),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def save_live_baseline(report: EvaluationReport, path: Path | None = None) -> Path:
    """Append/update qualitative score means from a live eval run (LLMJudge enabled)."""
    path = path or BASELINE_LIVE_PATH
    metrics = _aggregate_report(report)
    payload: dict[str, Any] = {
        "version": 1,
        "suite": "spot_agent_evals_live",
        "metrics": metrics,
    }
    if path.is_file():
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(prior, dict) and prior.get("metrics"):
                payload["prior_runs"] = prior.get("prior_runs", []) + [prior.get("metrics")]
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_baseline(path: Path | None = None) -> dict[str, Any]:
    path = path or BASELINE_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def format_regression_markdown(diff: dict[str, Any]) -> str:
    lines = ["# Agent eval regression report", ""]
    if diff.get("ok"):
        lines.append("**Status:** pass (no regressions vs baseline.json)")
    else:
        lines.append("**Status:** fail")
    lines.extend(["", "## Regressions", ""])
    for item in diff.get("regressions") or []:
        lines.append(f"- {item}")
    if not diff.get("regressions"):
        lines.append("- (none)")
    lines.extend(["", "## Candidate metrics", "", "```json", json.dumps(diff.get("candidate"), indent=2), "```"])
    return "\n".join(lines) + "\n"


def regression_diff(
    candidate: EvaluationReport,
    baseline_path: Path | None = None,
    *,
    score_tolerance: float = 0.05,
    expect_scope: str | None = None,
) -> dict[str, Any]:
    base = load_baseline(baseline_path)
    cand = _aggregate_report(candidate)
    regressions: list[str] = []
    base_scope = base.get("scope")
    if expect_scope and base_scope and base_scope != expect_scope:
        regressions.append(f"baseline scope {base_scope!r} != expected {expect_scope!r}")
    base_count = base.get("metrics", {}).get("case_count")
    if base_count is not None and int(base_count) != int(cand.get("case_count", 0)):
        regressions.append(
            f"case_count mismatch: ran {cand.get('case_count')} cases, baseline expects {base_count}"
        )
    base_assert = base.get("metrics", {}).get("assertion_pass_rate", {})
    cand_assert = cand.get("assertion_pass_rate", {})
    for name, rate in base_assert.items():
        cur = cand_assert.get(name, 0.0)
        if cur + 1e-9 < float(rate):
            regressions.append(f"assertion {name}: {cur:.3f} < baseline {float(rate):.3f}")
    base_scores = base.get("metrics", {}).get("score_means", {})
    cand_scores = cand.get("score_means", {})
    for name, mean in base_scores.items():
        cur = cand_scores.get(name, 0.0)
        if cur < float(mean) - score_tolerance:
            regressions.append(
                f"score {name}: {cur:.3f} < baseline {float(mean):.3f} - tol {score_tolerance}"
            )
    return {
        "ok": len(regressions) == 0,
        "regressions": regressions,
        "baseline": base,
        "candidate": cand,
    }
