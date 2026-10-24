"""Load eval cases from backend/evals/cases/*.yaml (merged dataset)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic_evals import Case, Dataset

from evals.evaluators import CUSTOM_EVALUATOR_TYPES, dataset_evaluators

CASES_DIR = Path(__file__).with_name("cases")
LEGACY_CASES = Path(__file__).with_name("cases.yaml")


def _read_case_file(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return []
    cases = raw.get("cases")
    if not isinstance(cases, list):
        return []
    return [c for c in cases if isinstance(c, dict)]


def case_files() -> list[Path]:
    if CASES_DIR.is_dir():
        files = sorted(CASES_DIR.glob("*.yaml"))
        if files:
            return files
    if LEGACY_CASES.is_file():
        return [LEGACY_CASES]
    return []


def load_dataset(*, include_judges: bool = False) -> Dataset:
    all_cases: list[Case] = []
    suite_name = "spot_agent_evals"
    for path in case_files():
        blob = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(blob, dict) and blob.get("name"):
            suite_name = str(blob["name"])
        for row in _read_case_file(path):
            name = str(row.get("name") or "")
            inputs = row.get("inputs")
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            if not name or inputs is None:
                continue
            all_cases.append(Case(name=name, inputs=inputs, metadata=metadata))
    ds = Dataset(
        name=suite_name,
        cases=all_cases,
        evaluators=[],
    )
    for ev in dataset_evaluators(include_judges=include_judges):
        if ev not in ds.evaluators:
            ds.evaluators.append(ev)
    return ds


def feature_summary() -> list[dict[str, Any]]:
    """Aggregate case counts by metadata.feature (or group) for README tables."""
    counts: dict[str, dict[str, int]] = {}
    modes: dict[str, set[str]] = {}
    checks: dict[str, set[str]] = {}
    for path in case_files():
        for row in _read_case_file(path):
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            feature = str(meta.get("feature") or meta.get("group") or "other")
            bucket = counts.setdefault(feature, {"total": 0, "ci": 0})
            bucket["total"] += 1
            if meta.get("ci"):
                bucket["ci"] += 1
            inp = row.get("inputs") if isinstance(row.get("inputs"), dict) else {}
            mode = str(inp.get("mode") or "mock")
            modes.setdefault(feature, set()).add(mode)
            ct = str(meta.get("check_type") or ("deterministic" if meta.get("ci") else "judge"))
            checks.setdefault(feature, set()).add(ct)
    rows = []
    for feature in sorted(counts):
        rows.append(
            {
                "feature": feature,
                "case_count": counts[feature]["total"],
                "ci_cases": counts[feature]["ci"],
                "check_type": ", ".join(sorted(checks.get(feature, {"deterministic"}))),
                "mode": ", ".join(sorted(modes.get(feature, {"mock"}))),
            }
        )
    return rows


def feature_summary_json() -> str:
    return json.dumps(feature_summary(), indent=2)
