"""CI-fast mock eval plumbing (no Ollama / no cloud keys)."""

from __future__ import annotations

from pathlib import Path

import pytest
import respx
from pydantic_evals import Dataset

from evals.evaluators import CUSTOM_EVALUATOR_TYPES, dataset_evaluators
from evals.task import eval_task

CASES_PATH = Path(__file__).resolve().parents[1] / "evals" / "cases.yaml"


@pytest.fixture
def ci_dataset(data_dir) -> Dataset:
    ds = Dataset.from_file(CASES_PATH, custom_evaluator_types=CUSTOM_EVALUATOR_TYPES)
    ds.evaluators = [ev for ev in dataset_evaluators(include_judges=False) if ev]
    ci_cases = [c for c in ds.cases if isinstance(c.metadata, dict) and c.metadata.get("ci")]
    return Dataset(name=f"{ds.name}_ci", cases=ci_cases, evaluators=ds.evaluators)


@respx.mock
def test_ci_mock_eval_suite_passes(ci_dataset: Dataset) -> None:
    report = ci_dataset.evaluate_sync(
        lambda inputs: eval_task(inputs, force_mock=True),
        name="ci_mock_smoke",
        progress=False,
    )
    failures = []
    for case in report.cases:
        for name, res in (case.assertions or {}).items():
            if not res.value:
                failures.append(f"{case.name}:{name}={res.value} {getattr(res, 'reason', '')}")
    assert not failures, "CI eval failures:\n" + "\n".join(failures)
