"""CI-fast mock eval plumbing (no Ollama / no cloud keys)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_evals import Dataset

from evals.case_loader import load_dataset
from evals.task import eval_task


@pytest.fixture
def ci_dataset(data_dir) -> Dataset:
    ds = load_dataset(include_judges=False)
    ci_cases = [c for c in ds.cases if isinstance(c.metadata, dict) and c.metadata.get("ci")]
    return Dataset(name=f"{ds.name}_ci", cases=ci_cases, evaluators=ds.evaluators)


def test_ci_mock_eval_suite_passes(ci_dataset: Dataset) -> None:
    report = ci_dataset.evaluate_sync(
        lambda inputs: eval_task(inputs, force_mock=True),
        name="ci_mock_smoke",
        progress=False,
        max_concurrency=1,
    )
    failures = []
    for case in report.cases:
        for name, res in (case.assertions or {}).items():
            if isinstance(res.value, str) and res.value == "skipped":
                continue
            if isinstance(res.value, bool) and not res.value:
                failures.append(f"{case.name}:{name}={res.value} {getattr(res, 'reason', '')}")
    assert not failures, "CI eval failures:\n" + "\n".join(failures)
