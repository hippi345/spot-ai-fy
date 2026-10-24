"""Ensure eval outputs never contain API key material."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.case_loader import load_dataset
from evals.key_leak_scan import assert_no_key_leaks
from evals.report_json import evaluation_report_to_json
from evals.task import eval_task


def test_mock_ci_report_has_no_key_patterns() -> None:
    ds = load_dataset(include_judges=False)
    ci = [c for c in ds.cases if isinstance(c.metadata, dict) and c.metadata.get("ci")]
    ds = type(ds)(name=f"{ds.name}_ci", cases=ci, evaluators=ds.evaluators)
    import respx

    fake_key = "AIzaSyMustNotAppearInEvalReport123456789"
    with respx.mock:
        report = ds.evaluate_sync(
            lambda inputs: eval_task(inputs, force_mock=True),
            name="key_leak_scan",
            progress=False,
            max_concurrency=1,
        )
    blob = evaluation_report_to_json(report)
    assert_no_key_leaks(blob, extra_secrets=(fake_key,))


def test_gemini_eval_artifact_guard_unchanged(tmp_path) -> None:
    from tests.gemini_eval_harness import EvalReport
    from tests.gemini_eval_sanitize import write_sanitized_eval_artifact

    secret = "AIzaSySecretKeyThatMustNotAppearInJsonFile"
    report = EvalReport(crash_detail=secret)
    out = tmp_path / "out.json"
    write_sanitized_eval_artifact(report, out, secrets=(secret,))
    assert_no_key_leaks(out.read_text(encoding="utf-8"), (secret,))
