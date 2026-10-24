from __future__ import annotations

import json
from pathlib import Path

import respx
import yaml

from evals.report_json import evaluation_report_to_json
from evals.run import main as eval_main
from pydantic_evals import Dataset
from evals.evaluators import CUSTOM_EVALUATOR_TYPES, dataset_evaluators
from evals.task import eval_task


def test_evaluation_report_to_json_roundtrip() -> None:
    from evals.case_loader import load_dataset

    ds = load_dataset(include_judges=False)
    ds.evaluators = [ev for ev in dataset_evaluators(include_judges=False) if ev]
    ci = [c for c in ds.cases if isinstance(c.metadata, dict) and c.metadata.get("ci")][:1]
    ds = Dataset(name="one_case", cases=ci, evaluators=ds.evaluators)

    with respx.mock:
        report = ds.evaluate_sync(lambda inputs: eval_task(inputs, force_mock=True), progress=False)
        raw = evaluation_report_to_json(report)
        data = json.loads(raw)
        assert data["name"]
        assert data["cases"]


def test_ci_cli_writes_report_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "a" * 32)
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"
    code = eval_main(
        [
            "--mode",
            "mock",
            "--ci",
            "--regression-report",
            "--report-json",
            str(out_json),
            "--report-md",
            str(out_md),
        ]
    )
    assert code == 0
    assert out_json.is_file()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "cases" in payload
    assert out_md.is_file()
