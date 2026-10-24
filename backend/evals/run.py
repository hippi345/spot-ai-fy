"""CLI entrypoint for the agent eval suite.

Verified against pydantic-evals 2.61.0 (pydantic-ai-slim 2.61.0):
- Dataset.evaluate / evaluate_sync accept ``repeat`` (default 1); see
  https://pydantic.dev/docs/ai/evals/how-to/multi-run/
- MaxDuration(seconds=...) built-in evaluator
- EvaluationReport.print(baseline=...) for A/B diff tables
- LLMJudge requires pydantic-ai-slim (ModelSettings, model providers); see
  https://pydantic.dev/docs/ai/evals/evaluators/llm-judge/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import respx
from pydantic_evals import Dataset

from evals.baseline import (
    BASELINE_PATH,
    format_regression_markdown,
    load_baseline,
    regression_diff,
    save_baseline,
)
from evals.evaluators import CUSTOM_EVALUATOR_TYPES, dataset_evaluators
from evals.task import eval_task

CASES_PATH = Path(__file__).with_name("cases.yaml")


def _load_dataset() -> Dataset:
    ds = Dataset.from_file(
        CASES_PATH,
        custom_evaluator_types=CUSTOM_EVALUATOR_TYPES,
    )
    for ev in dataset_evaluators(include_judges=False):
        if ev not in ds.evaluators:
            ds.evaluators.append(ev)
    return ds


def _providers_to_run(requested: str) -> list[str]:
    if requested == "all":
        out = ["ollama"]
        for name, env in (
            ("gemini", "GEMINI_API_KEY"),
            ("openai", "OPENAI_API_KEY"),
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("xai", "XAI_API_KEY"),
        ):
            if (os.environ.get(env) or "").strip():
                out.append(name)
        return out
    return [requested]


def _make_task(provider: str, *, force_mock: bool):
    def task(inputs: Any) -> Any:
        return eval_task(inputs, provider=provider, force_mock=force_mock)

    task.__name__ = f"spot_agent_{provider}_{'mock' if force_mock else 'live'}"
    return task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Spot-AI-fy agent eval suite (pydantic-evals)")
    parser.add_argument("--mode", choices=("mock", "live"), default="mock")
    parser.add_argument("--provider", default="ollama", help="ollama|gemini|openai|anthropic|xai|all")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--ci", action="store_true", help="Only cases with metadata.ci=true")
    parser.add_argument("--compare", action="store_true", help="Diff against baseline.json")
    parser.add_argument("--save-baseline", action="store_true", help="Write baseline.json from this run")
    parser.add_argument("--regression-report", action="store_true", help="Exit 1 if metrics regress vs baseline")
    parser.add_argument("--judge", default="", help="Enable LLM judges (e.g. ollama:qwen2.5:7b)")
    parser.add_argument("--output", type=Path, default=None, help="JSON report path")
    parser.add_argument("--report-json", type=Path, default=None, help="Write pydantic-evals report JSON")
    parser.add_argument("--report-md", type=Path, default=None, help="Write regression markdown summary")
    args = parser.parse_args(argv)

    if args.judge:
        os.environ["EVAL_JUDGE_MODEL"] = args.judge

    dataset = _load_dataset()
    if args.judge:
        for ev in dataset_evaluators(include_judges=True):
            if ev not in dataset.evaluators:
                dataset.evaluators.append(ev)

    if args.ci:
        dataset = Dataset(
            name=dataset.name,
            cases=[c for c in dataset.cases if isinstance(c.metadata, dict) and c.metadata.get("ci")],
            evaluators=dataset.evaluators,
            report_evaluators=dataset.report_evaluators,
        )

    force_mock = args.mode == "mock"
    providers = _providers_to_run(args.provider)
    last_report = None
    baseline_report = None
    if args.compare or args.regression_report:
        try:
            baseline_report = load_baseline()
        except (OSError, json.JSONDecodeError):
            baseline_report = None

    mock_ctx = respx.mock if force_mock else nullcontext
    for provider in providers:
        task = _make_task(provider, force_mock=force_mock)
        with mock_ctx():
            report = dataset.evaluate_sync(
                task,
                name=f"{dataset.name}_{provider}_{args.mode}",
                repeat=max(1, int(args.repeat)),
                progress=True,
            )
        last_report = report
        print(f"\n=== {provider} ({args.mode}) ===")
        if baseline_report and args.compare:
            # Re-run baseline metrics are stored in baseline.json; print diff vs stored snapshot
            from pydantic_evals.reporting import EvaluationReport

            # print() accepts a baseline EvaluationReport; load stored run is summary-only, so show table without diff
            report.print(include_input=True, include_output=True)
        else:
            report.print(include_input=True, include_output=True)

        if args.output:
            out_path = args.output if len(providers) == 1 else args.output.with_stem(
                f"{args.output.stem}_{provider}"
            )
            out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        if args.report_json:
            rj = args.report_json if len(providers) == 1 else args.report_json.with_stem(
                f"{args.report_json.stem}_{provider}"
            )
            rj.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    if last_report is None:
        print("No cases to run.", file=sys.stderr)
        return 2

    if args.save_baseline:
        path = save_baseline(last_report)
        print(f"Wrote baseline to {path}")

    if args.regression_report:
        diff = regression_diff(last_report)
        print(json.dumps(diff, indent=2))
        if args.report_md:
            args.report_md.write_text(format_regression_markdown(diff), encoding="utf-8")
        return 0 if diff["ok"] else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
