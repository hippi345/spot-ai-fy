"""Optional real-Gemini multi-turn laptop replay (mocked Spotify HTTP)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from spot_backend.config import Settings
from tests.gemini_eval_harness import format_eval_table, run_multiturn_gemini_eval

_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "gemini_eval_latest.json"

pytestmark = pytest.mark.gemini_eval


@pytest.fixture
def gemini_eval_settings(data_dir, signed_in_tokens):
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        pytest.skip("GEMINI_API_KEY not set")
    s = Settings()
    s.gemini_api_key = key
    return s


def test_gemini_multiturn_laptop_script(gemini_eval_settings) -> None:
    report = run_multiturn_gemini_eval(gemini_eval_settings)
    table = format_eval_table(report)
    print(table)
    _ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    _ARTIFACT.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
    print(f"Wrote {_ARTIFACT}")
    if report.crashed:
        raise AssertionError(report.crash_detail or "gemini eval crashed")
