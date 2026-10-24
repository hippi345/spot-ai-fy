"""Optional real-Gemini multi-turn laptop replay (mocked Spotify HTTP)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from spot_backend.config import Settings
from tests.gemini_eval_harness import EvalReport, format_eval_table, run_multiturn_gemini_eval
from tests.gemini_eval_sanitize import assert_no_secrets_in_text, write_sanitized_eval_artifact

_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "gemini_eval_latest.json"

pytestmark = pytest.mark.gemini_eval


def _fake_gemini_key() -> str:
    return "AIzaSyMockGeminiEvalTestKey1234567890abcd"


@pytest.fixture
def gemini_eval_settings(data_dir, signed_in_tokens, monkeypatch):
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if os.environ.get("RUN_GEMINI_EVAL_LIVE") == "1":
        if not key:
            pytest.skip("RUN_GEMINI_EVAL_LIVE=1 but GEMINI_API_KEY not set")
    else:
        key = _fake_gemini_key()
        monkeypatch.setenv("GEMINI_API_KEY", key)
    s = Settings()
    s.gemini_api_key = key
    return s


def test_gemini_eval_redacts_respx_not_mocked_query_key(tmp_path) -> None:
    """respx errors embed the full request URL including ?key= when auth was query-based."""
    secret = "AIzaSyRespxNotMockedLeakKeyForUnitTest12"
    crash = (
        "respx.models.RouteAssertionError: GET request not mocked for "
        f"https://generativelanguage.googleapis.com/v1beta/models?key={secret}"
    )
    report = EvalReport(crashed=True, crash_detail=crash)
    out = tmp_path / "gemini_eval_latest.json"
    write_sanitized_eval_artifact(report, out, secrets=(secret,))
    text = out.read_text(encoding="utf-8")
    assert secret not in text
    assert_no_secrets_in_text(text, (secret,))


def test_gemini_eval_artifact_redacts_api_key(tmp_path) -> None:
    secret = "AIzaSySecretKeyThatMustNotAppearInJsonFile"
    report = EvalReport(crashed=True, crash_detail=f"error body mentioned {secret}")
    report.turns.append(
        {
            "prompt": f"prompt with {secret}",
            "pass": False,
            "note": secret,
            "tools": [{"tool": "spotify_me", "args": {"key": secret}}],
            "reply": secret,
        }
    )
    out = tmp_path / "gemini_eval_latest.json"
    write_sanitized_eval_artifact(report, out, secrets=(secret,))
    text = out.read_text(encoding="utf-8")
    assert_no_secrets_in_text(text, (secret,))
    table = format_eval_table(report, secrets=(secret,))
    assert_no_secrets_in_text(table, (secret,))


@pytest.mark.skipif(
    os.environ.get("RUN_GEMINI_EVAL_LIVE") != "1",
    reason="Real Gemini laptop script: set RUN_GEMINI_EVAL_LIVE=1 and GEMINI_API_KEY",
)
def test_gemini_multiturn_laptop_script(gemini_eval_settings) -> None:
    secrets = (gemini_eval_settings.gemini_api_key,)
    report = run_multiturn_gemini_eval(gemini_eval_settings)
    table = format_eval_table(report, secrets=secrets)
    print(table)
    write_sanitized_eval_artifact(report, _ARTIFACT, secrets=secrets)
    print(f"Wrote {_ARTIFACT}")
    artifact_text = _ARTIFACT.read_text(encoding="utf-8")
    assert_no_secrets_in_text(artifact_text, secrets)
    if report.crashed:
        from tests.gemini_eval_sanitize import redact_text

        detail = redact_text(report.crash_detail or "gemini eval crashed", secrets)
        raise AssertionError(format_eval_table(report, secrets=secrets) + "\n" + detail[:500])
