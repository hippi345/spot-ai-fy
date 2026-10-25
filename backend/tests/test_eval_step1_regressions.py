from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import respx

from evals.evaluators import BlankTrackTitleRejected, MoodIntentJudge, NoMockHttpLeakInReply
from evals.types import AgentTurnResult, ToolCallRecord
from pydantic_evals.evaluators import EvaluatorContext
from spot_backend.config import Settings
from spot_backend.spotify_tools import SpotifyToolRunner


def _ctx(output: AgentTurnResult, metadata: dict) -> EvaluatorContext:
    return EvaluatorContext(
        name="t",
        inputs={"prompt": "x"},
        metadata=metadata,
        expected_output=None,
        output=output,
        duration=output.duration_s,
        _span_tree=None,
        attributes={},
        metrics={},
    )


def test_play_track_rejects_blank_artist(data_dir, signed_in_tokens) -> None:
    runner = SpotifyToolRunner(settings=Settings(data_dir=data_dir))
    raw = runner.run("spotify_play_track", {"track_name": "Down", "artist_name": "  "})
    runner.close()
    assert "artist_name is required" in raw


def test_no_mock_http_leak_in_reply_evaluator() -> None:
    ev = NoMockHttpLeakInReply()
    ctx = _ctx(
        AgentTurnResult(reply="AllMockedAssertionError: RESPX: not mocked", tool_calls=[]),
        {},
    )
    assert ev.evaluate(ctx) is False


def test_mood_intent_judge_can_fail() -> None:
    ev = MoodIntentJudge()
    ctx = _ctx(AgentTurnResult(reply="nope", tool_calls=[]), {"mood_request": True})
    with patch.dict("os.environ", {"EVAL_JUDGE_MODEL": "openai:test"}):
        with patch("evals.judges._run_judge_eval_sync", return_value=False):
            assert ev.evaluate(ctx) is False


def test_blank_track_title_evaluator_fails_on_catalog_guess() -> None:
    ev = BlankTrackTitleRejected()
    ctx = _ctx(
        AgentTurnResult(
            reply="Playing",
            tool_calls=[
                ToolCallRecord(
                    name="spotify_play_track",
                    arguments={"track_name": "   ", "artist_name": "Jay Sean"},
                    result='{"ok": true, "playback_verified": true}',
                )
            ],
        ),
        {},
    )
    assert ev.evaluate(ctx) is False


def test_blank_track_title_evaluator_fails_both_blank_success() -> None:
    ev = BlankTrackTitleRejected()
    ctx = _ctx(
        AgentTurnResult(
            reply="ok",
            tool_calls=[
                ToolCallRecord(
                    name="spotify_play_track",
                    arguments={"track_name": "", "artist_name": ""},
                    result='{"ok": true}',
                )
            ],
        ),
        {},
    )
    assert ev.evaluate(ctx) is False


def test_blank_track_title_rejected_evaluator() -> None:
    ev = BlankTrackTitleRejected()
    ctx = _ctx(
        AgentTurnResult(
            reply="",
            tool_calls=[
                ToolCallRecord(
                    name="spotify_play_track",
                    arguments={"track_name": "  ", "artist_name": "Jay Sean"},
                    result='{"error": "track_name is required"}',
                )
            ],
        ),
        {},
    )
    assert ev.evaluate(ctx) is True
    bad = _ctx(
        AgentTurnResult(
            reply="",
            tool_calls=[
                ToolCallRecord(
                    name="spotify_play_track",
                    arguments={"track_name": "  ", "artist_name": "Jay Sean"},
                    result='{"error": "ok"}',
                )
            ],
        ),
        {},
    )
    assert ev.evaluate(bad) is False


def test_mood_intent_judge_skipped_without_judge_model() -> None:
    from pydantic_evals.evaluators import EvaluationReason

    ev = MoodIntentJudge()
    ctx = _ctx(AgentTurnResult(reply="ok", tool_calls=[]), {"mood_request": True})
    with patch.dict("os.environ", {}, clear=True):
        out = ev.evaluate(ctx)
    assert isinstance(out, EvaluationReason)
    assert out.value == "skipped"
    assert "EVAL_JUDGE_MODEL" in (out.reason or "")


@respx.mock
def test_gemini_generate_content_uses_header_not_query_key() -> None:
    from spot_backend.gemini_llm import _gemini_post_with_retry

    seen_url = ""

    def capture(request: httpx.Request) -> httpx.Response:
        nonlocal seen_url
        seen_url = str(request.url)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/.*").mock(side_effect=capture)
    with httpx.Client() as client:
        _gemini_post_with_retry(
            client,
            "https://generativelanguage.googleapis.com/v1beta/models/m:generateContent",
            headers={"x-goog-api-key": "AIzaSyTestKey1234567890"},
            json_body={},
        )
    assert "key=" not in seen_url
    assert "AIzaSy" not in seen_url


@respx.mock
def test_gemini_list_models_uses_header_not_query_key() -> None:
    from spot_backend.llm_provider_lists import fetch_gemini_chat_model_names

    seen_url = ""
    seen_headers: dict[str, str] = {}

    def capture(request: httpx.Request) -> httpx.Response:
        nonlocal seen_url
        seen_url = str(request.url)
        seen_headers.update(dict(request.headers))
        return httpx.Response(200, json={"models": []})

    respx.get(url__regex=r"https://generativelanguage\.googleapis\.com/v1beta/models.*").mock(
        side_effect=capture
    )
    fetch_gemini_chat_model_names("AIzaSyTestKey1234567890", page_size=10, timeout=5.0)
    assert "key=" not in seen_url
    assert "AIzaSy" not in seen_url
    assert "x-goog-api-key" in {k.lower() for k in seen_headers}
