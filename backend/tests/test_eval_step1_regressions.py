from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import respx

from evals.evaluators import MoodIntentJudge, NoMockHttpLeakInReply
from evals.types import AgentTurnResult
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
