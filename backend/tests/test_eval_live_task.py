"""Live eval task plumbing (isolated DATA_DIR, llm_invoked)."""

from __future__ import annotations

from unittest.mock import patch

from evals.task import eval_task, run_live_case
from evals.types import EvalCaseInput


def test_eval_task_uses_live_path_when_not_force_mock() -> None:
    from evals.types import AgentTurnResult

    with patch("evals.task.run_live_case") as live:
        live.return_value = AgentTurnResult(reply="ok", llm_invoked=True)
        out = eval_task({"prompt": "play x", "mode": "mock"}, force_mock=False)
        live.assert_called_once()
        assert out.llm_invoked


def test_run_live_case_llm_invoked_only_on_llm_delta() -> None:
    events = [
        {"type": "tool_start", "name": "spotify_me"},
        {"type": "tool_done", "name": "spotify_me", "preview": "{}"},
        {"type": "final", "text": "You are logged in."},
    ]

    def fake_iter(*_a, **_k):
        yield from events

    with patch("evals.task.iter_chat_events", fake_iter):
        result = run_live_case(EvalCaseInput(prompt="who am i"), provider="gemini")
    assert result.llm_invoked is False
    assert result.provider == "gemini"


def test_run_live_case_llm_invoked_true_when_model_streams() -> None:
    events = [
        {"type": "llm_delta", "text": "Hi"},
        {"type": "final", "text": "Hi there"},
    ]

    def fake_iter(*_a, **_k):
        yield from events

    with patch("evals.task.iter_chat_events", fake_iter):
        result = run_live_case(EvalCaseInput(prompt="hello"), provider="gemini")
    assert result.llm_invoked is True


def test_run_live_case_provider_before_settings() -> None:
    captured: list[str] = []

    def fake_iter(_prompt, settings, **_k):
        captured.append(settings.llm_provider)
        yield {"type": "final", "text": "ok"}

    with patch("evals.task.iter_chat_events", fake_iter):
        run_live_case(EvalCaseInput(prompt="hi"), provider="gemini")
    assert captured == ["gemini"]


def test_item09_gemini_iter_marks_llm_invoked(data_dir, signed_in_tokens) -> None:
    from spot_backend.config import Settings
    from spot_backend.gemini_llm import iter_gemini_chat_events

    settings = Settings(gemini_api_key="gemini-test-key-1234567890")

    def fake_gemini(user_text, settings, history=None, *, emit=None, conversation_id=None):
        if emit:
            emit({"type": "llm_delta", "text": "ok"})
        return "ok"

    with patch("spot_backend.gemini_llm.run_chat_turn_gemini", fake_gemini):
        types = [e["type"] for e in iter_gemini_chat_events("who am i", settings)]
    assert "llm_delta" in types
