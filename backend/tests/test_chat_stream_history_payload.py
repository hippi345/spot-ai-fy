"""Regression: lenient /api/chat/stream history parsing (no pre-stream hard reject)."""

# pylint: disable=unused-argument

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from spot_backend.app import app
from spot_backend.chat_messages import FRIENDLY_SPOTIFY_GUIDANCE


def _collect_sse_events(stream_resp) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in stream_resp.text.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    events.append(json.loads(payload))
    return events


def _laptop_history_prefix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    script = [
        ("user", "Play something by The Weeknd"),
        ("assistant", "Playing The Weeknd on Spotify."),
        ("user", "play his latest single"),
        ("assistant", "Playing the newest single."),
        ("user", "Can you play podcasts via this interface?"),
        ("assistant", "No — podcasts are not supported here."),
        ("user", "Play one of my playlists"),
        ("assistant", "Playing Jamz · My Artists · 2026-09-25."),
    ]
    for role, content in script:
        rows.append({"role": role, "content": content})
    return rows


@respx.mock
def test_chat_stream_accepts_fourteen_turn_history_with_messy_rows(
    data_dir, signed_in_tokens, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key-history-payload")
    respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": "Five studio albums."}]},
                    }
                ]
            },
        )
    )
    history = _laptop_history_prefix()
    for i in range(6):
        history.append({"role": "user", "content": f"extra user turn {i}"})
        history.append({"role": "assistant", "content": f"extra assistant {i}"})
    history.append({"role": "assistant", "content": ""})
    history.append({"role": "assistant", "content": FRIENDLY_SPOTIFY_GUIDANCE})
    history.append({"role": "user", "text": "legacy text field turn"})
    history.append({"role": "assistant", "text": "legacy assistant · middle dot · ok"})

    client = TestClient(app)
    with patch(
        "spot_backend.gemini_llm.gemini_deterministic_shortcut_reply",
        return_value=None,
    ):
        resp = client.post(
            "/api/chat/stream",
            json={
                "message": "How many albums does Drake have?",
                "history": history,
                "conversation_id": "hist-payload-14",
            },
        )
    assert resp.status_code == 200
    events = _collect_sse_events(resp)
    err = next((e for e in events if e.get("type") == "error"), None)
    assert err is None, events
    final = next((e for e in events if e.get("type") == "final"), None)
    assert final and str(final.get("text") or "").strip()


def test_chat_stream_trims_oversized_history_instead_of_rejecting(
    data_dir, signed_in_tokens, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key-history-trim")
    respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": "OK"}]},
                    }
                ]
            },
        )
    )
    history = []
    for i in range(55):
        history.append({"role": "user", "content": f"u{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})
    client = TestClient(app)
    with patch(
        "spot_backend.gemini_llm.gemini_deterministic_shortcut_reply",
        return_value=None,
    ):
        resp = client.post(
            "/api/chat/stream",
            json={"message": "pause", "history": history, "conversation_id": "trim"},
        )
    assert resp.status_code == 200
    assert _collect_sse_events(resp)


def test_chat_stream_validation_error_returns_sse_error_not_bare_400(
    data_dir, signed_in_tokens,
) -> None:
    client = TestClient(app)
    resp = client.post("/api/chat/stream", json={"message": "", "history": []})
    assert resp.status_code == 200
    events = _collect_sse_events(resp)
    assert any(e.get("type") == "error" for e in events)
    assert any(e.get("type") == "done" for e in events)


def test_chat_stream_truncates_huge_assistant_content(
    data_dir, signed_in_tokens, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turn 5+ failed when a prior assistant reply exceeded strict 48k pydantic max."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key-huge-hist")
    respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": "Paused."}]},
                    }
                ]
            },
        )
    )
    huge = "x" * 50_000
    history = _laptop_history_prefix()
    history[-1] = {"role": "assistant", "content": huge}
    client = TestClient(app)
    with patch(
        "spot_backend.gemini_llm.gemini_deterministic_shortcut_reply",
        return_value=None,
    ):
        resp = client.post(
            "/api/chat/stream",
            json={"message": "pause", "history": history, "conversation_id": "huge"},
        )
    assert resp.status_code == 200
    events = _collect_sse_events(resp)
    assert not any(e.get("type") == "error" for e in events)


def test_chat_history_max_turns_constant_matches_frontend() -> None:
    from spot_backend.chat_request import CHAT_HISTORY_MAX_TURNS as backend_max

    assert backend_max == 40
