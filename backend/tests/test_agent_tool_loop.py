from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx

from spot_backend.agent import run_chat_turn_ollama
from spot_backend.config import Settings
from spot_backend.spotify_tools import SpotifyToolRunner
from tests.recheck_helpers import FakeOllamaStream


def _ollama_tool_call_line(name: str, arguments: dict | None = None) -> str:
    args = arguments or {}
    payload = {
        "message": {
            "role": "assistant",
            "tool_calls": [{"function": {"name": name, "arguments": args}}],
        },
        "done": True,
    }
    return json.dumps(payload)


def _ollama_final_line(text: str) -> str:
    return json.dumps({"message": {"role": "assistant", "content": text}, "done": True})


def _run_with_fake_ollama(
    settings: Settings,
    user_text: str,
    streams: list[FakeOllamaStream],
) -> tuple[str, list[dict[str, Any]]]:
    bodies: list[dict[str, Any]] = []
    idx = {"i": 0}

    def fake_stream(_client_self, _method, _url, **kwargs):
        bodies.append(kwargs["json"])
        i = idx["i"]
        idx["i"] += 1
        return streams[i]

    with patch("httpx.Client.stream", fake_stream):
        out = run_chat_turn_ollama(user_text, settings)
    return out, bodies


def test_agent_tool_call_then_final_answer(data_dir, signed_in_tokens) -> None:
    settings = Settings()
    me_tool_result = '{"id":"testuser"}'
    final_text = "You are logged in as testuser."
    streams = [
        FakeOllamaStream([_ollama_tool_call_line("spotify_me")]),
        FakeOllamaStream([_ollama_final_line(final_text)]),
    ]
    with patch.object(SpotifyToolRunner, "_me", return_value=me_tool_result):
        out, bodies = _run_with_fake_ollama(settings, "who am I?", streams)

    assert out == final_text
    assert len(bodies) == 2
    second_messages = bodies[1]["messages"]
    assert second_messages[1] == {"role": "user", "content": "who am I?"}
    assert second_messages[2]["role"] == "assistant"
    assert second_messages[2]["tool_calls"] == [
        {"function": {"name": "spotify_me", "arguments": {}}}
    ]
    assert second_messages[2]["content"] == ""
    assert second_messages[3] == {
        "role": "tool",
        "name": "spotify_me",
        "content": me_tool_result,
    }


def test_agent_unknown_tool_error_fed_back_to_model(data_dir, signed_in_tokens) -> None:
    settings = Settings()
    tool_name = "spotify_not_a_real_tool"
    tool_result = json.dumps({"error": f"Unknown tool: {tool_name}"})
    final_text = "That tool does not exist."
    streams = [
        FakeOllamaStream([_ollama_tool_call_line(tool_name)]),
        FakeOllamaStream([_ollama_final_line(final_text)]),
    ]
    out, bodies = _run_with_fake_ollama(settings, "do something weird", streams)

    assert out == final_text
    second_messages = bodies[1]["messages"]
    assert second_messages[3] == {
        "role": "tool",
        "name": tool_name,
        "content": tool_result,
    }


def test_agent_dispatch_exception_fed_back_to_model(data_dir, signed_in_tokens) -> None:
    settings = Settings()
    tool_result = json.dumps({"error": "RuntimeError: boom"})
    final_text = "The search tool crashed."
    streams = [
        FakeOllamaStream([_ollama_tool_call_line("spotify_search", {"query": "x", "types": "track"})]),
        FakeOllamaStream([_ollama_final_line(final_text)]),
    ]
    real_dispatch = SpotifyToolRunner._dispatch

    def boom_on_search(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "spotify_search":
            raise RuntimeError("boom")
        return real_dispatch(self, name, arguments)

    with patch.object(SpotifyToolRunner, "_dispatch", boom_on_search):
        out, bodies = _run_with_fake_ollama(settings, "search x", streams)

    assert out == final_text
    second_messages = bodies[1]["messages"]
    assert second_messages[3] == {
        "role": "tool",
        "name": "spotify_search",
        "content": tool_result,
    }
