from __future__ import annotations

import json
from unittest.mock import patch

import httpx

from spot_backend.agent import run_chat_turn_ollama
from spot_backend.config import Settings
from spot_backend.spotify_tools import SpotifyToolRunner


class _FakeOllamaStream:
    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code
        self._text = ""

    def __enter__(self) -> _FakeOllamaStream:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=httpx.Request("POST", "http://x"), response=httpx.Response(self.status_code))

    def read(self) -> bytes:
        return b""

    def iter_lines(self):
        for line in self._lines:
            yield line


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


def test_agent_tool_call_then_final_answer(data_dir, signed_in_tokens, monkeypatch) -> None:
    settings = Settings()
    responses = [
        _FakeOllamaStream([_ollama_tool_call_line("spotify_me")]),
        _FakeOllamaStream([_ollama_final_line("You are logged in as testuser.")]),
    ]
    idx = {"i": 0}

    def fake_stream(self, method, url, **kwargs):
        i = idx["i"]
        idx["i"] += 1
        return responses[i]

    with patch.object(SpotifyToolRunner, "_me", return_value='{"id":"testuser"}'):
        with patch("httpx.Client.stream", fake_stream):
            out = run_chat_turn_ollama("who am I?", settings)
    assert "testuser" in out.lower() or "logged in" in out.lower()


def test_unknown_tool_returns_error_json(data_dir, signed_in_tokens) -> None:
    runner = SpotifyToolRunner(settings=Settings())
    try:
        raw = runner.run("spotify_not_a_real_tool", {})
        data = json.loads(raw)
        assert "Unknown tool" in data["error"]
    finally:
        runner.close()


def test_tool_dispatch_exception_becomes_json(data_dir, signed_in_tokens) -> None:
    runner = SpotifyToolRunner(settings=Settings())
    try:
        with patch.object(runner, "_dispatch", side_effect=RuntimeError("boom")):
            raw = runner.run("spotify_search", {"query": "x"})
        data = json.loads(raw)
        assert data["error"] == "RuntimeError: boom"
    finally:
        runner.close()
