"""Unit tests for multi-provider LLM adapters (mocked HTTP only)."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
import respx

from spot_backend.action_claim_guard import tool_summarize_reprompt
from spot_backend.anthropic_llm import run_chat_turn_anthropic
from spot_backend.config import Settings
from spot_backend.llm_secret_safety import redact_known_api_keys
from spot_backend.llm_tool_loop import ToolLoopState, finalize_assistant_text, sanitize_tool_arguments
from spot_backend.openai_compat_llm import run_chat_turn_openai_compat
from spot_backend.spotify_tools import SpotifyToolRunner, _sanitize_model_device_id


def test_sanitize_model_device_id_strips_placeholders() -> None:
    assert _sanitize_model_device_id("default") == ""
    assert _sanitize_model_device_id("dev-real-id") == "dev-real-id"


def test_tool_loop_sanitize_device_in_args() -> None:
    args = sanitize_tool_arguments("spotify_pause", {"device_id": "default"})
    assert args["device_id"] == ""


def test_tool_summarize_reprompt_non_empty() -> None:
    body = tool_summarize_reprompt(['{"ok": true}'])
    assert "tool" in body.lower() or "summary" in body.lower()


def test_finalize_assistant_text_summarize_reprompt_on_boilerplate() -> None:
    state = ToolLoopState()
    state.turn_tool_calls.append(("spotify_pause", '{"ok": true}'))
    state.tool_results.append('{"ok": true}')
    action = finalize_assistant_text(
        "I wasn't able to run the Spotify action just now.",
        state,
        user_text="pause",
    )
    assert action.kind == "reprompt"
    assert "tool" in action.reprompt_user_content.lower()


def test_api_keys_redacted_from_error_text() -> None:
    secret = "sk-testkey1234567890abcdefghij"
    raw = f"Auth failed for {secret}"
    assert secret not in redact_known_api_keys(raw, [secret])


@respx.mock
def test_openai_adapter_tool_call_then_text(data_dir, signed_in_tokens) -> None:
    settings = Settings(openai_api_key="sk-test-openai-key", agent_max_steps=4)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = json.loads(request.content.decode())
        if calls["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "type": "function",
                                        "function": {
                                            "name": "spotify_pause",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Paused playback."}}]},
        )

    respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=handler)
    respx.put("https://api.spotify.com/v1/me/player/pause").mock(return_value=httpx.Response(204))
    text = run_chat_turn_openai_compat(
        "pause playback",
        settings,
        base_url="https://api.openai.com/v1",
        api_key="sk-test-openai-key",
        provider_id="openai",
    )
    assert "pause" in text.lower()
    assert "sk-test-openai-key" not in text


@respx.mock
def test_xai_adapter_uses_xai_host(data_dir, signed_in_tokens) -> None:
    settings = Settings(xai_api_key="xai-testkey1234567890", agent_max_steps=2)
    route = respx.post("https://api.x.ai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Hello from Grok."}}]},
        )
    )
    text = run_chat_turn_openai_compat(
        "hi",
        settings,
        base_url="https://api.x.ai/v1",
        api_key="xai-testkey1234567890",
        provider_id="xai",
    )
    assert route.called
    assert "grok" in text.lower()
    assert "xai-testkey" not in text


@respx.mock
def test_anthropic_adapter_tool_result(data_dir, signed_in_tokens) -> None:
    settings = Settings(anthropic_api_key="sk-ant-test-key-abcdefghij", agent_max_steps=4)
    step = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        step["n"] += 1
        if step["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_1",
                            "name": "spotify_pause",
                            "input": {"device_id": "default"},
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "Playback paused."}]},
        )

    respx.post("https://api.anthropic.com/v1/messages").mock(side_effect=handler)
    respx.put("https://api.spotify.com/v1/me/player/pause").mock(return_value=httpx.Response(204))
    text = run_chat_turn_anthropic("pause", settings)
    assert "pause" in text.lower()
    assert "sk-ant-test" not in text


@respx.mock
@pytest.mark.no_catalog_get_stub
def test_openai_error_never_echoes_api_key() -> None:
    secret = "sk-testkey1234567890abcdefghij"
    settings = Settings(openai_api_key=secret, agent_max_steps=1)
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(401, text=f"Invalid key {secret}")
    )
    text = run_chat_turn_openai_compat(
        "hi",
        settings,
        base_url="https://api.openai.com/v1",
        api_key=secret,
        provider_id="openai",
    )
    assert secret not in text
