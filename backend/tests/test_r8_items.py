"""Round-8 PR items — dedicated test_r8_itemN_* per requirement."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from spot_backend.agent import iter_ollama_chat_events
from spot_backend.config import Settings
from spot_backend.gemini_llm import (
    build_gemini_generate_content_body,
    run_chat_turn_gemini,
)
from spot_backend.prompt_intent import (
    SPOTIFY_MUTATING_TOOL_NAMES,
    SPOTIFY_READ_ONLY_TOOL_NAMES,
    informational_system_suffix,
    prompt_is_informational,
    prompt_is_pure_how_to,
)
from tests.recheck_helpers import (
    FakeOllamaStream,
    gemini_candidates_payload,
    gemini_stop_candidate,
    make_gemini_post_recorder,
)

R8_LIBRARY_INFORMATIONAL_PHRASES = (
    "what are my top artists?",
    "what have I been listening to lately?",
    "how many playlists do I have?",
    "who's on my most played list?",
    "what's my most recent liked song?",
)

PURE_HOW_TO_PHRASE = "how do I make a playlist private?"


def _ollama_tool_names(body: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for entry in body.get("tools") or []:
        fn = entry.get("function") if isinstance(entry, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.append(fn["name"])
    return names


def _gemini_declaration_names(body: dict[str, Any]) -> set[str]:
    decls = (body.get("tools") or [{}])[0].get("functionDeclarations") or []
    return {d.get("name") for d in decls if isinstance(d, dict) and d.get("name")}


def _capture_gemini_first_body(user_text: str) -> dict[str, Any]:
    settings = Settings(gemini_api_key="k")

    def handler(_body: dict[str, Any], _n: int, req: httpx.Request) -> httpx.Response:
        payload = gemini_candidates_payload(
            gemini_stop_candidate({"text": "Here is the answer."})
        )
        return httpx.Response(200, json=payload, request=req)

    bodies, fake_post = make_gemini_post_recorder(handler)
    with patch("httpx.Client.post", fake_post):
        run_chat_turn_gemini(user_text, settings)
    assert bodies
    return bodies[0]


def _capture_ollama_first_body(user_text: str) -> dict[str, Any]:
    settings = Settings()
    bodies: list[dict[str, Any]] = []
    stream = FakeOllamaStream(
        [json.dumps({"message": {"role": "assistant", "content": "OK"}, "done": True})]
    )

    def fake_stream(_self, _method, _url, **kwargs):
        bodies.append(kwargs["json"])
        return stream

    with patch("httpx.Client.stream", fake_stream):
        list(iter_ollama_chat_events(user_text, settings))
    assert bodies
    return bodies[0]


@pytest.mark.parametrize("phrase", R8_LIBRARY_INFORMATIONAL_PHRASES)
def test_r8_item1_informational_library_questions_readonly_tools_both_providers(
    data_dir, signed_in_tokens, phrase: str,
) -> None:
    assert prompt_is_informational(phrase) is True
    assert prompt_is_pure_how_to(phrase) is False

    gemini_body = _capture_gemini_first_body(phrase)
    gemini_names = _gemini_declaration_names(gemini_body)
    assert gemini_names
    assert not gemini_names & SPOTIFY_MUTATING_TOOL_NAMES
    assert "spotify_top_artists" in gemini_names or "spotify_recently_played" in gemini_names
    assert "toolConfig" in gemini_body

    ollama_body = _capture_ollama_first_body(phrase)
    ollama_names = set(_ollama_tool_names(ollama_body))
    assert ollama_names
    assert not ollama_names & SPOTIFY_MUTATING_TOOL_NAMES
    assert ollama_names <= SPOTIFY_READ_ONLY_TOOL_NAMES


def test_r8_item1_pure_how_to_offers_readonly_but_guidance_forbids_lookup(
    data_dir, signed_in_tokens,
) -> None:
    assert prompt_is_informational(PURE_HOW_TO_PHRASE) is True
    assert prompt_is_pure_how_to(PURE_HOW_TO_PHRASE) is True
    guidance = informational_system_suffix(PURE_HOW_TO_PHRASE).lower()
    assert "do not call" in guidance and "lookup" in guidance

    gemini_body = _capture_gemini_first_body(PURE_HOW_TO_PHRASE)
    assert not _gemini_declaration_names(gemini_body) & SPOTIFY_MUTATING_TOOL_NAMES

    ollama_body = _capture_ollama_first_body(PURE_HOW_TO_PHRASE)
    assert not set(_ollama_tool_names(ollama_body)) & SPOTIFY_MUTATING_TOOL_NAMES


def test_r8_item2_diag_payload_has_no_tool_config_without_declarations() -> None:
    empty_body = build_gemini_generate_content_body(
        system_text="sys",
        user_prompt="hi",
        decls=[],
        fc_mode="AUTO",
    )
    assert "toolConfig" not in empty_body
    assert "tools" not in empty_body

    decl = [{"name": "spotify_pause", "description": "x", "parameters": {"type": "object"}}]
    with_tools = build_gemini_generate_content_body(
        system_text="sys",
        user_prompt="pause",
        decls=decl,
        fc_mode="AUTO",
    )
    assert with_tools["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
    assert with_tools["tools"][0]["functionDeclarations"] == decl

    diag_no_tools = build_gemini_generate_content_body(
        system_text="test",
        user_prompt="Say hi",
        decls=[],
    )
    assert "toolConfig" not in diag_no_tools
