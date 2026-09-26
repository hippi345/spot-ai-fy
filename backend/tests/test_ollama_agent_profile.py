from __future__ import annotations

import logging

import pytest

from spot_backend.config import Settings
from spot_backend.llm_prefs import write_ollama_small_model_mode
from spot_backend.ollama_agent_profile import (
    SMALL_MODEL_SYSTEM_PROMPT,
    apply_ollama_request_tuning,
    estimate_prompt_tokens,
    filter_ollama_tools,
    maybe_warn_prompt_exceeds_ctx,
    parse_ollama_model_billions,
    should_retry_ollama_without_think,
    use_small_model_mode,
)
from spot_backend.spotify_tools import OLLAMA_TOOLS


def test_parse_ollama_model_billions() -> None:
    assert parse_ollama_model_billions("qwen3:4b-instruct") == 4.0
    assert parse_ollama_model_billions("llama3.1:8b") == 8.0


def test_small_model_auto_and_override(data_dir) -> None:
    settings = Settings(data_dir=data_dir)
    assert use_small_model_mode(settings, "qwen3:4b-instruct") is True
    write_ollama_small_model_mode(data_dir, "off")
    assert use_small_model_mode(settings, "qwen3:4b-instruct") is False
    write_ollama_small_model_mode(data_dir, "on")
    assert use_small_model_mode(settings, "qwen3:70b-instruct") is True


def test_small_model_tool_subset() -> None:
    small = filter_ollama_tools(OLLAMA_TOOLS, small=True)
    names = {t["function"]["name"] for t in small}
    assert names == {
        "spotify_search",
        "spotify_play_playlist",
        "spotify_start_resume_playback",
        "spotify_create_playlist",
        "spotify_top_artists",
        "spotify_top_tracks",
        "spotify_pause",
        "spotify_skip_next",
        "spotify_skip_previous",
        "spotify_add_to_queue",
        "spotify_devices",
        "spotify_playback_state",
        "spotify_user_playlists",
        "spotify_add_tracks_by_query",
        "spotify_set_volume",
        "spotify_set_shuffle",
        "spotify_set_repeat",
        "spotify_unfollow_playlist",
    }


def test_small_model_prompt_under_4k_tokens() -> None:
    small_tools = filter_ollama_tools(OLLAMA_TOOLS, small=True)
    messages = [{"role": "system", "content": SMALL_MODEL_SYSTEM_PROMPT}, {"role": "user", "content": "hi"}]
    est = estimate_prompt_tokens(messages, small_tools)
    assert est < 4000


def test_prompt_exceeds_ctx_warning(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="spot_backend.ollama_agent_profile")
    huge = "x" * 40000
    maybe_warn_prompt_exceeds_ctx([{"role": "system", "content": huge}], OLLAMA_TOOLS, 8192)
    assert any("exceeds num_ctx" in r.message for r in caplog.records)


def test_apply_ollama_request_tuning_think_and_threads() -> None:
    settings = Settings(ollama_think=False, ollama_num_thread=4)
    body: dict = {}
    apply_ollama_request_tuning(body, settings, {"num_ctx": 16384, "num_thread": 4})
    assert body["think"] is False
    assert body["options"]["num_thread"] == 4


def test_should_retry_without_think_on_400() -> None:
    body = {"think": False}
    assert should_retry_ollama_without_think(400, "unknown field think", body) is True
    assert should_retry_ollama_without_think(200, "", body) is False
