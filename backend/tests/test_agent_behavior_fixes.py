"""Multi-turn agent behavior fixes (mocked LLM + Spotify)."""

from __future__ import annotations

import json
import re
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from spot_backend.action_claim_guard import (
    numeric_factual_claim_honest_fallback,
    reply_claims_unbacked_action,
    reply_contains_unbacked_numeric_factual_claim,
)
from spot_backend.agent import run_chat_turn_ollama
from spot_backend.prompt_intent import OLLAMA_VAGUE_PLAYLIST_PLAY_NUDGE
from spot_backend.spotify_tools import SpotifyToolRunner
from spot_backend.anthropic_llm import run_chat_turn_anthropic
from spot_backend.config import Settings
from spot_backend.gemini_llm import run_chat_turn_gemini
from spot_backend.llm_tool_loop import ToolLoopState, finalize_assistant_text, run_tool_calls
from spot_backend.openai_compat_llm import run_chat_turn_openai_compat
from spot_backend.prompt_intent import prompt_is_capability_question, prompt_is_informational
from spot_backend.reply_tool_trace import append_tool_trace_record, tool_trace_log_path
from tests.recheck_helpers import FakeOllamaStream


def test_capability_question_is_informational_no_action_verb() -> None:
    q = "Can you play podcasts via this interface?"
    assert prompt_is_capability_question(q)
    assert prompt_is_informational(q)


def test_capability_question_blocks_playback_tool_refusal(data_dir, signed_in_tokens) -> None:
    settings = Settings()
    runner = SpotifyToolRunner(settings=settings)
    state = ToolLoopState()
    try:
        msgs = run_tool_calls(
            runner,
            [
                {
                    "id": "1",
                    "function": {
                        "name": "spotify_play_playlist",
                        "arguments": {"playlist_id": "abc12345678901234567890"},
                    },
                }
            ],
            state,
            informational_turn=True,
            user_text="Can you play podcasts via this interface?",
        )
        assert msgs
        payload = json.loads(msgs[0]["content"])
        assert payload.get("informational_refusal") or payload.get("error")
    finally:
        runner.close()


def test_failed_playback_claim_flagged() -> None:
    failed = ('spotify_play_track', '{"ok": false, "error": "nope"}')
    assert reply_claims_unbacked_action(
        "Playing his latest single now.",
        set(),
        user_text="play his latest single",
        turn_tool_calls=[failed],
    )


def test_finalize_honest_fallback_after_failed_play_claim() -> None:
    state = ToolLoopState()
    state.turn_tool_calls.append(("spotify_play_track", '{"ok": false}'))
    state.tool_results.append('{"ok": false}')
    action = finalize_assistant_text(
        "Playing Blinding Lights for you.",
        state,
        user_text="play it",
    )
    assert action.kind in ("reprompt", "return")
    if action.kind == "return":
        assert "wasn't able" in action.text.lower() or "can't confirm" in action.text.lower()


@respx.mock
def test_openai_adapter_includes_full_chat_history(data_dir, signed_in_tokens) -> None:
    settings = Settings(openai_api_key="sk-test-openai-key", agent_max_steps=2)
    seen: list[list[dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen.append(body.get("messages") or [])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "The Weeknd is an artist."}}]},
        )

    respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=handler)
    history = [
        {"role": "user", "content": "Tell me about The Weeknd"},
        {"role": "assistant", "content": "The Weeknd is a Canadian artist."},
    ]
    with patch(
        "spot_backend.openai_compat_llm.gemini_deterministic_shortcut_reply",
        return_value=None,
    ):
        run_chat_turn_openai_compat(
            "Can you play his latest single?",
            settings,
            base_url="https://api.openai.com/v1",
            api_key="sk-test-openai-key",
            provider_id="openai",
            history=history,
        )
    assert seen
    roles = [m.get("role") for m in seen[0]]
    assert roles.count("user") >= 2
    assert any("Weeknd" in str(m.get("content", "")) for m in seen[0])


@respx.mock
def test_anthropic_adapter_includes_full_chat_history(data_dir, signed_in_tokens) -> None:
    settings = Settings(anthropic_api_key="sk-ant-test-key-abcdefghij", agent_max_steps=2)
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "Sure."}]},
        )

    respx.post("https://api.anthropic.com/v1/messages").mock(side_effect=handler)
    history = [{"role": "user", "content": "We were talking about The Weeknd"}]
    with patch(
        "spot_backend.anthropic_llm.gemini_deterministic_shortcut_reply",
        return_value=None,
    ):
        run_chat_turn_anthropic(
            "Play his latest single",
            settings,
            history=history,
        )
    assert seen
    msgs = seen[0].get("messages") or []
    assert len(msgs) >= 2
    assert msgs[0]["content"] == "We were talking about The Weeknd"


@respx.mock
def test_gemini_adapter_includes_full_chat_history(data_dir, signed_in_tokens) -> None:
    settings = Settings(gemini_api_key="gemini-test-key-1234567890", agent_max_steps=2)
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": "Okay."}]},
                    }
                ]
            },
        )

    respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/.*").mock(
        side_effect=handler
    )
    history = [{"role": "user", "content": "The Weeknd"}]
    with patch(
        "spot_backend.gemini_llm.gemini_deterministic_shortcut_reply",
        return_value=None,
    ):
        run_chat_turn_gemini("play his latest single", settings, history=history)
    assert seen
    contents = seen[0].get("contents") or []
    assert len(contents) >= 2


def test_ollama_failed_playback_claim_uses_honest_fallback(data_dir, signed_in_tokens) -> None:
    settings = Settings(agent_max_steps=3)
    fail_json = '{"ok": false, "error": "playback failed"}'
    streams = [
        FakeOllamaStream(
            [
                json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "spotify_play_track",
                                        "arguments": {"track_id": "aaaaaaaaaaaaaaaaaaaaa1"},
                                    }
                                }
                            ],
                        },
                        "done": True,
                    }
                )
            ]
        ),
        FakeOllamaStream(
            [
                json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Playing his latest single now.",
                        },
                        "done": True,
                    }
                )
            ]
        ),
    ]
    idx = {"i": 0}

    def fake_stream(_self, _method, _url, **kwargs):
        i = idx["i"]
        idx["i"] += 1
        return streams[min(i, len(streams) - 1)]

    with patch("httpx.Client.stream", fake_stream), patch(
        "spot_backend.agent.ollama_deterministic_shortcut_events",
        return_value=None,
    ), patch.object(SpotifyToolRunner, "run", return_value=fail_json):
        reply = run_chat_turn_ollama("play his latest single", settings)
    low = reply.lower()
    assert any(
        phrase in low
        for phrase in (
            "wasn't able",
            "can't confirm",
            "unable",
            "playback failed",
            "didn't confirm",
            "something went wrong",
        )
    )


def test_ollama_vague_playlist_nudge_after_user_playlists_only(
    data_dir, signed_in_tokens
) -> None:
    settings = Settings(agent_max_steps=4)
    playlists_json = json.dumps({"ok": True, "playlists": [{"id": "p1", "name": "Evening Acoustic"}]})
    streams = [
        FakeOllamaStream(
            [
                json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "spotify_user_playlists",
                                        "arguments": {},
                                    }
                                }
                            ],
                        },
                        "done": True,
                    }
                )
            ]
        ),
        FakeOllamaStream(
            [
                json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Playing Evening Acoustic for you.",
                        },
                        "done": True,
                    }
                )
            ]
        ),
    ]
    bodies: list[dict[str, Any]] = []
    idx = {"i": 0}

    def fake_stream(_self, _method, _url, **kwargs):
        bodies.append(kwargs["json"])
        i = idx["i"]
        idx["i"] += 1
        return streams[min(i, len(streams) - 1)]

    def fake_run(_self, name: str, arguments: dict[str, Any] | None = None, **kwargs: Any) -> str:
        if name == "spotify_user_playlists":
            return playlists_json
        return '{"ok": true}'

    with patch("httpx.Client.stream", fake_stream), patch(
        "spot_backend.agent.ollama_deterministic_shortcut_events",
        return_value=None,
    ), patch.object(SpotifyToolRunner, "run", fake_run):
        run_chat_turn_ollama("play one of my playlists", settings)

    assert len(bodies) >= 2
    second_msgs = bodies[1]["messages"]
    nudge_msgs = [
        m
        for m in second_msgs
        if m.get("role") == "user" and OLLAMA_VAGUE_PLAYLIST_PLAY_NUDGE in str(m.get("content"))
    ]
    assert nudge_msgs, "Expected vague-playlist nudge user message before second model round"


def test_ollama_adapter_includes_full_chat_history(data_dir, signed_in_tokens) -> None:
    settings = Settings()
    history = [
        {"role": "user", "content": "Let's talk about The Weeknd"},
        {"role": "assistant", "content": "Sure — The Weeknd is a popular artist."},
    ]
    streams = [
        FakeOllamaStream(
            [
                json.dumps(
                    {
                        "message": {"role": "assistant", "content": "Playing the newest single."},
                        "done": True,
                    }
                )
            ]
        )
    ]
    bodies: list[dict[str, Any]] = []
    idx = {"i": 0}

    def fake_stream(_self, _method, _url, **kwargs):
        bodies.append(kwargs["json"])
        i = idx["i"]
        idx["i"] += 1
        return streams[min(i, len(streams) - 1)]

    with patch("httpx.Client.stream", fake_stream), patch(
        "spot_backend.agent.ollama_deterministic_shortcut_events",
        return_value=None,
    ):
        run_chat_turn_ollama("play his latest single", settings, history=history)
    assert bodies
    msgs = bodies[0]["messages"]
    joined = json.dumps(msgs)
    assert "Weeknd" in joined
    assert "play his latest single" in joined


@respx.mock
def test_artist_latest_album_prefers_single(data_dir, signed_in_tokens) -> None:
    artist_id = "0aHjOrDlHWSXDSF1DXJuUY"
    respx.get(url__regex=rf"https://api\.spotify\.com/v1/artists/{artist_id}/albums.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "name": "Old Album",
                        "release_date": "2020-01-01",
                        "album_type": "album",
                    },
                    {
                        "name": "Fresh Single",
                        "release_date": "2026-03-01",
                        "album_type": "single",
                    },
                ],
                "total": 2,
            },
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    try:
        raw = runner.run(
            "spotify_artist_latest_album",
            {"artist_id": artist_id, "prefer": "single", "include_groups": "album,single"},
        )
        data = json.loads(raw)
        assert data.get("ok") is True
        latest = data.get("latest_release") or data.get("latest_album") or {}
        assert latest.get("name") == "Fresh Single"
    finally:
        runner.close()


@respx.mock
def test_artist_albums_studio_count_paged_deduped(data_dir, signed_in_tokens) -> None:
    artist_id = "0aHjOrDlHWSXDSF1DXJuUY"
    page1 = [
        {"id": "a1", "name": "Take Care", "album_type": "album"},
        {"id": "a2", "name": "Take Care (Deluxe)", "album_type": "album"},
    ]
    page2 = [
        {"id": "a3", "name": "Views", "album_type": "album"},
        {"id": "a4", "name": "Views (Explicit)", "album_type": "album"},
    ]

    def albums_handler(request: httpx.Request) -> httpx.Response:
        url = httpx.URL(str(request.url))
        offset = int(url.params.get("offset") or 0)
        if offset == 0:
            items = page1
            total = 4
        else:
            items = page2
            total = 4
        return httpx.Response(200, json={"items": items, "total": total, "limit": 10, "offset": offset})

    respx.get(url__regex=rf"https://api\.spotify\.com/v1/artists/{artist_id}/albums.*").mock(
        side_effect=albums_handler
    )
    runner = SpotifyToolRunner(settings=Settings())
    try:
        raw = runner.run("spotify_artist_albums", {"artist_id": artist_id})
        data = json.loads(raw)
        assert data.get("studio_album_count_deduped") == 2
        assert (data.get("discography_counts") or {}).get("albums") == 2
    finally:
        runner.close()


@respx.mock
def test_play_artist_popular_track_picks_highest_popularity(data_dir, signed_in_tokens) -> None:
    artist_id = "aaaaaaaaaaaaaaaaaaaaaa"
    album_id = "bbbbbbbbbbbbbbbbbbbbbb"
    respx.get(f"https://api.spotify.com/v1/artists/{artist_id}").mock(
        return_value=httpx.Response(200, json={"id": artist_id, "name": "Band of Horses"})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [
                        {
                            "id": "t1",
                            "uri": "spotify:track:t111111111111111111111",
                            "name": "Lesser",
                            "popularity": 40,
                            "album": {"id": album_id, "uri": f"spotify:album:{album_id}"},
                            "artists": [{"id": artist_id, "name": "Band of Horses"}],
                        },
                        {
                            "id": "t2",
                            "uri": "spotify:track:t222222222222222222222",
                            "name": "The Funeral",
                            "popularity": 72,
                            "album": {"id": album_id, "uri": f"spotify:album:{album_id}"},
                            "artists": [{"id": artist_id, "name": "Band of Horses"}],
                        },
                    ],
                }
            },
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player.*").mock(
        return_value=httpx.Response(204)
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player.*").mock(
        return_value=httpx.Response(204)
    )
    runner = SpotifyToolRunner(settings=Settings())
    try:
        raw = runner.run(
            "spotify_play_artist_popular_track",
            {"artist_id": artist_id, "artist": "Band of Horses"},
        )
        data = json.loads(raw)
        assert (data.get("track") or {}).get("name") == "The Funeral"
    finally:
        runner.close()


def test_numeric_honesty_when_artist_albums_failed() -> None:
    failed = ("spotify_artist_albums", '{"error": "Spotify HTTP 400"}')
    assert reply_contains_unbacked_numeric_factual_claim(
        "Drake has 21 albums.",
        [failed],
    )
    fb = numeric_factual_claim_honest_fallback().lower()
    assert "reliable number" in fb or "couldn't get" in fb
    assert not re.search(r"\d", numeric_factual_claim_honest_fallback())


def test_numeric_honesty_allows_backed_count() -> None:
    ok = (
        "spotify_artist_albums",
        '{"studio_album_count_deduped": 21, "discography_counts": {"albums": 21}}',
    )
    assert not reply_contains_unbacked_numeric_factual_claim(
        "Drake has 21 studio albums on Spotify.",
        [ok],
    )


def test_ollama_tool_trace_persists_refusal_and_run(data_dir, signed_in_tokens) -> None:
    settings = Settings(agent_max_steps=2)
    streams = [
        FakeOllamaStream(
            [
                json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "spotify_play_playlist",
                                        "arguments": {"playlist_id": "plist000000000000000001"},
                                    }
                                }
                            ],
                        },
                        "done": True,
                    }
                )
            ]
        ),
        FakeOllamaStream(
            [
                json.dumps(
                    {
                        "message": {"role": "assistant", "content": "No — podcasts are not supported."},
                        "done": True,
                    }
                )
            ]
        ),
    ]
    idx = {"i": 0}

    def fake_stream(_self, _method, _url, **kwargs):
        i = idx["i"]
        idx["i"] += 1
        return streams[min(i, len(streams) - 1)]

    with patch("httpx.Client.stream", fake_stream), patch(
        "spot_backend.agent.ollama_deterministic_shortcut_events",
        return_value=None,
    ):
        run_chat_turn_ollama(
            "Can you play podcasts via this interface?",
            settings,
        )
    path = tool_trace_log_path(data_dir)
    assert path.is_file()
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    playlist_rows = [
        json.loads(ln)
        for ln in lines
        if json.loads(ln).get("tool") == "spotify_play_playlist"
    ]
    assert playlist_rows
    assert playlist_rows[-1].get("outcome") == "refused"


def test_tool_trace_never_persists_api_key(data_dir) -> None:
    secret = "sk-supersecret-test-key-12345"
    append_tool_trace_record(
        data_dir,
        conversation_id="conv1",
        tool_name="spotify_me",
        args_summary=json.dumps({"note": secret}),
        outcome="ok",
        known_secrets=[secret],
    )
    path = tool_trace_log_path(data_dir)
    text = path.read_text(encoding="utf-8")
    assert secret not in text


def test_top_artists_time_range_wording_in_system_prompt() -> None:
    from spot_backend.agent_system_extras import SHARED_AGENT_BEHAVIOR_SUFFIX

    low = SHARED_AGENT_BEHAVIOR_SUFFIX.lower()
    assert "short_term" in low and "medium_term" in low
