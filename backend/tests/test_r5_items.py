"""Round-5 PR items — dedicated test_r5_itemN_* per requirement."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from spot_backend.agent import iter_ollama_chat_events, run_chat_turn_ollama
from spot_backend.chat_messages import scrub_internal_tool_references
from spot_backend.config import Settings
from spot_backend.gemini_llm import gemini_intent_allowed_function_names, run_chat_turn_gemini
from spot_backend.prompt_intent import SPOTIFY_MUTATING_TOOL_NAMES, prompt_is_informational
from spot_backend.spotify_tools import SpotifyToolRunner
from tests.recheck_helpers import (
    FakeOllamaStream,
    gemini_candidates_payload,
    gemini_stop_candidate,
    make_gemini_post_recorder,
    mock_player_album_context_playing,
    run_playback_restriction_violated,
    run_unknown_device_playback_fallback,
)

HOW_TO_PHRASES = (
    "how do I make a playlist private?",
    "how can I play an artist?",
    "what does shuffle do?",
)


def _ollama_tool_names(body: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for entry in body.get("tools") or []:
        fn = entry.get("function") if isinstance(entry, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.append(fn["name"])
    return names


def _ollama_line_tool(name: str, args: dict[str, Any] | None = None) -> str:
    args = args or {}
    return json.dumps(
        {
            "message": {
                "role": "assistant",
                "tool_calls": [{"function": {"name": name, "arguments": args}}],
            },
            "done": True,
        }
    )


def _ollama_line_text(text: str) -> str:
    return json.dumps({"message": {"role": "assistant", "content": text}, "done": True})


# r5-item1 — informational prompts: no mutating tools offered or executed
@pytest.mark.parametrize("phrase", HOW_TO_PHRASES)
def test_r5_item1_ollama_informational_offers_no_mutating_tools(
    data_dir, signed_in_tokens, phrase: str,
) -> None:
    assert prompt_is_informational(phrase)
    settings = Settings()
    streams = [FakeOllamaStream([_ollama_line_text("In the Spotify app, open the playlist menu.")])]
    bodies: list[dict[str, Any]] = []

    def fake_stream(_self, _method, _url, **kwargs):
        bodies.append(kwargs["json"])
        return streams[0]

    with patch("httpx.Client.stream", fake_stream):
        list(iter_ollama_chat_events(phrase, settings))

    assert bodies
    offered = set(_ollama_tool_names(bodies[0]))
    assert not offered & SPOTIFY_MUTATING_TOOL_NAMES


@pytest.mark.parametrize("phrase", HOW_TO_PHRASES)
def test_r5_item1_ollama_informational_refuses_mutating_tool_call(
    data_dir, signed_in_tokens, phrase: str,
) -> None:
    settings = Settings()
    create_calls: list[str] = []
    streams = [
        FakeOllamaStream(
            [_ollama_line_tool("spotify_create_playlist", {"name": "x", "public": False})]
        ),
        FakeOllamaStream([_ollama_line_text("Make the playlist private from the ⋯ menu in Spotify.")]),
    ]
    idx = {"i": 0}

    def fake_stream(_self, _method, _url, **kwargs):
        i = idx["i"]
        idx["i"] += 1
        return streams[min(i, len(streams) - 1)]

    real_run = SpotifyToolRunner.run

    def guard_run(self, name, arguments):
        if name == "spotify_create_playlist":
            create_calls.append(name)
        return real_run(self, name, arguments)

    with patch("httpx.Client.stream", fake_stream):
        with patch.object(SpotifyToolRunner, "run", guard_run):
            out = run_chat_turn_ollama(phrase, settings)
    assert "spotify_create_playlist" not in create_calls
    assert out


@pytest.mark.parametrize("phrase", HOW_TO_PHRASES)
def test_r5_item1_gemini_informational_no_mutating_declarations(
    data_dir, signed_in_tokens, phrase: str,
) -> None:
    settings = Settings(gemini_api_key="k")

    def handler(body: dict[str, Any], _n: int, req: httpx.Request) -> httpx.Response:
        decls = (body.get("tools") or [{}])[0].get("functionDeclarations") or []
        names = {d.get("name") for d in decls if isinstance(d, dict)}
        assert not names & SPOTIFY_MUTATING_TOOL_NAMES
        payload = gemini_candidates_payload(
            gemini_stop_candidate(
                {"text": "Use the playlist ⋯ menu and choose Make secret."}
            )
        )
        return httpx.Response(200, json=payload, request=req)

    bodies, fake_post = make_gemini_post_recorder(handler)
    with patch("httpx.Client.post", fake_post):
        run_chat_turn_gemini(phrase, settings)
    assert bodies
    decls = (bodies[0].get("tools") or [{}])[0].get("functionDeclarations") or []
    names = {d.get("name") for d in decls if isinstance(d, dict)}
    assert not names & SPOTIFY_MUTATING_TOOL_NAMES


@pytest.mark.parametrize("phrase", HOW_TO_PHRASES)
def test_r5_item1_gemini_informational_refuses_mutating_function_call(
    data_dir, signed_in_tokens, phrase: str,
) -> None:
    settings = Settings(gemini_api_key="k")
    create_hits: list[str] = []

    def handler(_body: dict[str, Any], n: int, req: httpx.Request) -> httpx.Response:
        if n == 1:
            payload = gemini_candidates_payload(
                gemini_stop_candidate(
                    {"functionCall": {"name": "spotify_create_playlist", "args": {"name": "x"}}}
                )
            )
        else:
            payload = gemini_candidates_payload(
                gemini_stop_candidate({"text": "Use the playlist menu in Spotify to change visibility."})
            )
        return httpx.Response(200, json=payload, request=req)

    real_run = SpotifyToolRunner.run

    def guard_run(self, name, arguments):
        if name == "spotify_create_playlist":
            create_hits.append(name)
        return real_run(self, name, arguments)

    _, fake_post = make_gemini_post_recorder(handler)
    with patch("httpx.Client.post", fake_post):
        with patch.object(SpotifyToolRunner, "run", guard_run):
            run_chat_turn_gemini(phrase, settings)
    assert not create_hits


# r5-item2 — promise nudge continues Ollama + Gemini agent loops
@respx.mock
def test_r5_item2_ollama_promise_nudge_then_search_and_play(data_dir, signed_in_tokens) -> None:
    album_id = "7o93KZ9kX8c3a3Z9kX8c3a"
    good_album = "aaaaaaaaaaaaaaaaaaaaaa"
    settings = Settings()
    respx.get(f"https://api.spotify.com/v1/albums/{album_id}").mock(
        return_value=httpx.Response(404, json={"error": {"status": 404}})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "albums": {
                    "items": [{"id": good_album, "name": "Velvet Lanterns", "uri": f"spotify:album:{good_album}"}]
                }
            },
        )
    )
    respx.get(f"https://api.spotify.com/v1/albums/{good_album}").mock(
        return_value=httpx.Response(200, json={"id": good_album})
    )
    mock_player_album_context_playing(good_album)
    streams = [
        FakeOllamaStream(
            [
                _ollama_line_tool(
                    "spotify_start_resume_playback",
                    {"context_uri": f"spotify:album:{album_id}"},
                )
            ]
        ),
        FakeOllamaStream([_ollama_line_text("I'll search for the album now.")]),
        FakeOllamaStream(
            [
                _ollama_line_tool(
                    "spotify_search",
                    {"query": "Velvet Lanterns The Glass Foxes", "types": "album"},
                )
            ]
        ),
        FakeOllamaStream(
            [
                _ollama_line_tool(
                    "spotify_start_resume_playback",
                    {"context_uri": f"spotify:album:{good_album}"},
                )
            ]
        ),
        FakeOllamaStream([_ollama_line_text("Playing Velvet Lanterns now.")]),
    ]
    idx = {"i": 0}
    tool_names: list[str] = []

    def fake_stream(_self, _method, _url, **kwargs):
        i = idx["i"]
        idx["i"] += 1
        return streams[min(i, len(streams) - 1)]

    with patch("httpx.Client.stream", fake_stream):
        events = list(
            iter_ollama_chat_events(
                "play the album Velvet Lanterns by The Glass Foxes",
                settings,
            )
        )
    for ev in events:
        if ev.get("type") == "tool_start":
            tool_names.append(str(ev.get("name")))
    assert "spotify_search" in tool_names
    assert tool_names.count("spotify_start_resume_playback") >= 2


@respx.mock
def test_r5_item2_gemini_promise_nudge_then_search_and_play(data_dir, signed_in_tokens) -> None:
    bad = "7o93KZ9kX8c3a3Z9kX8c3a"
    good = "aaaaaaaaaaaaaaaaaaaaaa"
    settings = Settings(gemini_api_key="k")
    respx.get(f"https://api.spotify.com/v1/albums/{bad}").mock(
        return_value=httpx.Response(404, json={"error": {"status": 404}})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(
            200,
            json={"albums": {"items": [{"id": good, "uri": f"spotify:album:{good}"}]}},
        )
    )
    respx.get(f"https://api.spotify.com/v1/albums/{good}").mock(
        return_value=httpx.Response(200, json={"id": good})
    )
    mock_player_album_context_playing(good)
    seq = {"n": 0}
    tools_run: list[str] = []

    def handler(_body: dict[str, Any], _n: int, req: httpx.Request) -> httpx.Response:
        seq["n"] += 1
        step = seq["n"]
        if step == 1:
            part = {"functionCall": {"name": "spotify_start_resume_playback", "args": {"context_uri": f"spotify:album:{bad}"}}}
        elif step == 2:
            part = {"text": "I'll search for the album now."}
        elif step == 3:
            part = {"functionCall": {"name": "spotify_search", "args": {"query": "Velvet Lanterns", "types": "album"}}}
        elif step == 4:
            part = {
                "functionCall": {
                    "name": "spotify_start_resume_playback",
                    "args": {"context_uri": f"spotify:album:{good}"},
                }
            }
        else:
            part = {"text": "Now playing Velvet Lanterns."}
        payload = gemini_candidates_payload(gemini_stop_candidate(part))
        return httpx.Response(200, json=payload, request=req)

    real_run = SpotifyToolRunner.run

    def track_run(self, name, arguments):
        tools_run.append(name)
        return real_run(self, name, arguments)

    _, fake_post = make_gemini_post_recorder(handler)
    with patch("httpx.Client.post", fake_post):
        with patch.object(SpotifyToolRunner, "run", track_run):
            run_chat_turn_gemini("play the album Velvet Lanterns by The Glass Foxes", settings)
    assert "spotify_search" in tools_run
    assert tools_run.count("spotify_start_resume_playback") >= 2


# r5-item3 — user-facing scrub + no search on how-to play artist (Gemini)
def test_r5_item3_scrubber_removes_tool_and_parameter_phrasing() -> None:
    raw = "use the `Spotify` tool and set the `public` parameter to `False`"
    cleaned = scrub_internal_tool_references(raw)
    assert "parameter" not in cleaned.lower()
    assert "`" not in cleaned
    assert "public" not in cleaned.lower()


def test_r5_item3_gemini_how_to_play_artist_does_not_search(data_dir, signed_in_tokens) -> None:
    settings = Settings(gemini_api_key="k")
    search_calls: list[str] = []

    def handler(_body: dict[str, Any], _n: int, req: httpx.Request) -> httpx.Response:
        payload = gemini_candidates_payload(
            gemini_stop_candidate(
                {
                    "text": (
                        "Type something like “play Radiohead” here, or search for the artist "
                        "in the Spotify app and press Play."
                    )
                }
            )
        )
        return httpx.Response(200, json=payload, request=req)

    real_run = SpotifyToolRunner.run

    def guard(self, name, arguments):
        if name == "spotify_search":
            search_calls.append(str(arguments))
        return real_run(self, name, arguments)

    _, fake_post = make_gemini_post_recorder(handler)
    with patch("httpx.Client.post", fake_post):
        with patch.object(SpotifyToolRunner, "run", guard):
            run_chat_turn_gemini("how can I play an artist?", settings)
    assert not search_calls


# r5-item4 — unknown device_id falls back
@respx.mock
def test_r5_item4_unknown_device_id_falls_back_to_saved_device(data_dir, signed_in_tokens) -> None:
    settings = Settings()
    data, play_requests = run_unknown_device_playback_fallback(
        saved_device_id="real_device_abc",
        settings=settings,
    )
    assert data.get("ok") is True
    assert "device_123" in (data.get("device_fallback_note") or "")
    assert play_requests
    assert "device_id=real_device_abc" in str(play_requests[-1].url) or "real_device_abc" in str(
        play_requests[-1].url
    )


# r5-item5 — empty Ollama stream then non-stream JSON fallback (real agent path)
@respx.mock
def test_r5_item5_ollama_empty_stream_fallback_uses_tuning_and_tool(data_dir, signed_in_tokens) -> None:
    settings = Settings(ollama_keep_alive="9m", ollama_num_ctx=8192)
    tools_err = FakeOllamaStream([], status_code=400)
    empty = FakeOllamaStream([json.dumps({"message": {"role": "assistant", "content": ""}, "done": True})])
    ns_tool = {
        "message": {
            "role": "assistant",
            "content": "```json\n[{\"name\":\"spotify_me\",\"arguments\":{}}]\n```",
        }
    }
    posts: list[dict[str, Any]] = []
    stream_queue = [tools_err, empty]
    stream_i = {"n": 0}

    def fake_stream(_self, _method, _url, **kwargs):
        bodies.append(kwargs["json"])
        i = stream_i["n"]
        stream_i["n"] += 1
        stream = stream_queue[min(i, len(stream_queue) - 1)]
        if stream.status_code == 400:
            stream._err_body = json.dumps({"error": "does not support tools"})  # type: ignore[attr-defined]
        return stream

    def fake_post(_self, url, **kwargs):
        posts.append(kwargs["json"] or {})
        req = httpx.Request("POST", str(url))
        return httpx.Response(200, json=ns_tool, request=req)

    bodies: list[dict[str, Any]] = []

    _orig_read = FakeOllamaStream.read

    def read_with_err(self):
        if self.status_code == 400:
            return getattr(self, "_err_body", b'{"error":"does not support tools"}').encode()
        return _orig_read(self)

    with patch.object(FakeOllamaStream, "read", read_with_err):
        with patch("httpx.Client.stream", fake_stream):
            with patch("httpx.Client.post", fake_post):
                with patch.object(SpotifyToolRunner, "_me", return_value='{"id":"x"}'):
                    events = list(iter_ollama_chat_events("who am I?", settings))
    assert posts
    assert posts[0].get("keep_alive") == "9m"
    assert posts[0].get("options", {}).get("num_ctx") == 8192
    assert posts[0].get("stream") is False
    assert any(e.get("name") == "spotify_me" for e in events if e.get("type") == "tool_start")


# r5-item7 — intent mapping shuffle / repeat / bare play
def test_r5_item7_intent_bare_play_resume() -> None:
    names = gemini_intent_allowed_function_names("play")
    assert names == ["spotify_start_resume_playback"]


def test_r5_item7_intent_shuffle_on_off() -> None:
    assert gemini_intent_allowed_function_names("shuffle on") == ["spotify_set_shuffle"]
    assert gemini_intent_allowed_function_names("shuffle off") == ["spotify_set_shuffle"]


def test_r5_item7_intent_repeat_modes() -> None:
    assert gemini_intent_allowed_function_names("repeat") == ["spotify_set_repeat"]
    assert gemini_intent_allowed_function_names("repeat track") == ["spotify_set_repeat"]


# r5-item8 — Gemini ANY mode only when appropriate
def test_r5_item8_gemini_function_calling_mode_in_payload(data_dir, signed_in_tokens) -> None:
    settings = Settings(gemini_api_key="k")
    cases = [
        ("how do I save a playlist?", "AUTO"),
        ("pause", "ANY"),
        ("add a track then play my workout playlist", "AUTO"),
    ]
    for user_text, expected_mode in cases:
        bodies: list[dict[str, Any]] = []

        def handler(body: dict[str, Any], _n: int, req: httpx.Request) -> httpx.Response:
            bodies.append(body)
            if expected_mode == "ANY":
                part = {"functionCall": {"name": "spotify_pause", "args": {}}}
            else:
                part = {"text": "Here is how."}
            payload = gemini_candidates_payload(gemini_stop_candidate(part))
            return httpx.Response(200, json=payload, request=req)

        _, fake_post = make_gemini_post_recorder(handler)
        with patch("httpx.Client.post", fake_post):
            run_chat_turn_gemini(user_text, settings)
        mode = (bodies[0].get("toolConfig") or {}).get("functionCallingConfig", {}).get("mode")
        assert mode == expected_mode, user_text


# r5-item9 — sixteen short commands → exactly one tool call each
_GEMINI_16_COMMANDS: list[tuple[str, str]] = [
    ("play", "spotify_start_resume_playback"),
    ("pause", "spotify_pause"),
    ("skip", "spotify_skip_next"),
    ("previous", "spotify_skip_previous"),
    ("shuffle on", "spotify_set_shuffle"),
    ("shuffle off", "spotify_set_shuffle"),
    ("repeat track", "spotify_set_repeat"),
    ("volume 50", "spotify_set_volume"),
    ("favorite this song", "spotify_save_tracks"),
    ("save this album", "spotify_save_albums"),
    ("play artist Radiohead", "spotify_search"),
    ("play album OK Computer", "spotify_search"),
    ("queue this song next", "spotify_play_next"),
    ("what's playing", "spotify_playback_state"),
    ("what did I just play", "spotify_recently_played"),
    ("resume", "spotify_start_resume_playback"),
]


@respx.mock
def test_r5_item9_gemini_sixteen_commands_one_tool_each(data_dir, signed_in_tokens) -> None:
    settings = Settings(gemini_api_key="k", agent_max_steps=4)
    respx.put(url__regex=r"https://api\.spotify\.com/.*").mock(return_value=httpx.Response(204))
    respx.get(url__regex=r"https://api\.spotify\.com/.*").mock(
        return_value=httpx.Response(200, json={"items": [], "id": "x"})
    )
    for user_text, expected_tool in _GEMINI_16_COMMANDS:
        tool_calls: list[str] = []

        def handler(_body: dict[str, Any], n: int, req: httpx.Request) -> httpx.Response:
            if n == 1:
                part = {"functionCall": {"name": expected_tool, "args": {}}}
            else:
                part = {"text": "Done."}
            payload = gemini_candidates_payload(gemini_stop_candidate(part))
            return httpx.Response(200, json=payload, request=req)

        real_run = SpotifyToolRunner.run

        def once(self, name, arguments):
            tool_calls.append(name)
            return json.dumps({"ok": True})

        _, fake_post = make_gemini_post_recorder(handler)
        with patch("httpx.Client.post", fake_post):
            with patch.object(SpotifyToolRunner, "run", once):
                run_chat_turn_gemini(user_text, settings)
        assert tool_calls == [expected_tool], user_text


# r5-item10 — restriction violated returns clear error without transfer retry storm
@respx.mock
def test_r5_item10_restriction_violated_attempts_player_transfer(
    data_dir, signed_in_tokens,
) -> None:
    track_id = "3333333333333333333333"
    transfer = respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player(\?.*)?$").mock(
        return_value=httpx.Response(204)
    )
    data = run_playback_restriction_violated(track_id)
    assert not transfer.called
    assert "stuck state" in data.get("error", "").lower()
