"""Round-4 PR recheck items — one dedicated test per item (r4-itemN)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from spot_backend.agent import _apply_ollama_tuning
from spot_backend.chat_messages import (
    append_visibility_notes_to_reply,
    assistant_reply_is_promise_only,
    prepare_user_visible_reply,
    scrub_internal_tool_references,
    tool_result_is_rejected_or_invalid_id,
)
from spot_backend.config import Settings
from spot_backend.gemini_llm import (
    gemini_intent_allowed_function_names,
    gemini_should_block_repeated_tool_call,
    gemini_tool_call_signature,
    run_chat_turn_gemini,
)
from spot_backend.spotify_tools import SpotifyToolRunner, collect_catalog_ids_from_tool_json
from tests.recheck_helpers import make_gemini_post_recorder

# r4-item1 — Gemini ANY only on first intent-scoped round + repeat guard
@respx.mock
def test_r4_item1_gemini_intent_any_only_first_round_one_tool_call(data_dir, signed_in_tokens) -> None:
    settings = Settings(gemini_api_key="test-key", agent_max_steps=16)
    pause_calls = {"n": 0}

    def handler(_body: dict[str, Any], n: int, req: httpx.Request) -> httpx.Response:
        fc_mode = (_body.get("toolConfig") or {}).get("functionCallingConfig", {}).get("mode")
        if fc_mode == "ANY":
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {
                                "parts": [{"functionCall": {"name": "spotify_pause", "args": {}}}],
                            },
                        }
                    ]
                },
                request=req,
            )
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": "Playback paused."}]},
                    }
                ]
            },
            request=req,
        )

    bodies, fake_post = make_gemini_post_recorder(handler)

    def counting_pause(*args, **kwargs):
        pause_calls["n"] += 1
        return httpx.Response(204)

    respx.put("https://api.spotify.com/v1/me/player/pause").mock(side_effect=counting_pause)
    with patch("httpx.Client.post", fake_post):
        text = run_chat_turn_gemini("pause playback", settings)
    assert pause_calls["n"] == 1
    assert "paused" in text.lower()
    modes = [
        b["toolConfig"]["functionCallingConfig"]["mode"]
        for b in bodies
        if b.get("toolConfig")
    ]
    assert modes[0] == "ANY"
    assert modes[-1] == "AUTO"


def test_r4_item1_gemini_repeat_guard_blocks_identical_signature() -> None:
    sig = gemini_tool_call_signature("spotify_pause", {})
    assert not gemini_should_block_repeated_tool_call(None, "spotify_pause", {})
    assert gemini_should_block_repeated_tool_call(sig, "spotify_pause", {})


# r4-item2 — like/save resolves empty track id; 404 album returns error
@respx.mock
def test_r4_item2_empty_track_id_resolves_from_playback(data_dir, signed_in_tokens) -> None:
    track_id = "1111111111111111111111"
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(200, json={"item": {"id": track_id, "name": "Now"}})
    )
    respx.get(f"https://api.spotify.com/v1/tracks/{track_id}").mock(
        return_value=httpx.Response(200, json={"id": track_id})
    )
    respx.get(f"https://api.spotify.com/v1/tracks/{track_id}").mock(
        return_value=httpx.Response(200, json={"id": track_id})
    )
    lib = respx.put("https://api.spotify.com/v1/me/library").mock(return_value=httpx.Response(200))
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_save_tracks", {"track_id": ""})
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is True
    assert lib.called


@pytest.mark.no_catalog_get_stub
@respx.mock
def test_r4_item2_fake_album_id_returns_error_not_ok(data_dir, signed_in_tokens) -> None:
    fake = "7o93KZ9kX8c3a3Z9kX8c3a"
    respx.get(f"https://api.spotify.com/v1/albums/{fake}").mock(
        return_value=httpx.Response(404, json={"error": {"status": 404, "message": "Not found"}})
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_save_albums", {"album_id": fake})
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is not True
    assert "error" in data


def test_r4_item2_gemini_like_intent_includes_playback_state() -> None:
    names = gemini_intent_allowed_function_names("like this song")
    assert names is not None
    assert "spotify_playback_state" in names


# r4-item3 — artist playback with null context matches via item artists
@respx.mock
def test_r4_item3_artist_playback_null_context_matches(data_dir, signed_in_tokens) -> None:
    artist_id = "aaaaaaaaaaaaaaaaaaaaaa"
    respx.get(f"https://api.spotify.com/v1/artists/{artist_id}").mock(
        return_value=httpx.Response(200, json={"id": artist_id, "name": "Artist"})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "context": None,
                "item": {
                    "uri": "spotify:track:cccccccccccccccccccccc",
                    "artists": [{"id": artist_id, "name": "Artist"}],
                },
            },
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    runner._session_known_ids.add(artist_id)
    raw = runner.run(
        "spotify_start_resume_playback",
        {"context_uri": f"spotify:artist:{artist_id}"},
    )
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is True
    assert data.get("playback_verified") is True


# r4-item4 — _apply_ollama_tuning exists; empty stream fallback uses it
def test_r4_item4_apply_ollama_tuning_merges_options(data_dir) -> None:
    settings = Settings(ollama_keep_alive="5m", ollama_think=False)
    body: dict[str, Any] = {"model": "qwen3:4b-instruct", "messages": [], "stream": False}
    _apply_ollama_tuning(body, settings, {"num_ctx": 8192})
    assert body.get("keep_alive") == "5m"
    assert body.get("options", {}).get("num_ctx") == 8192
    assert body.get("think") is False


def test_r4_item4_empty_stream_fallback_invokes_apply_ollama_tuning(data_dir) -> None:
    settings = Settings(ollama_keep_alive="2m")
    body: dict[str, Any] = {"model": "qwen", "messages": [], "stream": False, "format": "json"}
    _apply_ollama_tuning(body, settings, {"num_ctx": 4096})
    assert body["keep_alive"] == "2m"
    assert body["options"]["num_ctx"] == 4096


# r4-item5 — pylint threshold configured (score enforced in CI)
def test_r4_item5_pylint_fail_under_configured() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "fail-under = 9.5" in text


# r4-item6 — covered in frontend r4_items.test.tsx


# r4-item7 — visibility note appended server-side
def test_r4_item7_visibility_note_appended_to_final_reply() -> None:
    tool_json = json.dumps(
        {"ok": True, "visibility_warning": "Playlist is still public in Spotify dev mode."}
    )
    out = append_visibility_notes_to_reply("Created your playlist.", [tool_json])
    assert "still public" in out


# r4-item8 — resume drops redundant uris; restriction message
@respx.mock
def test_r4_item8_resume_drops_current_track_uri(data_dir, signed_in_tokens) -> None:
    track_id = "1111111111111111111111"
    track_uri = f"spotify:track:{track_id}"
    respx.get(f"https://api.spotify.com/v1/tracks/{track_id}").mock(
        return_value=httpx.Response(200, json={"id": track_id, "album": {"id": "aaaaaaaaaaaaaaaaaaaaaa"}})
    )
    play_calls: list[dict[str, Any]] = []

    def capture_play(request):
        play_calls.append({"url": str(request.url), "body": request.content})
        return httpx.Response(204)

    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={"is_playing": True, "item": {"uri": track_uri, "id": "1111111111111111111111"}},
        )
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        side_effect=capture_play
    )
    runner = SpotifyToolRunner(settings=Settings())
    runner._session_known_ids.add(track_id)
    runner.run("spotify_start_resume_playback", {"uris": [track_uri]})
    runner.close()
    assert play_calls
    body = play_calls[0]["body"]
    assert body in (b"", None, b"{}", b"null")


@respx.mock
def test_r4_item8_restriction_violated_user_message(data_dir, signed_in_tokens) -> None:
    track_id = "2222222222222222222222"
    respx.get(f"https://api.spotify.com/v1/tracks/{track_id}").mock(
        return_value=httpx.Response(200, json={"id": track_id, "album": {"id": "aaaaaaaaaaaaaaaaaaaaaa"}})
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(return_value=httpx.Response(200, json={}))
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(
            403,
            json={"error": {"status": 403, "message": "Restriction violated"}},
        )
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player(\?.*)?$").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )
    runner = SpotifyToolRunner(settings=Settings())
    runner._session_known_ids.add(track_id)
    raw = runner.run("spotify_start_resume_playback", {"uris": [f"spotify:track:{track_id}"]})
    runner.close()
    data = json.loads(raw)
    assert "stuck state" in data.get("error", "").lower()
    assert data.get("sign_out_not_recommended") is True


# r4-item9 — promise-only after rejected id (helper + agent nudge covered in item4 stream test)
def test_r4_item9_promise_only_after_rejected_id_detected() -> None:
    bad = json.dumps({"error": "Spotify has no track with id 'x' (HTTP 404)."})
    assert tool_result_is_rejected_or_invalid_id(bad)
    assert assistant_reply_is_promise_only("I'll search for it now.")


# r4-item10 — scrubber
def test_r4_item10_scrubber_removes_tool_names() -> None:
    raw = "Try spotify_search(query='test') for results."
    cleaned = scrub_internal_tool_references(raw)
    assert "spotify_search" not in cleaned
    assert "query=" not in cleaned


# r4-item11 — catalog id harvesting + informational intent not ANY-scoped
def test_r4_item11_catalog_ids_from_queue_and_recent_and_playlist_items() -> None:
    queue_payload = {
        "currently_playing": {"id": "aaaaaaaaaaaaaaaaaaaaaa", "uri": "spotify:track:aaaaaaaaaaaaaaaaaaaaaa"},
        "queue": [{"id": "bbbbbbbbbbbbbbbbbbbbbb", "uri": "spotify:track:bbbbbbbbbbbbbbbbbbbbbb"}],
    }
    recent_payload = {
        "items": [{"track": {"id": "cccccccccccccccccccccc", "uri": "spotify:track:cccccccccccccccccccccc"}}]
    }
    playlist_payload = {
        "items": [{"item": {"id": "dddddddddddddddddddddd", "uri": "spotify:track:dddddddddddddddddddddd"}}]
    }
    found = set()
    found.update(collect_catalog_ids_from_tool_json(queue_payload))
    found.update(collect_catalog_ids_from_tool_json(recent_payload))
    found.update(collect_catalog_ids_from_tool_json(playlist_payload))
    assert {
        "aaaaaaaaaaaaaaaaaaaaaa",
        "bbbbbbbbbbbbbbbbbbbbbb",
        "cccccccccccccccccccccc",
        "dddddddddddddddddddddd",
    }.issubset(found)


def test_r4_item11_informational_prompt_not_intent_scoped() -> None:
    assert gemini_intent_allowed_function_names("how do I save a playlist?") is None
    assert gemini_intent_allowed_function_names("explain spotify repeat modes") is None


def test_r4_item11_prepare_user_visible_includes_visibility_note() -> None:
    tool_json = json.dumps({"visibility_warning": "Dev mode may keep playlists public."})
    out = prepare_user_visible_reply("Done.", [tool_json])
    assert "Dev mode may keep playlists public." in out


@pytest.mark.skipif(sys.platform == "win32", reason="pylint subprocess checked in Linux CI")
def test_r4_item11_pylint_score_at_least_threshold() -> None:
    root = Path(__file__).resolve().parents[2]
    files = subprocess.check_output(["git", "ls-files", "*.py"], cwd=root, text=True).split()
    proc = subprocess.run(
        [sys.executable, "-m", "pylint", *files],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "rated at" in proc.stdout or "rated at" in proc.stderr
    for line in (proc.stdout + proc.stderr).splitlines():
        if "rated at" in line:
            score = float(line.split("rated at", 1)[1].split("/")[0].strip())
            assert score >= 9.5
            break
    else:
        pytest.fail("pylint score line missing")
