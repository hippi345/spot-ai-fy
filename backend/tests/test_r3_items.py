"""Round-3 recheck items — each test is named/commented with its item number."""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from spot_backend.action_claim_guard import reply_claims_unbacked_action
from spot_backend.config import Settings
from spot_backend.gemini_llm import (
    gemini_candidate_is_effectively_empty,
    gemini_intent_allowed_function_names,
    run_chat_turn_gemini,
)
from spot_backend.setup_service import _ollama_cpu_profile
from spot_backend.spotify_tools import SpotifyToolRunner, collect_catalog_ids_from_tool_json
from tests.recheck_helpers import (
    devices_api_get,
    gemini_thought_then_pause_then_text_handler,
    install_cpu_profile_http_mocks,
    make_gemini_post_recorder,
    mock_artist_name_search,
    mock_player_track_playing,
    run_artist_latest_album_tool,
    run_create_playlist_private_flow,
    run_verified_album_context_playback,
)


# r3_item01 — empty session: fake album id rejected; real id verified via GET
@pytest.mark.no_catalog_get_stub
@respx.mock
def test_r3_item01_empty_session_fake_album_id_rejected(data_dir, signed_in_tokens) -> None:
    fake = "7o93KZ9kX8c3a3Z9kX8c3a"
    respx.get(f"https://api.spotify.com/v1/albums/{fake}").mock(
        return_value=httpx.Response(404, json={"error": {"status": 404, "message": "Not found"}})
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run(
        "spotify_start_resume_playback",
        {"context_uri": f"spotify:album:{fake}"},
    )
    runner.close()
    data = json.loads(raw)
    assert "error" in data
    assert fake in data["error"] or "404" in data["error"]


@respx.mock
def test_r3_item01_empty_session_real_album_verified_via_get(data_dir, signed_in_tokens) -> None:
    album_id = "aaaaaaaaaaaaaaaaaaaaaa"
    data = run_verified_album_context_playback(album_id)
    assert data.get("ok") is True or "context_uri" in data


# r3_item02 — top-tracks ids remembered; play without extra GET
@respx.mock
def test_r3_item02_top_tracks_then_play_without_catalog_get(data_dir, signed_in_tokens) -> None:
    artist_id = "bbbbbbbbbbbbbbbbbbbbbb"
    track_id = "cccccccccccccccccccccc"
    respx.get(f"https://api.spotify.com/v1/artists/{artist_id}").mock(
        return_value=httpx.Response(200, json={"id": artist_id, "name": "Artist"})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [{"id": track_id, "uri": f"spotify:track:{track_id}", "name": "Hit"}],
                }
            },
        )
    )
    mock_player_track_playing(track_id)
    runner = SpotifyToolRunner(settings=Settings())
    runner.run("spotify_artist_top_tracks", {"artist_id": artist_id})
    assert track_id in runner._session_known_ids
    raw = runner.run("spotify_start_resume_playback", {"uris": [f"spotify:track:{track_id}"]})
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is True or "uris" in json.dumps(data)


def test_r3_item02_collect_catalog_ids_nested() -> None:
    payload = {
        "tracks": {"items": [{"id": "aaaaaaaaaaaaaaaaaaaaaa", "uri": "spotify:track:bbbbbbbbbbbbbbbbbbbbbb"}]},
        "latest_album": {"id": "cccccccccccccccccccccc"},
    }
    found = collect_catalog_ids_from_tool_json(payload)
    assert "aaaaaaaaaaaaaaaaaaaaaa" in found
    assert "bbbbbbbbbbbbbbbbbbbbbb" in found
    assert "cccccccccccccccccccccc" in found


# r3_item03 — Gemini empty STOP / thought-only → retry with ANY + intent mapping
@respx.mock
def test_r3_item03_gemini_empty_stop_retries_with_function_call(data_dir, signed_in_tokens) -> None:
    settings = Settings(gemini_api_key="test-key")
    bodies: list[dict[str, Any]] = []
    _, fake_post = make_gemini_post_recorder(
        gemini_thought_then_pause_then_text_handler(bodies)
    )

    respx.put("https://api.spotify.com/v1/me/player/pause").mock(return_value=httpx.Response(204))
    with patch("httpx.Client.post", fake_post):
        text = run_chat_turn_gemini("pause", settings)
    assert len(bodies) >= 2
    assert bodies[0]["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"] == ["spotify_pause"]
    assert "paused" in text.lower() or "Playback" in text


def test_r3_item03_gemini_intent_mapping_pause() -> None:
    names = gemini_intent_allowed_function_names("please pause playback")
    assert names == ["spotify_pause"]


def test_r3_item03_gemini_candidate_empty_detector() -> None:
    cand = {"finishReason": "STOP", "content": {"parts": []}}
    assert gemini_candidate_is_effectively_empty(cand)


# r3_item04 — action claim guard negatives / positives
def test_r3_item04_guard_ignores_how_to_play_by_artist() -> None:
    assert not reply_claims_unbacked_action(
        "how do I play music by an artist on Spotify?",
        set(),
    )


def test_r3_item04_guard_honest_unsave_reply_not_blocked() -> None:
    assert not reply_claims_unbacked_action(
        "Removed that track from your liked songs.",
        {"spotify_unsave_tracks"},
    )


def test_r3_item04_guard_positive_now_playing_still_detected() -> None:
    assert reply_claims_unbacked_action("Now playing Creep", set())


# r3_item05 — playlist create always PUT private
@respx.mock
def test_r3_item05_create_playlist_always_puts_private(data_dir, signed_in_tokens) -> None:
    pid = "pppppppppppppppppppppp"
    _data, put_called = run_create_playlist_private_flow(
        pid, post_public=False, get_public=False
    )
    assert put_called


@respx.mock
def test_r3_item05_create_playlist_visibility_warning_when_still_public(
    data_dir, signed_in_tokens,
) -> None:
    pid = "qqqqqqqqqqqqqqqqqqqqqq"
    data, _put_called = run_create_playlist_private_flow(
        pid, post_public=False, get_public=True
    )
    assert data.get("visibility_warning")
    assert data.get("public") is True


# r3_item07 — CPU profile warm-then-ps
def test_r3_item07_cpu_profile_loaded_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    install_cpu_profile_http_mocks(monkeypatch, http_calls=calls)
    profile = _ollama_cpu_profile("http://127.0.0.1:11434", "qwen3:4b-instruct")
    assert profile["cpu_only"] is True
    assert profile.get("message")
    assert calls[0].endswith("/api/generate")


def test_r3_item07_cpu_profile_loaded_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    install_cpu_profile_http_mocks(monkeypatch, size_vram=4_000_000_000)
    profile = _ollama_cpu_profile("http://127.0.0.1:11434", "qwen3:4b-instruct")
    assert profile["cpu_only"] is False
    assert profile.get("message") is None


def test_r3_item07_cpu_profile_unknown_shows_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url, *args, **kwargs):
        req = httpx.Request("POST", str(url))
        return httpx.Response(200, json={"response": "OK"}, request=req)

    def fake_get(url, *args, **kwargs):
        raise httpx.HTTPError("down")

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    profile = _ollama_cpu_profile("http://127.0.0.1:11434", "qwen3:4b-instruct")
    assert profile.get("message")


# r3_item10 — latest album prefers albums; artist exact name
@respx.mock
def test_r3_item10_latest_album_skips_feature_single(data_dir, signed_in_tokens) -> None:
    artist_id = "aaaaaaaaaaaaaaaaaaaaaa"
    data = run_artist_latest_album_tool(
        artist_id,
        [
            {
                "name": "is it cool? (feat. SZA)",
                "release_date": "2025-01-01",
                "album_type": "single",
            },
            {"name": "Real Album", "release_date": "2024-01-01", "album_type": "album"},
        ],
    )
    assert data["latest_album"]["name"] == "Real Album"


@respx.mock
def test_r3_item10_artist_search_exact_name_match(data_dir, signed_in_tokens) -> None:
    sza_id = "ssssssssssssssssssssss"
    walker_id = "wwwwwwwwwwwwwwwwwwwwww"
    mock_artist_name_search(
        sza_id,
        "SZA",
        extra_items=[
            {"id": walker_id, "name": "Summer Walker"},
            {"id": sza_id, "name": "SZA"},
        ],
    )
    runner = SpotifyToolRunner(settings=Settings())
    cid = runner._first_artist_id_from_search("SZA", "US")
    runner.close()
    assert cid == sza_id


# r3_item11 — unfollow resolves playlist name
@respx.mock
def test_r3_item11_unfollow_playlist_resolves_name(data_dir, signed_in_tokens) -> None:
    pid = "pppppppppppppppppppppp"
    respx.get("https://api.spotify.com/v1/me/playlists").mock(
        return_value=httpx.Response(
            200,
            json={"items": [{"id": pid, "name": "Workout"}]},
        )
    )
    route = respx.delete(f"https://api.spotify.com/v1/playlists/{pid}/followers").mock(
        return_value=httpx.Response(200)
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_unfollow_playlist", {"playlist_id": "Workout"})
    runner.close()
    assert route.called
    assert json.loads(raw)["playlist_id"] == pid


# r3_item12 — save this album from playback
@respx.mock
def test_r3_item12_save_this_album_from_playback(data_dir, signed_in_tokens) -> None:
    album_id = "aaaaaaaaaaaaaaaaaaaaaa"
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={"item": {"album": {"id": album_id, "name": "Current"}}},
        )
    )
    lib = respx.put("https://api.spotify.com/v1/me/library").mock(return_value=httpx.Response(200))
    runner = SpotifyToolRunner(settings=Settings())
    runner._session_known_ids.add(album_id)
    raw = runner.run("spotify_save_albums", {"album_id": "this album"})
    runner.close()
    assert lib.called
    assert json.loads(raw)["saved_album_ids"] == [album_id]


# r3_item13 — top-tracks endpoint never called
@respx.mock
def test_r3_item13_artist_top_tracks_never_calls_removed_endpoint(
    data_dir, signed_in_tokens,
) -> None:
    artist_id = "aaaaaaaaaaaaaaaaaaaaaa"
    removed = respx.get(f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks").mock(
        return_value=httpx.Response(200, json={"tracks": []})
    )
    respx.get(f"https://api.spotify.com/v1/artists/{artist_id}").mock(
        return_value=httpx.Response(200, json={"id": artist_id, "name": "A"})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(200, json={"tracks": {"items": []}})
    )
    runner = SpotifyToolRunner(settings=Settings())
    runner.run("spotify_artist_top_tracks", {"artist_id": artist_id})
    runner.close()
    assert not removed.called


# r3_item14 — library chunking + scopes message
@respx.mock
def test_r3_item14_library_put_chunks_above_forty(data_dir, signed_in_tokens) -> None:
    ids = [f"{i:022d}" for i in range(45)]
    routes = respx.put("https://api.spotify.com/v1/me/library").mock(
        return_value=httpx.Response(200)
    )
    runner = SpotifyToolRunner(settings=Settings())
    runner._session_known_ids.update(ids)
    runner.run("spotify_save_tracks", {"track_ids": ids})
    runner.close()
    assert routes.call_count == 2


@respx.mock
def test_r3_item14_scopes_appear_sufficient_on_ambiguous_403(data_dir, signed_in_tokens) -> None:
    pid = "pppppppppppppppppppppp"
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(
            200,
            json={"id": "me"},
        )
    )
    respx.get(f"https://api.spotify.com/v1/playlists/{pid}").mock(
        return_value=httpx.Response(
            403,
            json={"error": {"status": 403, "message": "Forbidden"}},
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_get_playlist", {"playlist_id": pid})
    runner.close()
    data = json.loads(raw)
    assert data.get("scopes_appear_sufficient") is True
