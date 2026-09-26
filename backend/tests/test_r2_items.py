from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from spot_backend.action_claim_guard import (
    action_claim_honest_fallback,
    reply_claims_unbacked_action,
    tool_result_succeeded,
)
from spot_backend.agent import iter_ollama_chat_events
from spot_backend.app import app
from spot_backend.config import Settings
from spot_backend.gemini_llm import run_chat_turn_gemini
from spot_backend.ollama_agent_profile import SMALL_MODEL_TOOL_NAMES, small_model_route_hint
from spot_backend.setup_service import _ollama_cpu_profile
from spot_backend.spotify_tools import SpotifyToolRunner, pick_latest_album_release
from tests.recheck_helpers import install_cpu_only_ollama_profile_mocks, mock_artist_name_search
from fastapi.testclient import TestClient


@respx.mock
def test_r2_itemA_save_tracks_uses_me_library_put(data_dir, signed_in_tokens) -> None:
    track_id = "1111111111111111111111"
    route = respx.put("https://api.spotify.com/v1/me/library").mock(
        return_value=httpx.Response(200)
    )
    runner = SpotifyToolRunner(settings=Settings())
    runner._session_known_ids.add(track_id)
    raw = runner.run("spotify_save_tracks", {"track_ids": [track_id]})
    runner.close()
    assert json.loads(raw)["ok"] is True
    assert route.called
    assert route.calls.last.request.url.params["uris"] == f"spotify:track:{track_id}"


@respx.mock
def test_r2_itemA_follow_artist_uses_me_library_put(data_dir, signed_in_tokens) -> None:
    artist_id = "2222222222222222222222"
    route = respx.put("https://api.spotify.com/v1/me/library").mock(
        return_value=httpx.Response(200)
    )
    runner = SpotifyToolRunner(settings=Settings())
    runner._session_known_ids.add(artist_id)
    raw = runner.run("spotify_follow_artist", {"artist_id": artist_id})
    runner.close()
    assert json.loads(raw)["ok"] is True
    assert route.calls.last.request.url.params["uris"] == f"spotify:artist:{artist_id}"


@respx.mock
def test_r2_itemA_unsave_tracks_uses_me_library_delete(data_dir, signed_in_tokens) -> None:
    track_id = "3333333333333333333333"
    route = respx.delete("https://api.spotify.com/v1/me/library").mock(
        return_value=httpx.Response(200)
    )
    runner = SpotifyToolRunner(settings=Settings())
    runner.run("spotify_unsave_tracks", {"track_id": track_id})
    runner.close()
    assert route.calls.last.request.url.params["uris"] == f"spotify:track:{track_id}"


def test_r2_itemB_gemini_empty_turn_retries_then_succeeds(data_dir, signed_in_tokens) -> None:
    settings = Settings(gemini_api_key="test-key")
    responses = [
        {
            "candidates": [
                {
                    "finishReason": "MALFORMED_FUNCTION_CALL",
                    "content": {"parts": []},
                }
            ]
        },
        {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": "Here is your answer."}]},
                }
            ]
        },
    ]
    call_idx = 0

    def fake_post(_self, url, **kwargs):
        nonlocal call_idx
        body = responses[min(call_idx, len(responses) - 1)]
        call_idx += 1
        req = httpx.Request("POST", str(url))
        return httpx.Response(200, json=body, request=req)

    with patch("httpx.Client.post", fake_post):
        text = run_chat_turn_gemini("hello there", settings)
    assert text == "Here is your answer."
    assert call_idx == 2


def test_r2_itemC_action_guard_reprompts_when_claim_without_tool(data_dir, signed_in_tokens) -> None:
    settings = Settings(gemini_api_key="test-key")
    bodies: list[dict[str, Any]] = []

    def fake_post(_self, url, **kwargs):
        json_body = kwargs.get("json") or {}
        bodies.append(json_body)
        req = httpx.Request("POST", str(url))
        if len(bodies) == 1:
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {"parts": [{"text": "Now playing Creep by Radiohead."}]},
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
                        "content": {"parts": [{"text": action_claim_honest_fallback()}]},
                    }
                ]
            },
            request=req,
        )

    with patch("httpx.Client.post", fake_post):
        text = run_chat_turn_gemini("play music by Radiohead", settings)
    assert "wasn't able to run" in text or "can't confirm" in text
    assert len(bodies) >= 2


@respx.mock
def test_r2_itemC_play_playlist_resolves_artist_name(data_dir, signed_in_tokens) -> None:
    artist_id = "aaaaaaaaaaaaaaaaaaaaaa"
    mock_artist_name_search(artist_id, "Radiohead")
    play = respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={"is_playing": True, "context": {"uri": f"spotify:artist:{artist_id}"}, "item": {}},
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_play_playlist", {"playlist_id": "Radiohead"})
    runner.close()
    data = json.loads(raw)
    assert data.get("context_uri") == f"spotify:artist:{artist_id}"
    assert play.called


@respx.mock
def test_r2_itemD_create_playlist_forces_private_when_spotify_returns_public(
    data_dir, signed_in_tokens,
) -> None:
    pid = "pppppppppppppppppppppp"
    respx.post("https://api.spotify.com/v1/me/playlists").mock(
        return_value=httpx.Response(200, json={"id": pid, "name": "x", "public": True})
    )
    put_route = respx.put(f"https://api.spotify.com/v1/playlists/{pid}").mock(
        return_value=httpx.Response(200)
    )
    respx.get(f"https://api.spotify.com/v1/playlists/{pid}").mock(
        return_value=httpx.Response(200, json={"id": pid, "public": False, "name": "x"})
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_create_playlist", {"name": "Secret"})
    runner.close()
    data = json.loads(raw)
    assert put_route.called
    assert data.get("public") is False


def test_r2_itemF_cpu_only_always_shows_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    install_cpu_only_ollama_profile_mocks(monkeypatch)
    profile = _ollama_cpu_profile("http://127.0.0.1:11434", "qwen3:4b-instruct")
    assert profile["cpu_only"] is True
    assert profile.get("message")
    assert "CPU only" in profile["message"] or "cpu only" in profile["message"].lower()


def test_r2_itemH_small_model_includes_unfollow_playlist() -> None:
    assert "spotify_unfollow_playlist" in SMALL_MODEL_TOOL_NAMES


def test_r2_itemH_routes_delete_playlist_to_unfollow() -> None:
    assert small_model_route_hint("delete my playlist Workout") == "spotify_unfollow_playlist"


def test_r2_itemK_latest_album_picks_2025_deluxe_over_2022_original() -> None:
    items = [
        {"name": "Original", "release_date": "2022-06-01", "album_type": "album"},
        {"name": "Deluxe", "release_date": "2025-01-10", "album_type": "album"},
    ]
    latest = pick_latest_album_release(items)
    assert latest is not None
    assert latest["name"] == "Deluxe"


@respx.mock
def test_r2_itemK_artist_latest_album_tool(data_dir, signed_in_tokens) -> None:
    artist_id = "aaaaaaaaaaaaaaaaaaaaaa"
    respx.get(f"https://api.spotify.com/v1/artists/{artist_id}/albums").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {"name": "Old", "release_date": "2020-01-01"},
                    {"name": "New deluxe", "release_date": "2025-03-01"},
                ]
            },
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    runner._session_known_ids.add(artist_id)
    raw = runner.run("spotify_artist_latest_album", {"artist_id": artist_id})
    runner.close()
    data = json.loads(raw)
    assert data["latest_album"]["name"] == "New deluxe"


@respx.mock
def test_r2_itemL_devices_maps_spotify_500_to_502(data_dir, signed_in_tokens) -> None:
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    client = TestClient(app)
    r = client.get("/api/devices")
    assert r.status_code == 502


@respx.mock
def test_r2_itemL_devices_maps_network_error_to_503(data_dir, signed_in_tokens) -> None:
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(side_effect=httpx.ConnectError("down"))
    client = TestClient(app)
    r = client.get("/api/devices")
    assert r.status_code == 503


def test_r2_itemC_tool_success_detection() -> None:
    assert tool_result_succeeded(
        "spotify_play_playlist",
        json.dumps({"ok": True, "context_uri": "spotify:artist:x"}),
    )
    assert not reply_claims_unbacked_action(
        "Now playing Creep",
        {"spotify_play_playlist"},
    )
    assert reply_claims_unbacked_action("Now playing Creep", set())
