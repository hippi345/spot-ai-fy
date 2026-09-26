from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from spot_backend.agent import (
    _accumulate_ollama_stream_message,
    _coerce_chat_history,
    iter_ollama_chat_events,
)
from spot_backend.app import app
from spot_backend.chat_messages import (
    FRIENDLY_SPOTIFY_GUIDANCE,
    friendly_reply_for_empty_model_output,
    is_unpersisted_assistant_fallback,
)
from spot_backend.config import Settings
from spot_backend.gemini_llm import iter_gemini_chat_events, run_chat_turn_gemini
from spot_backend.ollama_agent_profile import SMALL_MODEL_TOOL_NAMES
from spot_backend.setup_service import _ollama_cpu_profile
from spot_backend.spotify_client import DEFAULT_SCOPES
from spot_backend.spotify_tools import OLLAMA_TOOLS, SpotifyToolRunner, _parse_spotify_context_ref
from tests.recheck_helpers import (
    FakeOllamaStream,
    devices_api_get,
    install_cpu_profile_http_mocks,
    run_album_playlist_play,
    run_artist_null_context_playback,
    run_create_playlist_private_flow,
    run_playback_restriction_violated,
)
from spot_backend.token_store import DeviceSelection, load_device, save_device
from spot_backend.url_safety import validate_ollama_base_url
from fastapi.testclient import TestClient


def test_item01_ollama_stream_accumulates_content_when_final_chunk_empty(
    data_dir, signed_in_tokens,
) -> None:
    settings = Settings()
    lines = [
        json.dumps({"message": {"role": "assistant", "content": "Hello "}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": "world"}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}),
    ]
    streams = [FakeOllamaStream(lines)]
    bodies: list[dict[str, Any]] = []

    def fake_stream(_client_self, _method, _url, **kwargs):
        bodies.append(kwargs["json"])
        return streams[0]

    events: list[dict[str, Any]] = []
    with patch("httpx.Client.stream", fake_stream):
        events = list(iter_ollama_chat_events("hi", settings))

    final = next(e for e in events if e.get("type") == "final")
    assert final["text"] == "Hello world"


def test_item02_fallback_not_in_coerced_history(data_dir, signed_in_tokens) -> None:
    hist = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": friendly_reply_for_empty_model_output("hi")},
    ]
    out = _coerce_chat_history(hist)
    assert out == [{"role": "user", "content": "hi"}]
    assert is_unpersisted_assistant_fallback(FRIENDLY_SPOTIFY_GUIDANCE)


@respx.mock
def test_item03_single_track_play_uses_album_context(data_dir, signed_in_tokens) -> None:
    track_id = "1111111111111111111111"
    album_id = "2222222222222222222222"
    respx.get(f"https://api.spotify.com/v1/tracks/{track_id}").mock(
        return_value=httpx.Response(200, json={"id": track_id, "album": {"id": album_id}})
    )
    play_route = respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )
    player_calls = {"n": 0}

    def player_state(_request):
        player_calls["n"] += 1
        if player_calls["n"] == 1:
            item_uri = "spotify:track:9999999999999999999999"
        else:
            item_uri = f"spotify:track:{track_id}"
        return httpx.Response(
            200,
            json={
                "is_playing": True,
                "context": {"uri": f"spotify:album:{album_id}"},
                "item": {"uri": item_uri},
            },
        )

    respx.get("https://api.spotify.com/v1/me/player").mock(side_effect=player_state)
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run(
        "spotify_start_resume_playback",
        {"uris": [f"spotify:track:{track_id}"]},
    )
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is True
    sent_bodies = [
        json.loads(c.request.content or b"{}")
        for c in play_route.calls
        if c.request.content
    ]
    assert sent_bodies
    assert sent_bodies[0]["context_uri"] == f"spotify:album:{album_id}"
    assert sent_bodies[0]["offset"] == {"uri": f"spotify:track:{track_id}"}


@respx.mock
def test_item03_restriction_violated_returns_clear_error(data_dir, signed_in_tokens) -> None:
    track_id = "1111111111111111111111"
    data = run_playback_restriction_violated(track_id, album_id="2222222222222222222222")
    assert data.get("ok") is False
    assert "stuck state" in data.get("error", "").lower()


@respx.mock
def test_item04_play_playlist_routes_album_uri(data_dir, signed_in_tokens) -> None:
    album_id = "3333333333333333333333"
    data = run_album_playlist_play(album_id)
    assert data["context_uri"] == f"spotify:album:{album_id}"
    assert "spotify:playlist:spotify:album" not in json.dumps(data)


def test_item04_play_playlist_rejects_unknown_id(data_dir, signed_in_tokens) -> None:
    unknown = "bbbbbbbbbbbbbbbbbbbbbb"
    respx.get(f"https://api.spotify.com/v1/playlists/{unknown}").mock(
        return_value=httpx.Response(404, json={"error": {"status": 404, "message": "Not found"}})
    )
    runner = SpotifyToolRunner(settings=Settings())
    runner._session_known_ids.add("aaaaaaaaaaaaaaaaaaaaaa")
    raw = runner.run("spotify_play_playlist", {"playlist_id": unknown})
    runner.close()
    data = json.loads(raw)
    assert "error" in data
    assert "404" in data["error"] or "search" in data.get("hint", "").lower()


def test_item05_ipv6_loopback_url_keeps_brackets(monkeypatch: pytest.MonkeyPatch) -> None:
    import ipaddress

    monkeypatch.setattr(
        "spot_backend.url_safety.resolve_host_ips",
        lambda _h: [ipaddress.ip_address("::1")],
    )
    assert validate_ollama_base_url("http://[::1]:11434") == "http://[::1]:11434"


def test_item06_ipv4_mapped_metadata_blocked() -> None:
    import ipaddress

    from spot_backend.url_safety import _classify_ip

    mapped = ipaddress.ip_address("::ffff:169.254.169.254")
    assert _classify_ip(mapped) == "blocked"
    mapped_ll = ipaddress.ip_address("::ffff:169.254.1.5")
    assert _classify_ip(mapped_ll) == "blocked"
    mapped_rfc = ipaddress.ip_address("::ffff:192.168.1.10")
    assert _classify_ip(mapped_rfc) == "allowed_private"


def test_item07_default_scopes_include_library_and_follow() -> None:
    for scope in (
        "user-library-modify",
        "user-library-read",
        "user-follow-modify",
        "user-follow-read",
    ):
        assert scope in DEFAULT_SCOPES.split()


@respx.mock
def test_item07_follow_artist_missing_scope_maps_reauth(data_dir) -> None:
    from spot_backend.token_store import TokenBundle, save_tokens

    settings = Settings()
    save_tokens(
        settings.resolved_token_path,
        TokenBundle(access_token="tok", refresh_token="ref", expires_at=time.time() + 3600, scope="user-read-private"),
    )
    runner = SpotifyToolRunner(settings=settings)
    raw = runner.run("spotify_follow_artist", {"artist_id": "1111111111111111111111"})
    runner.close()
    data = json.loads(raw)
    assert data.get("stale_scopes_need_reauth") is True
    assert "user-follow-modify" in data.get("missing_scopes", [])


def test_item11_small_model_subset_includes_create_and_top_tools() -> None:
    assert "spotify_create_playlist" in SMALL_MODEL_TOOL_NAMES
    assert "spotify_top_artists" in SMALL_MODEL_TOOL_NAMES
    assert "spotify_top_tracks" in SMALL_MODEL_TOOL_NAMES


def test_item11_tool_descriptions_distinguish_playlists_vs_create() -> None:
    by_name = {t["function"]["name"]: t["function"]["description"] for t in OLLAMA_TOOLS}
    user_pl = by_name["spotify_user_playlists"].lower()
    create_pl = by_name["spotify_create_playlist"].lower()
    top = by_name["spotify_top_artists"].lower()
    assert "list" in user_pl
    assert "create" in create_pl
    assert "top artists" in top
    assert "create" not in user_pl or "not" in user_pl


def test_item12_keepalive_sse_payload() -> None:
    from spot_backend.chat_sse import sse_data

    assert "keepalive" in sse_data({"type": "keepalive", "message": "Still working…"})


def test_item13_cpu_profile_recommends_gemini_when_slow(monkeypatch: pytest.MonkeyPatch) -> None:
    install_cpu_profile_http_mocks(monkeypatch, perf_steps=[0.0, 9.0, 9.0, 9.0])
    profile = _ollama_cpu_profile("http://127.0.0.1:11434", "qwen3:4b-instruct")
    assert profile["cpu_only"] is True
    assert profile["recommend_gemini"] is True
    assert "Gemini" in (profile.get("message") or "")


@respx.mock
def test_item14_devices_endpoint_friendly_error(data_dir, signed_in_tokens) -> None:
    r = devices_api_get(spotify_response=httpx.Response(500, json={"error": "boom"}))
    assert r.status_code == 502
    assert "Could not list Spotify devices" in r.json()["detail"]


def test_item15_delete_device_clears_saved_file(data_dir) -> None:
    settings = Settings()
    save_device(settings.resolved_device_path, DeviceSelection(device_id="dev-1"))
    client = TestClient(app)
    r = client.delete("/api/device")
    assert r.status_code == 200
    assert load_device(settings.resolved_device_path) is None


@respx.mock
def test_item18_friendly_guidance_for_nonsense(data_dir, signed_in_tokens) -> None:
    settings = Settings()
    streams = [FakeOllamaStream([json.dumps({"message": {"role": "assistant", "content": ""}, "done": True})])]

    def fake_stream(_client_self, _method, _url, **kwargs):
        return streams[0]

    with patch("httpx.Client.stream", fake_stream):
        events = list(iter_ollama_chat_events("flurble the gazorp", settings))
    final = next(e for e in events if e.get("type") == "final")
    assert final["text"] == FRIENDLY_SPOTIFY_GUIDANCE


@respx.mock
def test_item19_create_playlist_defaults_private(data_dir, signed_in_tokens) -> None:
    route = respx.post("https://api.spotify.com/v1/me/playlists").mock(
        return_value=httpx.Response(200, json={"id": "pppppppppppppppppppppp", "name": "x"})
    )
    runner = SpotifyToolRunner(settings=Settings())
    runner.run("spotify_create_playlist", {"name": "Secret list"})
    runner.close()
    body = json.loads(route.calls.last.request.content or b"{}")
    assert body.get("public") is False


def test_item20_remove_from_queue_explains_limitation(data_dir, signed_in_tokens) -> None:
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_remove_from_queue", {})
    runner.close()
    data = json.loads(raw)
    assert "cannot remove" in data["error"].lower()
    assert "spotify_skip_next" in data.get("try_instead", [])[0]


def test_item09_gemini_emits_tool_events(data_dir, signed_in_tokens) -> None:
    settings = Settings()

    def fake_gemini(user_text, settings, history=None, *, emit=None, conversation_id=None):
        if emit:
            emit({"type": "tool_start", "name": "spotify_me"})
            emit({"type": "tool_done", "name": "spotify_me", "preview": "{}"})
        return "ok"

    with patch("spot_backend.gemini_llm.run_chat_turn_gemini", fake_gemini):
        events = list(iter_gemini_chat_events("who am i", settings))
    types = [e["type"] for e in events]
    assert "tool_start" in types and "tool_done" in types and "final" in types


def test_item01_stream_message_accumulator_unit() -> None:
    msg: dict[str, Any] = {}
    _accumulate_ollama_stream_message(msg, {"content": "a", "role": "assistant"})
    _accumulate_ollama_stream_message(msg, {"content": "b"})
    assert msg["content"] == "ab"
