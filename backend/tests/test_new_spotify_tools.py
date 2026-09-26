from __future__ import annotations

import json

import httpx
import respx

from spot_backend.config import Settings
from spot_backend.spotify_client import DEFAULT_SCOPES  # noqa: F401 — asserted below
from spot_backend.spotify_tools import SpotifyToolRunner
from spot_backend.token_store import TokenBundle, save_tokens
import time


def _runner_with_scope(scope: str) -> SpotifyToolRunner:
    settings = Settings()
    save_tokens(
        settings.resolved_token_path,
        TokenBundle(
            access_token="tok",
            refresh_token="ref",
            expires_at=time.time() + 3600,
            scope=scope,
        ),
    )
    return SpotifyToolRunner(settings=settings)


@respx.mock
def test_recently_played_scope_gate(data_dir, signed_in_tokens) -> None:
    runner = _runner_with_scope("user-read-private")
    raw = runner.run("spotify_recently_played", {"limit": 5})
    data = json.loads(raw)
    assert "Reconnect Spotify to enable recently played history" in data["error"]
    assert "user-read-recently-played" in data["missing_scopes"]
    runner.close()


@respx.mock
def test_recently_played_ok(data_dir, signed_in_tokens) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player/recently-played.*").mock(
        return_value=httpx.Response(200, json={"items": []})
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_recently_played", {"limit": 5})
    assert json.loads(raw)["items"] == []
    runner.close()


@respx.mock
def test_save_tracks_missing_library_modify_scope(data_dir) -> None:
    runner = _runner_with_scope("user-read-private user-library-read")
    raw = runner.run("spotify_save_tracks", {"track_ids": ["1111111111111111111111"]})
    data = json.loads(raw)
    assert data["error"] == (
        "Reconnect Spotify to enable saving or removing tracks and albums in your library."
    )
    runner.close()


@respx.mock
def test_save_tracks(data_dir, signed_in_tokens) -> None:
    route = respx.put(url__regex=r"https://api\.spotify\.com/v1/me/tracks.*").mock(
        return_value=httpx.Response(200, json={})
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_save_tracks", {"track_ids": ["1111111111111111111111"]})
    assert json.loads(raw)["ok"] is True
    assert route.called
    runner.close()


@respx.mock
def test_get_queue(data_dir, signed_in_tokens) -> None:
    respx.get("https://api.spotify.com/v1/me/player/queue").mock(
        return_value=httpx.Response(200, json={"queue": []})
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_get_queue", {})
    assert "queue" in raw
    runner.close()


@respx.mock
def test_playlists_containing_track_truncated(data_dir, signed_in_tokens) -> None:
    tid = "2222222222222222222222"
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/playlists.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [{"id": "pl1", "name": "A", "owner": {"id": "u1"}}],
                "next": "http://next",
            },
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/playlists/pl1/items.*").mock(
        return_value=httpx.Response(
            200,
            json={"items": [{"item": {"id": tid, "uri": f"spotify:track:{tid}"}}], "next": None},
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run(
        "spotify_playlists_containing_track",
        {"track_id": tid, "max_playlists": 1},
    )
    data = json.loads(raw)
    assert data["truncated"] is True
    assert len(data["playlists"]) == 1
    runner.close()


def test_default_scopes_include_new(data_dir) -> None:
    assert "user-read-recently-played" in DEFAULT_SCOPES
    assert "user-library-modify" in DEFAULT_SCOPES
    assert "user-follow-modify" in DEFAULT_SCOPES
