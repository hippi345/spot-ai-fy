from __future__ import annotations

import json

import httpx
import respx

from spot_backend.config import Settings
from spot_backend.spotify_tools import SpotifyToolRunner


def _runner(data_dir, signed_in_tokens) -> SpotifyToolRunner:
    return SpotifyToolRunner(settings=Settings())


@respx.mock
def test_spotify_search_success(data_dir, signed_in_tokens) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [
                        {
                            "type": "track",
                            "id": "track123456789012345678",
                            "uri": "spotify:track:track123456789012345678",
                            "name": "Test Song",
                        }
                    ]
                }
            },
        )
    )
    runner = _runner(data_dir, signed_in_tokens)
    try:
        raw = runner.run("spotify_search", {"query": "test song", "types": "track"})
        data = json.loads(raw)
        assert data["tracks"]["items"][0]["name"] == "Test Song"
    finally:
        runner.close()


@respx.mock
def test_spotify_search_403_returns_structured_error(data_dir, signed_in_tokens) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search.*").mock(
        return_value=httpx.Response(
            403,
            json={"error": {"status": 403, "message": "Forbidden"}},
        )
    )
    runner = _runner(data_dir, signed_in_tokens)
    try:
        raw = runner.run("spotify_search", {"query": "x", "types": "track"})
        data = json.loads(raw)
        assert "Spotify HTTP 403" in data["error"]
        assert "hint" in data
        assert "reconnect_spotify_unnecessary" not in data
    finally:
        runner.close()


@respx.mock
def test_spotify_search_429_returns_structured_error(data_dir, signed_in_tokens) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search.*").mock(
        return_value=httpx.Response(429, json={"error": {"status": 429, "message": "API rate limit"}})
    )
    runner = _runner(data_dir, signed_in_tokens)
    try:
        raw = runner.run("spotify_search", {"query": "x", "types": "track"})
        data = json.loads(raw)
        assert "Spotify HTTP 429" in data["error"]
        assert data.get("reconnect_spotify_unnecessary") is True
    finally:
        runner.close()


@respx.mock
def test_add_tracks_by_query_happy_path(data_dir, signed_in_tokens) -> None:
    pid = "playlist12345678901234"
    track_id = "track123456789012345678"
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": "me-user-id"})
    )
    respx.get(url__regex=rf"https://api\.spotify\.com/v1/playlists/{pid}.*").mock(
        return_value=httpx.Response(
            200,
            json={"id": pid, "name": "My List", "owner": {"id": "me-user-id"}},
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [
                        {
                            "type": "track",
                            "id": track_id,
                            "uri": f"spotify:track:{track_id}",
                            "name": "Add Me",
                            "album": {"release_date": "2024-01-01"},
                        }
                    ]
                }
            },
        )
    )
    respx.get(url__regex=rf"https://api\.spotify\.com/v1/playlists/{pid}/items.*").mock(
        return_value=httpx.Response(200, json={"items": [], "next": None})
    )
    respx.post(url__regex=rf"https://api\.spotify\.com/v1/playlists/{pid}/items.*").mock(
        return_value=httpx.Response(200, json={"snapshot_id": "snap"})
    )

    runner = _runner(data_dir, signed_in_tokens)
    try:
        raw = runner.run(
            "spotify_add_tracks_by_query",
            {"playlist_id": pid, "query": "Add Me", "count": 1},
        )
        data = json.loads(raw)
        assert data.get("added_count", 0) >= 1 or data.get("added_tracks")
    finally:
        runner.close()


@respx.mock
def test_play_playlist_returns_playback_summary(data_dir, signed_in_tokens) -> None:
    pid = "playlist12345678901234"
    context_uri = f"spotify:playlist:{pid}"

    def player_state(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "is_playing": True,
                "context": {"uri": context_uri},
                "item": {"uri": "spotify:track:1111111111111111111111"},
            },
        )

    respx.put("https://api.spotify.com/v1/me/player/play").mock(return_value=httpx.Response(204))
    respx.get("https://api.spotify.com/v1/me/player").mock(side_effect=player_state)

    runner = _runner(data_dir, signed_in_tokens)
    try:
        raw = runner.run("spotify_play_playlist", {"playlist_id": pid})
        data = json.loads(raw)
        assert data["playlist_id"] == pid
        assert data.get("playback", {}).get("ok") is True
    finally:
        runner.close()


@respx.mock
def test_duplicate_playlist_owned_source(data_dir, signed_in_tokens) -> None:
    source = "srcplaylist00000000001"
    me_id = "meuser0000000000000001"
    new_id = "newpl000000000000000001"
    track_uri = "spotify:track:1111111111111111111111"
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": me_id})
    )
    respx.get(url__regex=rf"https://api\.spotify\.com/v1/playlists/{source}.*").mock(
        return_value=httpx.Response(
            200,
            json={"id": source, "name": "Source", "owner": {"id": me_id}},
        )
    )
    respx.get(url__regex=rf"https://api\.spotify\.com/v1/playlists/{source}/items.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [{"track": {"uri": track_uri}}],
                "next": None,
            },
        )
    )
    respx.post("https://api.spotify.com/v1/me/playlists").mock(
        return_value=httpx.Response(200, json={"id": new_id, "name": "Source (copy)"})
    )
    respx.post(url__regex=rf"https://api\.spotify\.com/v1/playlists/{new_id}/items.*").mock(
        return_value=httpx.Response(200, json={"snapshot_id": "snap"})
    )

    runner = _runner(data_dir, signed_in_tokens)
    try:
        raw = runner.run("spotify_duplicate_playlist", {"source_playlist_id": source})
        data = json.loads(raw)
        assert data.get("new_playlist_id") == new_id or data.get("playlist_id_for_add_tracks") == new_id
    finally:
        runner.close()
