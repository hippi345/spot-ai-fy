"""Dev-mode Spotify API page limit (max 10) and artist name matching."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from spot_backend.artist_name_match import artist_names_match
from spot_backend.chat_shortcuts import try_deterministic_chat_reply
from spot_backend.config import Settings
from spot_backend.spotify_dev_limits import SPOTIFY_DEV_MAX_PAGE, assert_spotify_page_limit
from spot_backend.spotify_tools import SpotifyToolRunner


def test_clamp_spotify_page_limit_helper() -> None:
    from spot_backend.spotify_dev_limits import clamp_spotify_page_limit

    assert clamp_spotify_page_limit(50) == SPOTIFY_DEV_MAX_PAGE
    assert clamp_spotify_page_limit(3) == 3
    assert_spotify_page_limit(10)


def test_artist_name_match_variants() -> None:
    assert artist_names_match("the weeknd", "The Weeknd")
    assert artist_names_match("beyonce", "Beyoncé")
    assert not artist_names_match("Zxqvbrt Plonk", "ZXVREN")


@respx.mock
def test_user_playlists_tool_never_requests_limit_over_10(data_dir, signed_in_tokens) -> None:
    seen_limits: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        lim = int(request.url.params.get("limit", SPOTIFY_DEV_MAX_PAGE))
        seen_limits.append(lim)
        assert_spotify_page_limit(lim, context=str(request.url))
        return httpx.Response(
            200,
            json={"items": [], "total": 0, "limit": lim, "offset": 0},
        )

    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/playlists.*").mock(side_effect=handler)
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": "me"})
    )
    runner = SpotifyToolRunner(settings=Settings())
    runner.run("spotify_user_playlists", {"limit": 50, "offset": 0})
    runner.run("spotify_user_playlists", {})
    runner.close()
    assert seen_limits
    assert all(l <= SPOTIFY_DEV_MAX_PAGE for l in seen_limits)


@respx.mock
def test_vague_playlist_shortcut_paginates_with_limit_10(data_dir, signed_in_tokens) -> None:
    me_id = "user123456789012345678901"
    owned_id = "ownedpl000000000000001"
    page_limits: list[int] = []
    call_count = 0

    def playlists_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        lim = int(request.url.params.get("limit", 10))
        offset = int(request.url.params.get("offset", 0))
        page_limits.append(lim)
        assert lim <= SPOTIFY_DEV_MAX_PAGE
        call_count += 1
        if offset == 0:
            items = [
                {
                    "id": "37i9dQZF1EpyS73yJ89M1Q",
                    "name": "Discover Weekly",
                    "owner": {"id": "spotify"},
                },
            ]
        else:
            items = [{"id": owned_id, "name": "My Mix", "owner": {"id": me_id}}]
        return httpx.Response(
            200,
            json={"items": items, "total": 20, "limit": lim, "offset": offset},
        )

    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": me_id})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/playlists.*").mock(
        side_effect=playlists_handler
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(200, json={"is_playing": True})
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )

    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply("Play one of my playlists", runner)
    runner.close()
    assert outcome is not None
    assert "My Mix" in outcome.reply
    assert page_limits
    assert all(l <= SPOTIFY_DEV_MAX_PAGE for l in page_limits)
    assert call_count >= 2


@respx.mock
def test_play_songs_by_nonsense_artist_with_fuzzy_search_hit_plays_nothing(
    data_dir, signed_in_tokens
) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*type=artist.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "artists": {
                    "items": [
                        {"id": "zzzzzzzzzzzzzzzzzzzzzz", "name": "ZXVREN", "popularity": 40},
                    ]
                }
            },
        )
    )
    play_route = respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply("Play songs by Zxqvbrt Plonk", runner)
    runner.close()
    assert outcome is not None
    assert "couldn't find an artist called Zxqvbrt Plonk" in outcome.reply
    assert not play_route.called


@respx.mock
def test_spotify_tool_http_error_includes_redacted_body(data_dir, signed_in_tokens) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/playlists.*").mock(
        return_value=httpx.Response(
            400,
            json={"error": {"status": 400, "message": "Invalid limit"}},
        )
    )
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": "me"})
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_user_playlists", {"limit": 10})
    runner.close()
    data = json.loads(raw)
    assert "spotify_error_body_redacted" in data
    assert "Invalid limit" in data["spotify_error_body_redacted"]
