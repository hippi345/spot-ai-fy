"""spotify_play_bare resolution and failure_reason for mononym artists (Ye)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from spot_backend.artist_name_match import resolve_artist_from_search_items
from spot_backend.chat_shortcuts import try_deterministic_chat_reply
from spot_backend.config import Settings
from spot_backend.reply_tool_trace import append_tool_trace_record, tool_trace_log_path
from spot_backend.spotify_tools import SpotifyToolRunner

YE_ARTIST_ID = "3TVXtAsR1Inumwj472S9r4"
KANYE_TRACK_URI = "spotify:track:runaway000000000001"
ALBUM_ID = "alb0000000000000000002"


def _mock_ye_playback_success() -> None:
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "item": {
                    "name": "Runaway",
                    "uri": KANYE_TRACK_URI,
                    "artists": [{"id": YE_ARTIST_ID, "name": "Kanye West"}],
                },
            },
        )
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )
    respx.get(f"https://api.spotify.com/v1/albums/{ALBUM_ID}/tracks").mock(
        return_value=httpx.Response(
            200,
            json={"items": [{"uri": KANYE_TRACK_URI}]},
        )
    )


def _track_search_noise() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "tracks": {
                "items": [
                    {
                        "name": "YES IT IS",
                        "popularity": 60,
                        "artists": [{"name": "Leon Thomas"}],
                        "uri": "spotify:track:yesitis00000000000001",
                        "album": {"id": "alb0000000000000000001"},
                    }
                ]
            }
        },
    )


def _artist_tracks_for_ye(request: httpx.Request) -> httpx.Response:
    q = request.url.params.get("q") or ""
    if YE_ARTIST_ID in q or 'artist:"Kanye West"' in q or 'artist:"Ye"' in q:
        return httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [
                        {
                            "name": "Runaway",
                            "popularity": 80,
                            "uri": KANYE_TRACK_URI,
                            "album": {"id": ALBUM_ID},
                            "artists": [{"id": YE_ARTIST_ID, "name": "Kanye West"}],
                        }
                    ]
                }
            },
        )
    return _track_search_noise()


def test_resolve_artist_from_search_items_ye_exact_and_alias() -> None:
    items = [
        {"id": YE_ARTIST_ID, "name": "Ye", "popularity": 85},
        {"id": "other0000000000000000001", "name": "Ye-Ye", "popularity": 10},
    ]
    row = resolve_artist_from_search_items("Ye", items)
    assert row is not None
    assert row[0] == YE_ARTIST_ID
    assert row[3] == "exact_name"

    kanye_only = [{"id": YE_ARTIST_ID, "name": "Kanye West", "popularity": 90}]
    row2 = resolve_artist_from_search_items("Ye", kanye_only)
    assert row2 is not None
    assert row2[0] == YE_ARTIST_ID
    assert row2[3] == "alias"


@respx.mock
def test_play_bare_ye_artist_named_ye(data_dir, signed_in_tokens) -> None:
    def search_handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("type") == "artist":
            return httpx.Response(
                200,
                json={
                    "artists": {
                        "items": [{"id": YE_ARTIST_ID, "name": "Ye", "popularity": 85}],
                    }
                },
            )
        return _track_search_noise()

    def combined_search(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("type") == "artist":
            return search_handler(request)
        return _artist_tracks_for_ye(request)

    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        side_effect=combined_search
    )
    respx.get(f"https://api.spotify.com/v1/artists/{YE_ARTIST_ID}").mock(
        return_value=httpx.Response(200, json={"id": YE_ARTIST_ID, "name": "Ye"})
    )
    _mock_ye_playback_success()

    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_play_bare", {"query": "Ye"})
    runner.close()
    data = json.loads(raw)
    assert data.get("bare_play_mode") == "artist"
    assert data.get("ok") is True
    assert data.get("failure_reason") is None
    inner = json.loads(data["playback_result"])
    assert inner.get("artist_id") == YE_ARTIST_ID


@respx.mock
def test_play_bare_ye_resolves_kanye_west_display_name(data_dir, signed_in_tokens) -> None:
    def search_handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("type") == "artist":
            return httpx.Response(
                200,
                json={
                    "artists": {
                        "items": [{"id": YE_ARTIST_ID, "name": "Kanye West", "popularity": 90}],
                    }
                },
            )
        return _track_search_noise()

    def combined_search(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("type") == "artist":
            return search_handler(request)
        return _artist_tracks_for_ye(request)

    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        side_effect=combined_search
    )
    respx.get(f"https://api.spotify.com/v1/artists/{YE_ARTIST_ID}").mock(
        return_value=httpx.Response(200, json={"id": YE_ARTIST_ID, "name": "Kanye West"})
    )
    _mock_ye_playback_success()

    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply("Play Ye", runner)
    runner.close()
    assert outcome is not None
    assert outcome.tool_names()[0] == "spotify_play_bare"
    assert "could not start playback" not in outcome.reply.lower()
    raw = outcome.tool_steps[-1][2]
    data = json.loads(raw)
    assert data.get("failure_reason") is None
    inner = json.loads(data["playback_result"])
    assert inner.get("artist_id") == YE_ARTIST_ID


@respx.mock
def test_play_bare_ye_short_query_top_when_only_distant_names(data_dir, signed_in_tokens) -> None:
    def search_handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("type") == "artist":
            return httpx.Response(
                200,
                json={
                    "artists": {
                        "items": [
                            {"id": "zzzzzzzzzzzzzzzzzzzz1", "name": "Yello", "popularity": 40},
                            {"id": YE_ARTIST_ID, "name": "Kanye West", "popularity": 88},
                        ],
                    }
                },
            )
        return _track_search_noise()

    def combined_search(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("type") == "artist":
            return search_handler(request)
        return _artist_tracks_for_ye(request)

    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        side_effect=combined_search
    )
    respx.get(f"https://api.spotify.com/v1/artists/{YE_ARTIST_ID}").mock(
        return_value=httpx.Response(200, json={"id": YE_ARTIST_ID, "name": "Kanye West"})
    )
    _mock_ye_playback_success()

    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_play_bare", {"query": "Ye"})
    runner.close()
    data = json.loads(raw)
    assert data.get("bare_play_mode") == "artist"
    inner = json.loads(data["playback_result"])
    assert inner.get("artist_id") == YE_ARTIST_ID


@respx.mock
def test_play_bare_failure_reason_artist_not_found(data_dir, signed_in_tokens) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*type=artist.*").mock(
        return_value=httpx.Response(200, json={"artists": {"items": []}})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*type=track.*").mock(
        return_value=httpx.Response(200, json={"tracks": {"items": []}})
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_play_bare", {"query": "Ye"})
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is False
    assert data.get("failure_reason") == "artist_not_found"


@respx.mock
def test_play_bare_ye_track_search_uses_kanye_west_not_query_mononym(
    data_dir, signed_in_tokens
) -> None:
    """Resolved alias name must drive artist:\"...\" track search, not the bare query Ye."""
    track_search_qs: list[str] = []

    def search_handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("type") == "artist":
            return httpx.Response(
                200,
                json={
                    "artists": {
                        "items": [{"id": YE_ARTIST_ID, "name": "Kanye West", "popularity": 90}],
                    }
                },
            )
        q = request.url.params.get("q") or ""
        track_search_qs.append(q)
        if 'artist:"Kanye West"' in q:
            return httpx.Response(
                200,
                json={
                    "tracks": {
                        "items": [
                            {
                                "name": "Runaway",
                                "popularity": 80,
                                "uri": KANYE_TRACK_URI,
                                "album": {"id": ALBUM_ID},
                                "artists": [{"id": YE_ARTIST_ID, "name": "Kanye West"}],
                            }
                        ]
                    }
                },
            )
        return httpx.Response(200, json={"tracks": {"items": []}})

    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        side_effect=search_handler
    )
    respx.get(f"https://api.spotify.com/v1/artists/{YE_ARTIST_ID}").mock(
        return_value=httpx.Response(200, json={"id": YE_ARTIST_ID, "name": "Ye"})
    )
    _mock_ye_playback_success()

    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_play_bare", {"query": "Ye"})
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is True
    assert track_search_qs
    assert any('artist:"Kanye West"' in q for q in track_search_qs)
    assert not any('artist:"Ye"' in q for q in track_search_qs)


@respx.mock
def test_play_bare_ye_falls_back_to_artist_context_when_no_tracks(
    data_dir, signed_in_tokens
) -> None:
    def search_handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("type") == "artist":
            return httpx.Response(
                200,
                json={
                    "artists": {
                        "items": [{"id": YE_ARTIST_ID, "name": "Kanye West", "popularity": 90}],
                    }
                },
            )
        return httpx.Response(200, json={"tracks": {"items": []}})

    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        side_effect=search_handler
    )
    respx.get(f"https://api.spotify.com/v1/artists/{YE_ARTIST_ID}").mock(
        return_value=httpx.Response(200, json={"id": YE_ARTIST_ID, "name": "Kanye West"})
    )
    play_bodies: list[dict] = []

    def play_handler(request: httpx.Request) -> httpx.Response:
        play_bodies.append(json.loads(request.content.decode()))
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        side_effect=play_handler
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "context": {"uri": f"spotify:artist:{YE_ARTIST_ID}"},
                "item": {
                    "name": "Runaway",
                    "uri": KANYE_TRACK_URI,
                    "artists": [{"id": YE_ARTIST_ID, "name": "Kanye West"}],
                },
            },
        )
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )

    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_play_bare", {"query": "Ye"})
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is True
    assert play_bodies
    assert play_bodies[-1].get("context_uri") == f"spotify:artist:{YE_ARTIST_ID}"
    inner = json.loads(data["playback_result"])
    assert inner.get("playback_verified") is True


@respx.mock
def test_play_bare_failure_reason_no_tracks_for_artist(data_dir, signed_in_tokens) -> None:
    def search_handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("type") == "artist":
            return httpx.Response(
                200,
                json={
                    "artists": {
                        "items": [{"id": YE_ARTIST_ID, "name": "Ye", "popularity": 85}],
                    }
                },
            )
        return httpx.Response(200, json={"tracks": {"items": []}})

    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        side_effect=search_handler
    )
    respx.get(f"https://api.spotify.com/v1/artists/{YE_ARTIST_ID}").mock(
        return_value=httpx.Response(200, json={"id": YE_ARTIST_ID, "name": "Ye"})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(404, json={"error": {"status": 404, "message": "Not found"}})
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_play_bare", {"query": "Ye"})
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is False
    assert data.get("failure_reason") == "no_tracks_for_artist"


@respx.mock
def test_play_bare_trace_row_has_failure_reason_not_unknown(
    data_dir, signed_in_tokens
) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(200, json={"artists": {"items": []}, "tracks": {"items": []}})
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_play_bare", {"query": "Ye"})
    runner.close()
    append_tool_trace_record(
        data_dir,
        conversation_id="bare-ye-fail",
        tool_name="spotify_play_bare",
        args_summary='{"query":"Ye"}',
        outcome="error",
        raw_result=raw,
    )
    row = json.loads(tool_trace_log_path(data_dir).read_text(encoding="utf-8").strip().splitlines()[-1])
    assert row.get("failure_reason") == "artist_not_found"
    assert row.get("failure_reason") != "unknown_error"
