"""Respx fixtures for eval mock runs (no real Spotify network)."""

from __future__ import annotations

import re
from typing import Any

import httpx
import respx

TRACK_A = "1111111111111111111111"
TRACK_B = "2222222222222222222222"
ARTIST_SZA = "3333333333333333333333"
PL_RNB = "4444444444444444444444"
PL_WORKOUT = "5555555555555555555555"
DEVICE_ID = "evaldevice0000000001"


def _player_json(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "is_playing": True,
        "device": {"id": DEVICE_ID, "name": "Eval Speaker", "type": "Speaker", "is_active": True},
        "item": {
            "type": "track",
            "id": TRACK_A,
            "uri": f"spotify:track:{TRACK_A}",
            "name": "Down",
            "artists": [{"id": "aaaa", "name": "Jay Sean"}],
            "album": {"id": "bbbb", "name": "All or Nothing"},
        },
    }
    base.update(overrides)
    return base


def allow_eval_llm_passthrough() -> None:
    """Allow real HTTP to cloud LLM APIs while respx mocks Spotify in live evals."""
    respx.route(url__regex=r"https://generativelanguage\.googleapis\.com/.*").pass_through()
    respx.route(url__regex=r"https://api\.openai\.com/.*").pass_through()
    respx.route(url__regex=r"https://api\.anthropic\.com/.*").pass_through()
    respx.route(url__regex=r"https://api\.x\.ai/.*").pass_through()
    respx.route(url__regex=r"http://127\.0\.0\.1:\d+/.*").pass_through()
    respx.route(url__regex=r"http://localhost:\d+/.*").pass_through()


def install_profile(profile: str = "default") -> None:
    """Register Spotify API mocks for the given profile name."""
    respx.reset()
    respx.post("https://accounts.spotify.com/api/token").mock(
        return_value=httpx.Response(
            200,
            json={
                "access_token": "eval-access-token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "refresh_token": "eval-refresh-token",
                "scope": "user-read-playback-state",
            },
        )
    )
    if profile == "no_device":
        _install_no_active_device()
        return
    if profile == "rate_limited":
        _install_rate_limited()
        return
    if profile == "empty_search":
        _install_empty_search()
        return
    if profile == "canonical_library":
        _install_canonical_library()
        return
    _install_default()


def _install_default() -> None:
    player = _player_json()
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player(?:\?.*)?$").mock(
        return_value=httpx.Response(200, json=player)
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player/devices").mock(
        return_value=httpx.Response(
            200,
            json={"devices": [{"id": DEVICE_ID, "name": "Eval Speaker", "type": "Speaker", "is_active": True}]},
        )
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play").mock(
        return_value=httpx.Response(204, text="")
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/pause").mock(
        return_value=httpx.Response(204, text="")
    )
    respx.post(url__regex=r"https://api\.spotify\.com/v1/me/player/next").mock(
        return_value=httpx.Response(204, text="")
    )
    respx.post(url__regex=r"https://api\.spotify\.com/v1/me/player/previous").mock(
        return_value=httpx.Response(204, text="")
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/volume").mock(
        return_value=httpx.Response(204, text="")
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/shuffle").mock(
        return_value=httpx.Response(204, text="")
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/repeat").mock(
        return_value=httpx.Response(204, text="")
    )
    respx.post(url__regex=r"https://api\.spotify\.com/v1/me/player/queue").mock(
        return_value=httpx.Response(204, text="")
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/playlists").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {"id": PL_RNB, "name": "RNB2025", "uri": f"spotify:playlist:{PL_RNB}"},
                    {"id": PL_WORKOUT, "name": "Workout", "uri": f"spotify:playlist:{PL_WORKOUT}"},
                ],
                "total": 2,
            },
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/playlists/[^/]+$").mock(
        return_value=httpx.Response(
            200,
            json={"id": PL_RNB, "name": "RNB2025", "snapshot_id": "snap1", "public": True},
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/playlists/.*/items").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "track": {
                            "id": TRACK_B,
                            "uri": f"spotify:track:{TRACK_B}",
                            "name": "Kill Bill",
                            "artists": [{"name": "SZA"}],
                        }
                    }
                ]
            },
        )
    )
    respx.post(url__regex=r"https://api\.spotify\.com/v1/playlists/.*/items").mock(
        return_value=httpx.Response(201, json={"snapshot_id": "snap2"})
    )
    respx.post(url__regex=r"https://api\.spotify\.com/v1/me/playlists").mock(
        return_value=httpx.Response(201, json={"id": "newpl" + "x" * 17, "name": "Chill Mix"})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/playlists/[^/]+$").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(side_effect=_search_router)
    respx.get(url__regex=r"https://api\.spotify\.com/v1/tracks/[^/]+$").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": TRACK_A,
                "uri": f"spotify:track:{TRACK_A}",
                "name": "Down",
                "artists": [{"name": "Jay Sean"}],
            },
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me$").mock(
        return_value=httpx.Response(200, json={"id": "meuser", "display_name": "Eval User"})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/artists/[^/]+$").mock(
        return_value=httpx.Response(
            200,
            json={"id": ARTIST_SZA, "name": "SZA", "uri": f"spotify:artist:{ARTIST_SZA}"},
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/artists/[^/]+/top-tracks.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "tracks": [
                    {
                        "id": TRACK_B,
                        "uri": f"spotify:track:{TRACK_B}",
                        "name": "Kill Bill",
                        "artists": [{"name": "SZA"}],
                    }
                ]
            },
        )
    )
    _install_library_sync_mocks()


def _install_canonical_library() -> None:
    """Large saved-tracks library for vector-search live evals."""
    from evals.eval_library_fixture import spotify_saved_tracks_page, spotify_top_tracks

    _install_default()

    def _tracks_handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params.get("offset") or 0)
        limit = int(request.url.params.get("limit") or 50)
        return httpx.Response(200, json=spotify_saved_tracks_page(offset, limit))

    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/tracks.*").mock(side_effect=_tracks_handler)
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/top/tracks.*").mock(
        return_value=httpx.Response(200, json=spotify_top_tracks(50))
    )


def _install_library_sync_mocks() -> None:
    """Saved tracks/playlists/following for search_my_library and library tools in live evals."""
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/tracks.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "track": {
                            "id": TRACK_B,
                            "uri": f"spotify:track:{TRACK_B}",
                            "name": "Pump It",
                            "artists": [{"name": "Artist"}],
                            "album": {"name": "Workout", "release_date": "2020"},
                        }
                    },
                    {
                        "track": {
                            "id": "3333333333333333333333",
                            "uri": "spotify:track:3333333333333333333333",
                            "name": "Soft Rain",
                            "artists": [{"name": "Singer"}],
                            "album": {"name": "Gentle", "release_date": "2019"},
                        }
                    },
                ],
                "total": 2,
            },
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/playlists.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": PL_RNB,
                        "name": "RNB2025",
                        "uri": f"spotify:playlist:{PL_RNB}",
                    }
                ],
                "total": 1,
            },
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/following.*").mock(
        return_value=httpx.Response(200, json={"artists": {"items": [], "total": 0}})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/playlists/[^/]+/tracks.*").mock(
        return_value=httpx.Response(200, json={"items": [], "total": 0})
    )


def _search_router(request: httpx.Request) -> httpx.Response:
    q = (request.url.params.get("q") or "").lower()
    stype = request.url.params.get("type") or ""
    if "artist" in stype and "sza" in q:
        return httpx.Response(
            200,
            json={"artists": {"items": [{"id": ARTIST_SZA, "name": "SZA", "uri": f"spotify:artist:{ARTIST_SZA}"}]}},
        )
    if "artist" in stype and "unknownzz" in q:
        return httpx.Response(200, json={"artists": {"items": []}})
    if "track" in stype and "down" in q:
        return httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [
                        {
                            "id": TRACK_A,
                            "uri": f"spotify:track:{TRACK_A}",
                            "name": "Down",
                            "artists": [{"name": "Jay Sean"}],
                        }
                    ]
                }
            },
        )
    if "playlist" in stype:
        return httpx.Response(
            200,
            json={"playlists": {"items": [{"id": PL_RNB, "name": "RNB2025"}]}},
        )
    return httpx.Response(200, json={"tracks": {"items": []}, "artists": {"items": []}})


def _install_no_active_device() -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player").mock(
        return_value=httpx.Response(204, text="")
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play").mock(
        return_value=httpx.Response(404, json={"error": {"message": "No active device", "status": 404}})
    )


def _install_rate_limited() -> None:
    _install_default()
    respx.get(url__regex=re.compile(r"https://api\.spotify\.com/v1/search\?.*")).mock(
        return_value=httpx.Response(
            429,
            json={"error": {"message": "API rate limit exceeded", "status": 429}},
            headers={"Retry-After": "30"},
        )
    )


def _install_empty_search() -> None:
    _install_default()
    respx.get(url__regex=re.compile(r"https://api\.spotify\.com/v1/search\?.*")).mock(
        return_value=httpx.Response(200, json={"tracks": {"items": []}})
    )
