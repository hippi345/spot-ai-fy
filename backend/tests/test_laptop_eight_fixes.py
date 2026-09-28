"""Regression tests for laptop agent issues 1–8 (mocked Spotify + chat path)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from spot_backend.capability_replies import try_capability_question_reply
from spot_backend.chat_messages import collapse_duplicate_reply_text, prepare_user_visible_reply
from spot_backend.chat_shortcuts import try_deterministic_chat_reply
from spot_backend.config import Settings
from spot_backend.deterministic_chat import resolve_deterministic_chat_outcome
from spot_backend.gemini_llm import run_chat_turn_gemini
from spot_backend.playback_reply import format_skip_reply
from spot_backend.reply_tool_trace import append_tool_trace_record, tool_trace_log_path
from spot_backend.app import app
from spot_backend.spotify_tools import SpotifyToolRunner
from tests.recheck_helpers import make_gemini_post_recorder
from tests.test_laptop_multiturn_shortcuts import _collect_sse_events
from tests.test_play_track_selection import BLINDING_URI, _blinding_search_json
from fastapi.testclient import TestClient
from unittest.mock import patch

YE_ARTIST_ID = "3TVXtAsR1Inumwj472S9r4"
YE_TRACK_ID = "yesitis00000000000001"


@respx.mock
def test_issue1_podcast_capability_never_empty_gemini_fallback(data_dir, signed_in_tokens) -> None:
    reply = try_capability_question_reply("Can you play podcasts via this interface?")
    assert reply
    assert "podcast" in reply.lower() or "no" in reply.lower()
    outcome = resolve_deterministic_chat_outcome(
        "Can you play podcasts via this interface?",
        SpotifyToolRunner(settings=Settings()),
        conversation_id="cap-pod",
    )
    assert outcome is not None
    assert outcome.tool_names() == []


def test_issue1_gemini_empty_turn_retries_once(data_dir, signed_in_tokens) -> None:
    settings = Settings(gemini_api_key="test-key-empty-once")
    responses = [
        {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": []},
                }
            ]
        },
        {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": "Hello."}]},
                }
            ]
        },
    ]
    call_idx = {"i": 0}

    def handler(_body: dict, _n: int, req: httpx.Request) -> httpx.Response:
        body = responses[min(call_idx["i"], len(responses) - 1)]
        call_idx["i"] += 1
        return httpx.Response(200, json=body, request=req)

    _, fake_post = make_gemini_post_recorder(handler)
    with patch("httpx.Client.post", fake_post), patch(
        "spot_backend.gemini_llm.gemini_deterministic_shortcut_reply",
        return_value=None,
    ):
        text = run_chat_turn_gemini("hello", settings)
    assert text == "Hello."
    assert call_idx["i"] == 2


@respx.mock
def test_issue2_skips_empty_owned_playlist_and_retries(data_dir, signed_in_tokens) -> None:
    me_id = "user123456789012345678901"
    empty_id = "0b6XBbGmm43kPseTVuSlnR"
    good_id = "ownedpl000000000000001"
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": me_id, "display_name": "Me"})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/playlists.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": empty_id,
                        "name": "My new playlist",
                        "owner": {"id": me_id},
                        "tracks": {"total": 0},
                    },
                    {
                        "id": good_id,
                        "name": "My Mix",
                        "owner": {"id": me_id},
                        "tracks": {"total": 12},
                    },
                ]
            },
        )
    )

    def play_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode() or "{}")
        ctx = body.get("context_uri") or ""
        if empty_id in ctx:
            return httpx.Response(400, json={"error": {"message": "Playlist empty"}})
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        side_effect=play_handler
    )
    respx.get(f"https://api.spotify.com/v1/playlists/{good_id}").mock(
        return_value=httpx.Response(200, json={"id": good_id, "owner": {"id": me_id}})
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "context": {"uri": f"spotify:playlist:{good_id}"},
                "item": {"name": "Track A", "artists": [{"name": "Artist"}]},
            },
        )
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )

    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply("Play one of my playlists", runner)
    runner.close()
    assert outcome is not None
    assert empty_id not in outcome.reply
    assert "My Mix" in outcome.reply
    assert "playlist" in outcome.reply.lower()


@respx.mock
def test_issue2_skips_playlist_without_tracks_field_when_probe_empty(
    data_dir, signed_in_tokens
) -> None:
    me_id = "user123456789012345678901"
    empty_id = "0b6XBbGmm43kPseTVuSlnR"
    good_id = "ownedpl000000000000002"
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": me_id})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/playlists.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": empty_id,
                        "name": "My new playlist",
                        "owner": {"id": me_id},
                    },
                    {
                        "id": good_id,
                        "name": "Jamz · My Artists",
                        "owner": {"id": me_id},
                        "items": {"total": 4},
                    },
                ]
            },
        )
    )
    respx.get(url__regex=rf"https://api\.spotify\.com/v1/playlists/{empty_id}/items.*").mock(
        return_value=httpx.Response(200, json={"total": 0, "items": []})
    )
    respx.get(f"https://api.spotify.com/v1/playlists/{good_id}").mock(
        return_value=httpx.Response(200, json={"id": good_id, "owner": {"id": me_id}})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "context": {"uri": f"spotify:playlist:{good_id}"},
                "item": {"name": "Song", "artists": [{"name": "A"}]},
            },
        )
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )
    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply("Play one of my playlists", runner)
    runner.close()
    assert outcome is not None
    assert empty_id not in str(outcome.tool_names())
    assert "Jamz" in outcome.reply


@respx.mock
def test_issue2_skips_items_total_zero_without_play_attempt(data_dir, signed_in_tokens) -> None:
    me_id = "user123456789012345678901"
    empty_id = "emptypl00000000000001"
    good_id = "ownedpl000000000000003"
    play_calls: list[str] = []

    def play_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode() or "{}")
        play_calls.append(body.get("context_uri") or "")
        return httpx.Response(204)

    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": me_id})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/playlists.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": empty_id,
                        "name": "Empty via items",
                        "owner": {"id": me_id},
                        "items": {"total": 0},
                    },
                    {
                        "id": good_id,
                        "name": "Has songs",
                        "owner": {"id": me_id},
                        "tracks": {"total": 3},
                    },
                ]
            },
        )
    )
    respx.get(f"https://api.spotify.com/v1/playlists/{good_id}").mock(
        return_value=httpx.Response(200, json={"id": good_id, "owner": {"id": me_id}})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        side_effect=play_handler
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "context": {"uri": f"spotify:playlist:{good_id}"},
                "item": {"name": "Song", "artists": [{"name": "A"}]},
            },
        )
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )
    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply("Play one of my playlists", runner)
    runner.close()
    assert outcome is not None
    assert not any(empty_id in (c or "") for c in play_calls)
    assert any(good_id in (c or "") for c in play_calls)


@respx.mock
def test_issue2_alternate_playlist_excludes_current_context(data_dir, signed_in_tokens) -> None:
    me_id = "user123456789012345678901"
    current_id = "currentpl00000000000001"
    other_id = "otherpl000000000000001"
    play_calls: list[str] = []
    player_state = {
        "is_playing": True,
        "context": {"uri": f"spotify:playlist:{current_id}"},
        "item": {"name": "On Current", "artists": [{"name": "A"}]},
    }

    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": me_id})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/playlists.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {"id": current_id, "name": "Current", "owner": {"id": me_id}, "tracks": {"total": 5}},
                    {"id": other_id, "name": "Other Mix", "owner": {"id": me_id}, "tracks": {"total": 8}},
                ]
            },
        )
    )
    respx.get(f"https://api.spotify.com/v1/playlists/{other_id}").mock(
        return_value=httpx.Response(200, json={"id": other_id, "owner": {"id": me_id}})
    )

    def player_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=player_state)

    respx.get("https://api.spotify.com/v1/me/player").mock(side_effect=player_handler)

    def play_handler_update(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode() or "{}")
        play_calls.append(body.get("context_uri") or "")
        player_state["context"] = {"uri": f"spotify:playlist:{other_id}"}
        player_state["item"] = {"name": "On Other", "artists": [{"name": "B"}]}
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        side_effect=play_handler_update
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )
    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply("Actually, play a different playlist", runner)
    runner.close()
    assert outcome is not None
    assert not any(current_id in (c or "") for c in play_calls)
    assert "Other Mix" in outcome.reply


@respx.mock
def test_issue3_play_something_chill_plays_without_asking(data_dir, signed_in_tokens) -> None:
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": "me123456789012345678901"})
    )
    respx.get("https://api.spotify.com/v1/playlists/chillpl000000000000001").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chillpl000000000000001",
                "owner": {"id": "me123456789012345678901"},
            },
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*type=playlist.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "playlists": {
                    "items": [
                        {
                            "id": "chillpl000000000000001",
                            "name": "Chill Horsies",
                        }
                    ]
                }
            },
        )
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "context": {"uri": "spotify:playlist:chillpl000000000000001"},
                "item": {"name": "Horse Song", "artists": [{"name": "Chill Band"}]},
            },
        )
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )

    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply("Play something chill", runner)
    runner.close()
    assert outcome is not None
    assert "?" not in outcome.reply.split("—")[0]
    assert "Playing" in outcome.reply
    assert "different" in outcome.reply.lower()


@respx.mock
def test_issue4_play_ye_prefers_artist(data_dir, signed_in_tokens) -> None:
    def search_handler(request: httpx.Request) -> httpx.Response:
        qtype = request.url.params.get("type")
        if qtype == "artist":
            return httpx.Response(
                200,
                json={
                    "artists": {
                        "items": [
                            {"id": YE_ARTIST_ID, "name": "Ye", "popularity": 85},
                        ]
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [
                        {
                            "name": "YES IT IS",
                            "popularity": 60,
                            "artists": [{"name": "Leon Thomas"}],
                            "uri": f"spotify:track:{YE_TRACK_ID}",
                            "album": {"id": "alb0000000000000000001"},
                        }
                    ]
                }
            },
        )

    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        side_effect=search_handler
    )
    respx.get(f"https://api.spotify.com/v1/artists/{YE_ARTIST_ID}").mock(
        return_value=httpx.Response(200, json={"id": YE_ARTIST_ID, "name": "Ye"})
    )
    respx.get(f"https://api.spotify.com/v1/artists/{YE_ARTIST_ID}/albums").mock(
        return_value=httpx.Response(200, json={"items": []})
    )
    respx.get(url__regex=rf"https://api\.spotify\.com/v1/artists/{YE_ARTIST_ID}/top-tracks.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "tracks": [
                    {
                        "name": "Runaway",
                        "uri": "spotify:track:runaway000000000001",
                        "album": {"id": "alb0000000000000000002"},
                        "popularity": 80,
                    }
                ]
            },
        )
    )
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
                    "uri": "spotify:track:runaway000000000001",
                    "artists": [{"name": "Ye"}],
                },
            },
        )
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )

    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply("Play Ye", runner)
    runner.close()
    assert outcome is not None
    assert outcome.tool_names()[0] == "spotify_play_bare"
    assert "YES IT IS" not in outcome.reply
    assert "Runaway" in outcome.reply or "Ye" in outcome.reply


@respx.mock
def test_issue4_play_ye_verifies_kanye_west_credits(data_dir, signed_in_tokens) -> None:
    def search_handler(request: httpx.Request) -> httpx.Response:
        qtype = request.url.params.get("type")
        q = request.url.params.get("q") or ""
        if qtype == "artist":
            return httpx.Response(
                200,
                json={
                    "artists": {
                        "items": [
                            {"id": YE_ARTIST_ID, "name": "Ye", "popularity": 85},
                        ]
                    }
                },
            )
        if qtype == "track" and YE_ARTIST_ID in q:
            return httpx.Response(
                200,
                json={
                    "tracks": {
                        "items": [
                            {
                                "name": "Runaway",
                                "popularity": 80,
                                "uri": "spotify:track:runaway000000000001",
                                "album": {"id": "alb0000000000000000002"},
                                "artists": [
                                    {"id": YE_ARTIST_ID, "name": "Kanye West"},
                                ],
                            }
                        ]
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [
                        {
                            "name": "YES IT IS",
                            "popularity": 60,
                            "artists": [{"name": "Leon Thomas"}],
                            "uri": f"spotify:track:{YE_TRACK_ID}",
                            "album": {"id": "alb0000000000000000001"},
                        }
                    ]
                }
            },
        )

    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        side_effect=search_handler
    )
    respx.get(f"https://api.spotify.com/v1/artists/{YE_ARTIST_ID}").mock(
        return_value=httpx.Response(200, json={"id": YE_ARTIST_ID, "name": "Ye"})
    )
    respx.get(f"https://api.spotify.com/v1/artists/{YE_ARTIST_ID}/albums").mock(
        return_value=httpx.Response(200, json={"items": []})
    )
    respx.get(f"https://api.spotify.com/v1/albums/alb0000000000000000002/tracks").mock(
        return_value=httpx.Response(
            200,
            json={"items": [{"uri": "spotify:track:runaway000000000001"}]},
        )
    )
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
                    "uri": "spotify:track:runaway000000000001",
                    "artists": [
                        {"id": YE_ARTIST_ID, "name": "Kanye West"},
                    ],
                },
            },
        )
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )

    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply("Play Ye", runner)
    runner.close()
    assert outcome is not None
    assert outcome.tool_names()[0] == "spotify_play_bare"
    assert "could not start playback" not in outcome.reply.lower()
    assert "Runaway" in outcome.reply


@respx.mock
def test_issue4_play_blinding_lights_prefers_track(data_dir, signed_in_tokens) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(200, json=_blinding_search_json())
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "item": {
                    "name": "Blinding Lights",
                    "uri": BLINDING_URI,
                    "artists": [{"name": "The Weeknd"}],
                },
            },
        )
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )
    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply("Play Blinding Lights", runner)
    runner.close()
    assert outcome is not None
    assert "Blinding Lights" in outcome.reply


@respx.mock
def test_issue4_play_drake_prefers_artist(data_dir, signed_in_tokens) -> None:
    drake_id = "3TVXtAsR1Inumwj472S9r5"
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*type=artist.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "artists": {
                    "items": [{"id": drake_id, "name": "Drake", "popularity": 95}],
                }
            },
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*type=track.*").mock(
        return_value=httpx.Response(200, json={"tracks": {"items": []}})
    )
    respx.get(f"https://api.spotify.com/v1/artists/{drake_id}").mock(
        return_value=httpx.Response(200, json={"id": drake_id, "name": "Drake"})
    )
    respx.get(url__regex=rf"https://api\.spotify\.com/v1/artists/{drake_id}/top-tracks.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "tracks": [
                    {
                        "name": "One Dance",
                        "uri": "spotify:track:onedance00000000001",
                        "album": {"id": "albdrake00000000000001"},
                        "popularity": 90,
                    }
                ]
            },
        )
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "item": {
                    "name": "One Dance",
                    "artists": [{"name": "Drake"}],
                },
            },
        )
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )
    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply("Play Drake", runner)
    runner.close()
    assert outcome is not None
    assert "Drake" in outcome.reply or "One Dance" in outcome.reply


@respx.mock
def test_issue5_play_by_artist_reply_uses_player_not_search_guess(data_dir, signed_in_tokens) -> None:
    weeknd_id = "weeknd0000000000000001"
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*type=artist.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "artists": {
                    "items": [
                        {"id": weeknd_id, "name": "The Weeknd", "popularity": 95},
                    ]
                }
            },
        )
    )
    respx.get(f"https://api.spotify.com/v1/artists/{weeknd_id}").mock(
        return_value=httpx.Response(200, json={"id": weeknd_id, "name": "The Weeknd"})
    )
    respx.get(url__regex=rf"https://api\.spotify\.com/v1/artists/{weeknd_id}/top-tracks.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "tracks": [
                    {
                        "name": "Rockin'",
                        "id": "rockin0000000000000001",
                        "uri": "spotify:track:rockin0000000000000001",
                        "album": {"id": "alb0000000000000000001"},
                        "popularity": 80,
                    }
                ]
            },
        )
    )
    def track_search_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [
                        {
                            "name": "Rockin'",
                            "id": "rockin0000000000000001",
                            "uri": "spotify:track:rockin0000000000000001",
                            "album": {"id": "alb0000000000000000001"},
                            "popularity": 80,
                            "artists": [{"id": weeknd_id, "name": "The Weeknd"}],
                        }
                    ]
                }
            },
        )

    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*type=track.*").mock(
        side_effect=track_search_handler
    )
    respx.get("https://api.spotify.com/v1/tracks/rockin0000000000000001").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "rockin0000000000000001",
                "uri": "spotify:track:rockin0000000000000001",
                "name": "Rockin'",
                "album": {"id": "alb0000000000000000001"},
            },
        )
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "item": {
                    "name": "Or Nah",
                    "uri": "spotify:track:ornah00000000000000001",
                    "artists": [{"name": "The Weeknd"}],
                },
            },
        )
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )
    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply("Play something by The Weeknd", runner)
    runner.close()
    assert outcome is not None
    assert "Or Nah" in outcome.reply
    assert "Rockin'" not in outcome.reply


def test_issue6_collapse_duplicate_sentence_in_reply() -> None:
    dup = "Going back to Loose Ends by Ella Mai. Going back to Loose Ends by Ella Mai."
    assert collapse_duplicate_reply_text(dup) == "Going back to Loose Ends by Ella Mai."
    assert (
        prepare_user_visible_reply(dup)
        == "Going back to Loose Ends by Ella Mai."
    )


def test_issue6_go_back_one_shortcut_single_line() -> None:
    player = {
        "item": {
            "name": "Loose Ends",
            "artists": [{"name": "Ella Mai"}],
        }
    }
    reply = format_skip_reply(player, skipped=True, direction="previous")
    assert reply == "Going back to Loose Ends by Ella Mai."


@respx.mock
def test_issue7_what_playlists_includes_total(data_dir, signed_in_tokens) -> None:
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": "user123456789012345678901"})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/playlists.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "total": 87,
                "items": [
                    {
                        "id": "pl1",
                        "name": "Workout",
                        "owner": {"id": "user123456789012345678901"},
                    },
                    {
                        "id": "pl2",
                        "name": "Chill",
                        "owner": {"id": "user123456789012345678901"},
                    },
                ],
            },
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply("What playlists do I have?", runner)
    runner.close()
    assert outcome is not None
    assert "87" in outcome.reply
    assert "including" in outcome.reply.lower()


@respx.mock
def test_issue8_trace_includes_redacted_spotify_error_body(data_dir, signed_in_tokens) -> None:
    token = "test-access-token"
    pid = "6zCID88oNjNv9zx6puDHKj"
    respx.get(f"https://api.spotify.com/v1/playlists/{pid}").mock(
        return_value=httpx.Response(
            200,
            json={"id": pid, "owner": {"id": "user123456789012345678901"}},
        )
    )
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": "user123456789012345678901"})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(
            403,
            json={"error": {"message": "Restriction violated"}},
        )
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": [{"id": "d1", "is_active": True}]})
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run(
        "spotify_play_playlist",
        {"playlist_id": "6zCID88oNjNv9zx6puDHKj"},
    )
    runner.close()
    data = json.loads(raw)
    assert "spotify_error_body_redacted" in data
    assert token not in str(data.get("spotify_error_body_redacted") or "")
    append_tool_trace_record(
        data_dir,
        conversation_id="trace-fail-play",
        tool_name="spotify_play_playlist",
        args_summary='{"playlist_id":"6zCID88oNjNv9zx6puDHKj"}',
        outcome="error",
        raw_result=raw,
        known_secrets=[token],
    )
    row = json.loads(tool_trace_log_path(data_dir).read_text(encoding="utf-8").strip().splitlines()[-1])
    assert row.get("spotify_error_body_redacted")
    assert token not in row["spotify_error_body_redacted"]


@respx.mock
def test_issue8_sse_vague_playlist_trace_writes_redacted_error_on_disk(
    data_dir, signed_in_tokens, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key-trace-playlist")
    me_id = "user123456789012345678901"
    pid = "ownedpl000000000000004"
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": me_id})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/playlists.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": pid,
                        "name": "Trace Playlist",
                        "owner": {"id": me_id},
                        "tracks": {"total": 2},
                    },
                ]
            },
        )
    )
    respx.get(f"https://api.spotify.com/v1/playlists/{pid}").mock(
        return_value=httpx.Response(200, json={"id": pid, "owner": {"id": me_id}})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(
            403,
            json={"error": {"message": "Restriction violated", "status": 403}},
        )
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": [{"id": "d1", "is_active": True}]})
    )
    client = TestClient(app)
    resp = client.post(
        "/api/chat/stream",
        json={"message": "Play one of my playlists", "conversation_id": "trace-vague-pl"},
    )
    assert resp.status_code == 200
    path = tool_trace_log_path(data_dir)
    assert path.is_file()
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    play_rows = [r for r in rows if r.get("tool") == "spotify_play_playlist"]
    assert play_rows
    err_row = play_rows[-1]
    assert err_row.get("outcome") == "error"
    body = err_row.get("spotify_error_body_redacted") or ""
    assert body
    assert "test-access-token" not in body
    assert "Restriction" in body or "403" in body
