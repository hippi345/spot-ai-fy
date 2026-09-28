"""Regression tests for laptop agent issues 1–7 (mocked Spotify + chat path)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from spot_backend.capability_replies import try_capability_question_reply
from spot_backend.chat_shortcuts import try_deterministic_chat_reply
from spot_backend.config import Settings
from spot_backend.deterministic_chat import resolve_deterministic_chat_outcome
from spot_backend.gemini_llm import gemini_intent_allowed_function_names
from spot_backend.play_bare_intent import extract_bare_play_target, extract_play_music_by_artist
from spot_backend.playlist_pick import playlist_id_is_spotify_curated
from spot_backend.prompt_intent import prompt_is_capability_question
from spot_backend.reply_tool_trace import tool_trace_outcome
from spot_backend.spotify_tools import SpotifyToolRunner
from tests.test_play_track_selection import BLINDING_URI, _blinding_search_json


@respx.mock
def test_issue1_play_blinding_lights_uses_bare_track_not_obscure_artist(
    data_dir, signed_in_tokens
) -> None:
    def search_handler(request: httpx.Request) -> httpx.Response:
        qtype = request.url.params.get("type")
        if qtype == "track":
            return httpx.Response(200, json=_blinding_search_json())
        return httpx.Response(
            200,
            json={
                "artists": {
                    "items": [
                        {
                            "id": "obscureartist000000001",
                            "name": "Blinding Lights",
                            "popularity": 3,
                        }
                    ]
                }
            },
        )

    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        side_effect=search_handler
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
    respx.get("https://api.spotify.com/v1/me/player/queue").mock(
        return_value=httpx.Response(200, json={"queue": []})
    )

    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply("Play Blinding Lights", runner)
    runner.close()
    assert outcome is not None
    assert outcome.tool_names()[0] == "spotify_play_bare"
    assert "Weeknd" in outcome.reply or "Blinding Lights" in outcome.reply
    assert "obscure" not in outcome.reply.lower()


def test_issue1_gemini_any_mode_allows_track_for_bare_play() -> None:
    names = gemini_intent_allowed_function_names("Play Blinding Lights")
    assert names == ["spotify_play_track", "spotify_play_artist"]


@respx.mock
def test_issue2_vague_playlist_skips_curated_and_retries_owned(data_dir, signed_in_tokens) -> None:
    me_id = "user123456789012345678901"
    owned_id = "ownedpl000000000000001"
    curated_id = "37i9dQZF1EpyS73yJ89M1Q"
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": me_id, "display_name": "Me"})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/playlists.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": curated_id,
                        "name": "Discover Weekly",
                        "owner": {"id": "spotify"},
                    },
                    {
                        "id": owned_id,
                        "name": "My Mix",
                        "owner": {"id": me_id},
                    },
                ]
            },
        )
    )
    respx.get(f"https://api.spotify.com/v1/playlists/{owned_id}").mock(
        return_value=httpx.Response(
            200,
            json={"id": owned_id, "owner": {"id": me_id}, "name": "My Mix"},
        )
    )
    play_calls: list[str] = []

    def play_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode() or "{}")
        ctx = body.get("context_uri") or ""
        play_calls.append(ctx)
        if curated_id in ctx:
            return httpx.Response(404, json={"error": {"message": "Not found"}})
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        side_effect=play_handler
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
    assert curated_id not in outcome.reply
    assert "404" not in outcome.reply
    assert "37i9dQZF" not in outcome.reply
    assert "My Mix" in outcome.reply
    assert not any(curated_id in c for c in play_calls)


def test_issue3_capability_make_playlist_has_no_tools(data_dir, signed_in_tokens) -> None:
    q = "Can you make me a playlist?"
    assert prompt_is_capability_question(q)
    runner = SpotifyToolRunner(settings=Settings())
    outcome = resolve_deterministic_chat_outcome(q, runner, conversation_id="cap-pl")
    runner.close()
    assert outcome is not None
    assert outcome.tool_names() == []
    assert "vibe" in outcome.reply.lower() or "artists" in outcome.reply.lower()


@respx.mock
def test_issue3_create_playlist_refuses_empty(data_dir, signed_in_tokens) -> None:
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_create_playlist", {"name": "My new playlist"})
    runner.close()
    data = json.loads(raw)
    assert data.get("refuse_empty_playlist")
    assert respx.calls.call_count == 0


@respx.mock
def test_issue4_when_did_this_song_uses_playback_not_queue(data_dir, signed_in_tokens) -> None:
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "item": {
                    "name": "Alone Again",
                    "album": {"id": "alb1", "release_date": "2020-03-20"},
                    "artists": [{"name": "The Weeknd"}],
                }
            },
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply("When did this song come out?", runner)
    runner.close()
    assert outcome is not None
    assert outcome.tool_names() == ["spotify_playback_state"]
    assert "Alone Again" in outcome.reply
    assert "2020-03-20" in outcome.reply
    assert "Starboy" not in outcome.reply


def test_issue5_what_can_you_do_is_short_capability(data_dir, signed_in_tokens) -> None:
    reply = try_capability_question_reply("What can you do?")
    assert reply
    assert reply.count(".") <= 3
    assert len(reply) < 320


@respx.mock
def test_issue6_play_songs_by_unknown_artist(data_dir, signed_in_tokens) -> None:
    assert extract_play_music_by_artist("Play songs by Zxqvbrt Plonk") == "Zxqvbrt Plonk"
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*type=artist.*").mock(
        return_value=httpx.Response(200, json={"artists": {"items": []}})
    )
    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply("Play songs by Zxqvbrt Plonk", runner)
    runner.close()
    assert outcome is not None
    assert "couldn't find an artist called Zxqvbrt Plonk" in outcome.reply
    assert "track:\"songs\"" not in outcome.reply


def test_issue7_optional_playlist_lookup_trace_not_error() -> None:
    raw = json.dumps(
        {
            "ok": False,
            "optional_lookup_failure": True,
            "user_message": "That playlist isn't available to this app.",
        }
    )
    assert tool_trace_outcome(raw) == "ok"


def test_curated_playlist_id_detection() -> None:
    assert playlist_id_is_spotify_curated("37i9dQZF1EpyS73yJ89M1Q")


def test_bare_play_target_excludes_playlists_phrase() -> None:
    assert extract_bare_play_target("Play one of my playlists") is None
