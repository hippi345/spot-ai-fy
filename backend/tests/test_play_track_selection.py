from __future__ import annotations

import json

import httpx
import respx

from spot_backend.chat_shortcuts import try_deterministic_chat_reply
from spot_backend.play_track_intent import extract_play_track_request
from spot_backend.spotify_tools import SpotifyToolRunner, _score_track_search_candidate
from spot_backend.config import Settings


def test_extract_play_track_request() -> None:
    assert extract_play_track_request("Play Blinding Lights by The Weeknd") == (
        "Blinding Lights",
        "The Weeknd",
    )
    assert extract_play_track_request("play Radiohead") is None


def test_or_nah_remix_loses_to_blinding_lights() -> None:
    original = {
        "name": "Blinding Lights",
        "popularity": 90,
        "artists": [{"name": "The Weeknd"}],
    }
    remix = {
        "name": "Or Nah (feat. The Weeknd, Wiz Khalifa & DJ Mustard) - Remix",
        "popularity": 85,
        "artists": [{"name": "Ty Dolla $ign"}, {"name": "The Weeknd"}],
    }
    s_orig = _score_track_search_candidate(
        original, want_title="Blinding Lights", want_artist="The Weeknd"
    )
    s_remix = _score_track_search_candidate(
        remix, want_title="Blinding Lights", want_artist="The Weeknd"
    )
    assert s_orig > s_remix


@respx.mock
def test_play_track_uses_direct_uri_not_queue(data_dir, signed_in_tokens) -> None:
    blinding_uri = "spotify:track:blindinglights00001"
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [
                        {
                            "name": "Or Nah (feat. The Weeknd, Wiz Khalifa & DJ Mustard) - Remix",
                            "popularity": 80,
                            "artists": [{"name": "Ty Dolla $ign"}, {"name": "The Weeknd"}],
                            "uri": "spotify:track:ornahremix00000001",
                        },
                        {
                            "name": "Blinding Lights",
                            "popularity": 95,
                            "artists": [{"name": "The Weeknd"}],
                            "uri": blinding_uri,
                        },
                    ]
                }
            },
        )
    )
    play_calls = {"n": 0}

    def play_handler(request: httpx.Request) -> httpx.Response:
        play_calls["n"] += 1
        body = json.loads(request.content.decode() or "{}")
        assert body.get("uris") == [blinding_uri]
        assert "context_uri" not in body
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        side_effect=play_handler
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "item": {
                    "name": "Blinding Lights",
                    "uri": blinding_uri,
                    "artists": [{"name": "The Weeknd"}],
                },
            },
        )
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )
    respx.post(url__regex=r"https://api\.spotify\.com/v1/me/player/queue.*").mock(
        return_value=httpx.Response(204)
    )

    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run(
        "spotify_play_track",
        {"track_name": "Blinding Lights", "artist_name": "The Weeknd"},
    )
    runner.close()
    data = json.loads(raw)
    assert play_calls["n"] >= 1
    assert data.get("playback_verified") is True
    assert data.get("uri") == blinding_uri


@respx.mock
def test_skip_verifies_track_change(data_dir, signed_in_tokens) -> None:
    uris = ["spotify:track:aaaa1111", "spotify:track:bbbb2222"]
    calls = {"n": 0}

    def player_handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        idx = 0 if calls["n"] <= 2 else 1
        return httpx.Response(
            200,
            json={
                "is_playing": True,
                "item": {"name": f"Track {idx}", "uri": uris[idx], "artists": [{"name": "A"}]},
            },
        )

    respx.get("https://api.spotify.com/v1/me/player").mock(side_effect=player_handler)
    respx.post("https://api.spotify.com/v1/me/player/next").mock(return_value=httpx.Response(204))

    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply("Skip", runner)
    runner.close()
    assert outcome is not None
    assert "bbbb2222" in outcome.reply or "Track 1" in outcome.reply
    assert "did not change" not in outcome.reply


@respx.mock
def test_play_track_chat_reply_from_readback(data_dir, signed_in_tokens) -> None:
    blinding_uri = "spotify:track:blindinglights00001"
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [
                        {
                            "name": "Blinding Lights",
                            "popularity": 95,
                            "artists": [{"name": "The Weeknd"}],
                            "uri": blinding_uri,
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
                "item": {
                    "name": "Blinding Lights",
                    "uri": blinding_uri,
                    "artists": [{"name": "The Weeknd"}],
                },
            },
        )
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )

    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply(
        "Play Blinding Lights by The Weeknd",
        runner,
    )
    runner.close()
    assert outcome is not None
    assert outcome.tool_names() == ["spotify_play_track"]
    assert "Blinding Lights" in outcome.reply
    assert "Or Nah" not in outcome.reply
