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


BLINDING_TRACK_ID = "0VjIjW4GlUZAMYd2vXMi3b"
BLINDING_ALBUM_ID = "0S0KGZnfBGSIssfF54WSJh"
BLINDING_URI = f"spotify:track:{BLINDING_TRACK_ID}"


def _blinding_search_json(
    blinding_uri: str = BLINDING_URI,
    album_id: str = BLINDING_ALBUM_ID,
) -> dict:
    return {
        "tracks": {
            "items": [
                {
                    "name": "Or Nah (feat. The Weeknd, Wiz Khalifa & DJ Mustard) - Remix",
                    "popularity": 80,
                    "artists": [{"name": "Ty Dolla $ign"}, {"name": "The Weeknd"}],
                    "uri": "spotify:track:ornahremix00000001",
                    "album": {"id": "albumornah00000000001"},
                },
                {
                    "name": "Blinding Lights",
                    "popularity": 95,
                    "artists": [{"name": "The Weeknd"}],
                    "uri": blinding_uri,
                    "album": {"id": album_id},
                },
            ]
        }
    }


@respx.mock
def test_play_track_uses_album_context_first(data_dir, signed_in_tokens) -> None:
    blinding_uri = BLINDING_URI
    album_id = BLINDING_ALBUM_ID
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(200, json=_blinding_search_json())
    )
    play_bodies: list[dict] = []

    def play_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode() or "{}")
        play_bodies.append(body)
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
    respx.get("https://api.spotify.com/v1/me/player/queue").mock(
        return_value=httpx.Response(200, json={"queue": []})
    )

    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run(
        "spotify_play_track",
        {"track_name": "Blinding Lights", "artist_name": "The Weeknd"},
    )
    runner.close()
    data = json.loads(raw)
    assert play_bodies
    first = play_bodies[0]
    assert first.get("context_uri") == f"spotify:album:{album_id}"
    assert first.get("offset") == {"uri": blinding_uri}
    assert "uris" not in first
    assert data.get("playback_verified") is True
    assert data.get("uri") == blinding_uri


@respx.mock
def test_play_track_uris_fallback_when_context_readback_fails(data_dir, signed_in_tokens) -> None:
    blinding_uri = BLINDING_URI
    album_id = BLINDING_ALBUM_ID
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(200, json=_blinding_search_json())
    )
    play_bodies: list[dict] = []

    def play_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode() or "{}")
        play_bodies.append(body)
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        side_effect=play_handler
    )

    def player_handler(_request: httpx.Request) -> httpx.Response:
        if play_bodies and play_bodies[-1].get("uris"):
            return httpx.Response(
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
        return httpx.Response(
            200,
            json={
                "is_playing": True,
                "item": {
                    "name": "Old Song",
                    "uri": "spotify:track:bbbbbbbbbbbbbbbbbbbbbb",
                    "artists": [{"name": "Someone"}],
                },
            },
        )

    respx.get("https://api.spotify.com/v1/me/player").mock(side_effect=player_handler)
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )
    respx.get("https://api.spotify.com/v1/me/player/queue").mock(
        return_value=httpx.Response(200, json={"queue": []})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/shuffle.*").mock(
        return_value=httpx.Response(204)
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/repeat.*").mock(
        return_value=httpx.Response(204)
    )

    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run(
        "spotify_play_track",
        {"track_name": "Blinding Lights", "artist_name": "The Weeknd"},
    )
    runner.close()
    data = json.loads(raw)
    assert play_bodies[0].get("context_uri") == f"spotify:album:{album_id}"
    assert any(b.get("uris") == [blinding_uri] for b in play_bodies), "expected uris fallback PUT"
    assert data.get("playback_verified") is True


@respx.mock
def test_play_track_restore_never_posts_queue(data_dir, signed_in_tokens) -> None:
    blinding_uri = BLINDING_URI
    album_id = BLINDING_ALBUM_ID
    prior_track = "spotify:track:aaaaaaaaaaaaaaaaaaaaaa"
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(200, json=_blinding_search_json())
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    manual = "spotify:track:cccccccccccccccccccccc"

    def queue_get(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "currently_playing": {"uri": prior_track},
                "queue": [{"uri": manual}],
            },
        )

    respx.get("https://api.spotify.com/v1/me/player/queue").mock(side_effect=queue_get)
    queue_posts: list[str] = []

    def queue_post(request: httpx.Request) -> httpx.Response:
        queue_posts.append(request.url.params.get("uri", ""))
        return httpx.Response(204)

    respx.post(url__regex=r"https://api\.spotify\.com/v1/me/player/queue.*").mock(
        side_effect=queue_post
    )
    player_reads = {"n": 0}

    def player_get(_request: httpx.Request) -> httpx.Response:
        player_reads["n"] += 1
        if player_reads["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "is_playing": True,
                    "progress_ms": 1200,
                    "context": {"uri": f"spotify:album:{album_id}"},
                    "item": {"uri": prior_track, "name": "Prior"},
                },
            )
        return httpx.Response(
            200,
            json={
                "is_playing": True,
                "item": {"uri": prior_track, "name": "Prior"},
            },
        )

    respx.get("https://api.spotify.com/v1/me/player").mock(side_effect=player_get)
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/shuffle.*").mock(
        return_value=httpx.Response(204)
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/repeat.*").mock(
        return_value=httpx.Response(204)
    )

    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run(
        "spotify_play_track",
        {"track_name": "Blinding Lights", "artist_name": "The Weeknd"},
    )
    runner.close()
    data = json.loads(raw)
    assert data.get("playback_verified") is False
    assert queue_posts == []


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
    blinding_uri = BLINDING_URI
    album_id = BLINDING_ALBUM_ID
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
                    "uri": blinding_uri,
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
    outcome = try_deterministic_chat_reply(
        "Play Blinding Lights by The Weeknd",
        runner,
    )
    runner.close()
    assert outcome is not None
    assert outcome.tool_names() == ["spotify_play_track"]
    assert "Blinding Lights" in outcome.reply
    assert "Or Nah" not in outcome.reply
