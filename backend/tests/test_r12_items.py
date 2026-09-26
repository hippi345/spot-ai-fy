"""Round-12 — test_r12_itemN_* per PR requirements."""

from __future__ import annotations

import json
from typing import Any

import httpx
import respx

from spot_backend.chat_messages import prepare_user_visible_reply
from spot_backend.chat_shortcuts import try_deterministic_chat_reply
from spot_backend.config import Settings
from spot_backend.play_artist import format_play_artist_reply
from spot_backend.spotify_tools import SpotifyToolRunner
from tests.test_r10_items import _mock_artist_top_track_search


def _player_stuck_after_play(
    *,
    prior_context: str,
    prior_track_uri: str,
    prior_track_name: str,
    prior_progress_ms: int,
    prior_is_playing: bool,
) -> Any:
    """First GET = snapshot before play; later GETs = play never took."""
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "is_playing": prior_is_playing,
                    "progress_ms": prior_progress_ms,
                    "context": {"uri": prior_context},
                    "item": {"uri": prior_track_uri, "name": prior_track_name},
                },
            )
        return httpx.Response(200, json={"is_playing": False, "item": None})

    return handler


@respx.mock
def test_r12_item1_artist_play_zero_queue_explicit_queue_still_works(
    data_dir, signed_in_tokens,
) -> None:
    artist_id = "aaaaaaaaaaaaaaaaaaaaaa"
    track_ids = ["bbbbbbbbbbbbbbbbbbbbbb"]
    _mock_artist_top_track_search(artist_id, "Radiohead", track_ids)
    original_uri = "spotify:track:2222222222222222222222"
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [
                        {
                            "name": "Karma Police",
                            "popularity": 70,
                            "artists": [{"name": "Radiohead"}],
                            "uri": original_uri,
                        }
                    ]
                }
            },
        )
    )
    queue_n = {"n": 0}

    def queue_handler(_request: httpx.Request) -> httpx.Response:
        queue_n["n"] += 1
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.post(url__regex=r"https://api\.spotify\.com/v1/me/player/queue.*").mock(
        side_effect=queue_handler
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "context": {"uri": "spotify:album:0000000000000000000030"},
                "item": {"uri": f"spotify:track:{track_ids[0]}"},
            },
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    runner.run("spotify_play_artist", {"artist_name": "Radiohead"})
    assert queue_n["n"] == 0
    outcome = try_deterministic_chat_reply("queue Karma Police by Radiohead", runner)
    runner.close()
    assert outcome is not None
    assert outcome.tool_names() == ["spotify_add_to_queue"]
    assert queue_n["n"] >= 1


@respx.mock
def test_r12_item2_play_failure_restores_prior_state_and_bounded_puts(
    data_dir, signed_in_tokens,
) -> None:
    artist_id = "cccccccccccccccccccccc"
    track_ids = ["dddddddddddddddddddddd"]
    _mock_artist_top_track_search(artist_id, "Radiohead", track_ids)
    prior_context = "spotify:album:eeeeeeeeeeeeeeeeeeeeee"
    prior_track = "spotify:track:ffffffffffffffffffffff"
    prior_name = "Old Song"
    play_calls: list[dict[str, Any]] = []
    pause_n = {"n": 0}

    def capture_play(request: httpx.Request) -> httpx.Response:
        play_calls.append(json.loads(request.content or b"{}"))
        return httpx.Response(204)

    def capture_pause(_request: httpx.Request) -> httpx.Response:
        pause_n["n"] += 1
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        side_effect=capture_play
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/pause.*").mock(
        side_effect=capture_pause
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        side_effect=_player_stuck_after_play(
            prior_context=prior_context,
            prior_track_uri=prior_track,
            prior_track_name=prior_name,
            prior_progress_ms=45000,
            prior_is_playing=False,
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_play_artist", {"artist_name": "Radiohead"})
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is False
    assert len(play_calls) <= 3
    restore = play_calls[-1]
    assert restore["context_uri"] == prior_context
    assert restore["offset"]["uri"] == prior_track
    assert restore["position_ms"] == 45000
    assert "uris" not in restore
    assert pause_n["n"] == 1
    expected = (
        "Spotify wouldn't play Radiohead on this device, so I went back to Old Song."
    )
    assert data.get("user_message") == expected
    assert format_play_artist_reply("Radiohead", raw) == expected


@respx.mock
def test_r12_item2_play_failure_nothing_playing_before_no_restore(
    data_dir, signed_in_tokens,
) -> None:
    track_id = "1111111111111111111111"
    album_id = "2222222222222222222222"
    respx.get(f"https://api.spotify.com/v1/tracks/{track_id}").mock(
        return_value=httpx.Response(
            200,
            json={"id": track_id, "uri": f"spotify:track:{track_id}", "album": {"id": album_id}},
        )
    )
    respx.get(f"https://api.spotify.com/v1/albums/{album_id}").mock(
        return_value=httpx.Response(200, json={"id": album_id, "name": "OK Computer"})
    )
    play_n = {"n": 0}

    def capture_play(_request: httpx.Request) -> httpx.Response:
        play_n["n"] += 1
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        side_effect=capture_play
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        side_effect=_player_stuck_after_play(
            prior_context="",
            prior_track_uri="",
            prior_track_name="",
            prior_progress_ms=0,
            prior_is_playing=False,
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run(
        "spotify_start_resume_playback",
        {
            "context_uri": f"spotify:album:{album_id}",
            "offset": {"uri": f"spotify:track:{track_id}"},
            "playback_request_label": "Creep",
        },
    )
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is False
    assert data.get("playback_restored") is False
    assert data.get("user_message") == "Spotify wouldn't play Creep on this device."
    assert play_n["n"] <= 3


@respx.mock
def test_r12_item2_playlist_path_restore_on_stuck_play(data_dir, signed_in_tokens) -> None:
    playlist_id = "5555555555555555555555"
    prior_context = "spotify:playlist:3333333333333333333333"
    prior_track = "spotify:track:4444444444444444444444"
    play_calls: list[dict[str, Any]] = []

    respx.get(f"https://api.spotify.com/v1/playlists/{playlist_id}").mock(
        return_value=httpx.Response(200, json={"id": playlist_id, "name": "Mix"})
    )

    def capture_play(request: httpx.Request) -> httpx.Response:
        play_calls.append(json.loads(request.content or b"{}"))
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        side_effect=capture_play
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        side_effect=_player_stuck_after_play(
            prior_context=prior_context,
            prior_track_uri=prior_track,
            prior_track_name="Prior Track",
            prior_progress_ms=1200,
            prior_is_playing=True,
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run(
        "spotify_play_playlist",
        {"playlist_id": playlist_id, "playback_request_label": "Mix"},
    )
    runner.close()
    data = json.loads(raw)
    playback = data.get("playback") if isinstance(data, dict) else None
    assert isinstance(playback, dict)
    assert playback.get("ok") is False
    assert playback.get("restore_body", {}).get("context_uri") == prior_context
    assert len(play_calls) <= 3


def test_r12_item3_make_private_visibility_note_once() -> None:
    assistant = (
        "Your playlist is private now. "
        "Note: I asked Spotify to make it private, but Spotify still shows it as public "
        "(this can lag or be a known Spotify API quirk)."
    )
    tool_json = json.dumps(
        {
            "ok": False,
            "visibility_change_requested": True,
            "visibility_warning": (
                "I asked Spotify to make it private, but Spotify still shows it as public "
                "(this can lag or be a known Spotify API quirk)."
            ),
        }
    )
    reply = prepare_user_visible_reply(assistant, [tool_json])
    assert reply.count("still shows it as public") == 1


def test_r12_item3_create_reply_has_no_visibility_note() -> None:
    assistant = "Created your playlist Mix."
    tool_json = json.dumps(
        {
            "id": "aaaaaaaaaaaaaaaaaaaaaa",
            "name": "Mix",
            "public": True,
            "visibility_warning": (
                "Spotify still reports this playlist as public after creation. "
                "You may need to set visibility manually in the Spotify app."
            ),
        }
    )
    reply = prepare_user_visible_reply(assistant, [tool_json])
    assert "still reports this playlist as public" not in reply
    assert reply.strip() == "Created your playlist Mix."
