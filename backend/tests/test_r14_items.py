"""Round-14 — test_r14_itemN_* per PR requirements."""

from __future__ import annotations

import json
from typing import Any

import httpx
import respx

from spot_backend.chat_messages import prepare_user_visible_reply
from spot_backend.config import Settings
from spot_backend.spotify_tools import SpotifyToolRunner
from tests.test_r10_items import _mock_artist_top_track_search


def test_r14_item1_make_private_mismatch_strips_success_claim_one_note() -> None:
    assistant = 'I\'ve updated the playlist "zz-r13-test" to be private.'
    tool_json = json.dumps(
        {
            "ok": False,
            "visibility_mismatch": True,
            "visibility_change_requested": True,
            "public": True,
            "visibility_warning": (
                "I asked Spotify to make it private, but Spotify still shows it as public "
                "(this can lag or be a known Spotify API quirk)."
            ),
            "visibility_result": (
                "Update request sent, but Spotify still reports this playlist as public; "
                "do not say the playlist is private or that the change succeeded."
            ),
        }
    )
    reply = prepare_user_visible_reply(assistant, [tool_json])
    low = reply.lower()
    assert "to be private" not in low
    assert "is now private" not in low
    assert "made private" not in low
    assert reply.count("still shows it as public") == 1


def test_r14_item1_make_private_success_keeps_claim_no_note() -> None:
    assistant = 'I\'ve updated the playlist "zz-r13-test" to be private.'
    tool_json = json.dumps(
        {
            "ok": True,
            "visibility_change_requested": True,
            "public": False,
            "updated_fields": ["public"],
        }
    )
    reply = prepare_user_visible_reply(assistant, [tool_json])
    assert "to be private" in reply.lower()
    assert "still shows it as public" not in reply.lower()


@respx.mock
def test_r14_item2_restore_uses_saved_album_context_and_track_offset(
    data_dir, signed_in_tokens,
) -> None:
    album_id = "aaaaaaaaaaaaaaaaaaaaaa"
    current_track = "spotify:track:1111111111111111111111"
    next_track = "spotify:track:2222222222222222222222"
    artist_id = "bbbbbbbbbbbbbbbbbbbbbb"
    track_ids = ["cccccccccccccccccccccc"]
    _mock_artist_top_track_search(artist_id, "Gunna", track_ids)
    play_calls: list[dict[str, Any]] = []

    respx.get(f"https://api.spotify.com/v1/albums/{album_id}/tracks").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {"uri": current_track},
                    {"uri": next_track},
                    {"uri": "spotify:track:3333333333333333333333"},
                ]
            },
        )
    )

    def capture_play(request: httpx.Request) -> httpx.Response:
        play_calls.append(json.loads(request.content or b"{}"))
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/shuffle.*").mock(
        return_value=httpx.Response(204)
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/repeat.*").mock(
        return_value=httpx.Response(204)
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        side_effect=capture_play
    )
    respx.get("https://api.spotify.com/v1/me/player/queue").mock(
        return_value=httpx.Response(
            200,
            json={
                "currently_playing": {"uri": current_track},
                "queue": [{"uri": next_track}],
            },
        )
    )
    player_calls = {"n": 0}

    def player_get(_request: httpx.Request) -> httpx.Response:
        player_calls["n"] += 1
        if player_calls["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "is_playing": True,
                    "progress_ms": 42000,
                    "shuffle_state": True,
                    "repeat_state": "context",
                    "context": {"uri": f"spotify:album:{album_id}"},
                    "item": {"uri": current_track, "name": "On ICE"},
                },
            )
        return httpx.Response(200, json={"is_playing": False, "item": None})

    respx.get("https://api.spotify.com/v1/me/player").mock(side_effect=player_get)
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_play_artist", {"artist_name": "Gunna"})
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is False
    playback = data.get("playback") if isinstance(data.get("playback"), dict) else {}
    restore = playback.get("restore_body")
    assert isinstance(restore, dict)
    assert restore["context_uri"] == f"spotify:album:{album_id}"
    assert restore["offset"] == {"uri": current_track}
    assert restore["position_ms"] == 42000


@respx.mock
def test_r14_item2_restore_shuffle_repeat_bounded(
    data_dir, signed_in_tokens,
) -> None:
    album_id = "dddddddddddddddddddddd"
    current_track = "spotify:track:eeeeeeeeeeeeeeeeeeeeee"
    artist_id = "ffffffffffffffffffffff"
    track_ids = ["0000000000000000000001"]
    _mock_artist_top_track_search(artist_id, "Gunna", track_ids)
    shuffle_calls: list[dict[str, str]] = []
    repeat_calls: list[dict[str, str]] = []

    respx.get(f"https://api.spotify.com/v1/albums/{album_id}/tracks").mock(
        return_value=httpx.Response(
            200,
            json={"items": [{"uri": current_track}, {"uri": "spotify:track:9999999999999999999999"}]},
        )
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player/queue").mock(
        return_value=httpx.Response(200, json={"queue": []})
    )

    def capture_shuffle(request: httpx.Request) -> httpx.Response:
        shuffle_calls.append(dict(request.url.params))
        return httpx.Response(204)

    def capture_repeat(request: httpx.Request) -> httpx.Response:
        repeat_calls.append(dict(request.url.params))
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/shuffle.*").mock(
        side_effect=capture_shuffle
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/repeat.*").mock(
        side_effect=capture_repeat
    )
    player_calls = {"n": 0}

    def player_get(_request: httpx.Request) -> httpx.Response:
        player_calls["n"] += 1
        if player_calls["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "is_playing": True,
                    "progress_ms": 1000,
                    "shuffle_state": True,
                    "repeat_state": "off",
                    "context": {"uri": f"spotify:album:{album_id}"},
                    "item": {"uri": current_track, "name": "Current"},
                },
            )
        return httpx.Response(200, json={"is_playing": False, "item": None})

    respx.get("https://api.spotify.com/v1/me/player").mock(side_effect=player_get)
    runner = SpotifyToolRunner(settings=Settings())
    runner.run("spotify_play_artist", {"artist_name": "Gunna"})
    runner.close()
    assert len(shuffle_calls) == 1
    assert shuffle_calls[0].get("state") == "true"
    assert len(repeat_calls) == 1
    assert repeat_calls[0].get("state") == "off"


@respx.mock
def test_r14_item2_restore_play_body_unchanged_and_no_queue_posts(
    data_dir, signed_in_tokens,
) -> None:
    album_id = "1111111111111111111111"
    current_track = "spotify:track:2222222222222222222222"
    ctx_next = "spotify:track:3333333333333333333333"
    manual = "spotify:track:4444444444444444444444"
    artist_id = "5555555555555555555555"
    track_ids = ["6666666666666666666666"]
    _mock_artist_top_track_search(artist_id, "Gunna", track_ids)
    play_calls: list[dict[str, Any]] = []
    queue_posts: list[str] = []

    respx.get(f"https://api.spotify.com/v1/albums/{album_id}/tracks").mock(
        return_value=httpx.Response(
            200,
            json={"items": [{"uri": current_track}, {"uri": ctx_next}]},
        )
    )

    def capture_play(request: httpx.Request) -> httpx.Response:
        play_calls.append(json.loads(request.content or b"{}"))
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/shuffle.*").mock(
        return_value=httpx.Response(204)
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/repeat.*").mock(
        return_value=httpx.Response(204)
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        side_effect=capture_play
    )

    queue_reads = {"n": 0}

    def queue_get(_request: httpx.Request) -> httpx.Response:
        queue_reads["n"] += 1
        if queue_reads["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "currently_playing": {"uri": current_track},
                    "queue": [{"uri": ctx_next}, {"uri": manual}],
                },
            )
        return httpx.Response(200, json={"queue": []})

    respx.get("https://api.spotify.com/v1/me/player/queue").mock(side_effect=queue_get)

    def capture_queue_post(request: httpx.Request) -> httpx.Response:
        queue_posts.append(request.url.params.get("uri", ""))
        return httpx.Response(204)

    respx.post(url__regex=r"https://api\.spotify\.com/v1/me/player/queue.*").mock(
        side_effect=capture_queue_post
    )
    player_calls = {"n": 0}

    def player_get(_request: httpx.Request) -> httpx.Response:
        player_calls["n"] += 1
        if player_calls["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "is_playing": True,
                    "progress_ms": 500,
                    "shuffle_state": False,
                    "repeat_state": "off",
                    "context": {"uri": f"spotify:album:{album_id}"},
                    "item": {"uri": current_track, "name": "On ICE"},
                },
            )
        return httpx.Response(200, json={"is_playing": False, "item": None})

    respx.get("https://api.spotify.com/v1/me/player").mock(side_effect=player_get)
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_play_artist", {"artist_name": "Gunna"})
    runner.close()
    data = json.loads(raw)
    playback = data.get("playback") if isinstance(data.get("playback"), dict) else {}
    restore = playback.get("restore_body")
    assert isinstance(restore, dict)
    assert restore["context_uri"] == f"spotify:album:{album_id}"
    assert restore["offset"] == {"uri": current_track}
    assert "uris" not in restore
    assert queue_posts == []
