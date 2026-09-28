"""Round-13 — test_r13_itemN_* per PR requirements."""

from __future__ import annotations

import json
from typing import Any

import httpx
import respx

from spot_backend.chat_messages import prepare_user_visible_reply
from spot_backend.config import Settings
from spot_backend.spotify_tools import SpotifyToolRunner
from tests.test_r10_items import _mock_artist_top_track_search


def test_r13_item1_create_reply_private_claim_corrected_when_tool_public() -> None:
    tool_json = json.dumps(
        {
            "id": "aaaaaaaaaaaaaaaaaaaaaa",
            "name": "zz-r12-test",
            "public": True,
            "assistant_reply_instruction": "say public when public is true",
        }
    )
    assistant = "Created zz-r12-test. It's currently private."
    reply = prepare_user_visible_reply(assistant, [tool_json])
    assert "currently private" not in reply.lower()
    assert "public on Spotify" in reply


def test_r13_item1_create_reply_consistent_public_left_alone() -> None:
    tool_json = json.dumps(
        {"id": "bbbbbbbbbbbbbbbbbbbbbb", "name": "Mix", "public": True}
    )
    assistant = "Created Mix — it's public on Spotify."
    reply = prepare_user_visible_reply(assistant, [tool_json])
    assert reply == assistant


@respx.mock
def test_r13_item2_make_private_exactly_one_visibility_note() -> None:
    assistant = (
        "Done. Spotify is still showing it as public on my side; it may take a delay to sync."
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
    assert "may take a delay" not in reply.lower()


@respx.mock
def test_r13_item3_restore_does_not_requeue_manual_queue(
    data_dir, signed_in_tokens,
) -> None:
    album_id = "aaaaaaaaaaaaaaaaaaaaaa"
    cur_track = "spotify:track:1111111111111111111111"
    ctx_track2 = "spotify:track:2222222222222222222222"
    ctx_track3 = "spotify:track:3333333333333333333333"
    manual_a = "spotify:track:4444444444444444444444"
    manual_b = "spotify:track:5555555555555555555555"
    artist_id = "cccccccccccccccccccccc"
    track_ids = ["dddddddddddddddddddddd"]
    _mock_artist_top_track_search(artist_id, "Radiohead", track_ids)
    queue_posts: list[str] = []
    player_calls = {"n": 0}

    respx.get(f"https://api.spotify.com/v1/albums/{album_id}/tracks").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {"uri": cur_track},
                    {"uri": ctx_track2},
                    {"uri": ctx_track3},
                ]
            },
        )
    )

    queue_reads = {"n": 0}

    def queue_get(_request: httpx.Request) -> httpx.Response:
        queue_reads["n"] += 1
        if queue_reads["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "currently_playing": {"uri": cur_track},
                    "queue": [
                        {"uri": ctx_track2},
                        {"uri": manual_a},
                        {"uri": manual_b},
                    ],
                },
            )
        return httpx.Response(200, json={"currently_playing": {"uri": cur_track}, "queue": []})

    respx.get("https://api.spotify.com/v1/me/player/queue").mock(side_effect=queue_get)

    def capture_queue_post(request: httpx.Request) -> httpx.Response:
        queue_posts.append(request.url.params.get("uri", ""))
        return httpx.Response(204)

    respx.post(url__regex=r"https://api\.spotify\.com/v1/me/player/queue.*").mock(
        side_effect=capture_queue_post
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )

    def player_get(_request: httpx.Request) -> httpx.Response:
        player_calls["n"] += 1
        if player_calls["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "is_playing": True,
                    "progress_ms": 1000,
                    "context": {"uri": f"spotify:album:{album_id}"},
                    "item": {"uri": cur_track, "name": "Current"},
                },
            )
        return httpx.Response(200, json={"is_playing": False, "item": None})

    respx.get("https://api.spotify.com/v1/me/player").mock(side_effect=player_get)
    runner = SpotifyToolRunner(settings=Settings())
    runner.run("spotify_play_artist", {"artist_name": "Radiohead"})
    runner.close()
    assert queue_posts == []


@respx.mock
def test_r13_item3_queue_read_failure_adds_caution(data_dir, signed_in_tokens) -> None:
    track_id = "1111111111111111111111"
    album_id = "2222222222222222222222"
    respx.get(f"https://api.spotify.com/v1/tracks/{track_id}").mock(
        return_value=httpx.Response(
            200,
            json={"id": track_id, "uri": f"spotify:track:{track_id}", "album": {"id": album_id}},
        )
    )
    respx.get(f"https://api.spotify.com/v1/albums/{album_id}").mock(
        return_value=httpx.Response(200, json={"id": album_id})
    )
    respx.get("https://api.spotify.com/v1/me/player/queue").mock(
        return_value=httpx.Response(500, json={"error": "nope"})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
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
                    "context": {"uri": f"spotify:album:{album_id}"},
                    "item": {"uri": f"spotify:track:{track_id}", "name": "Old"},
                },
            )
        return httpx.Response(200, json={"is_playing": False, "item": None})

    respx.get("https://api.spotify.com/v1/me/player").mock(side_effect=player_get)
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
    assert "Your queue may have been cleared." in data.get("user_message", "")


@respx.mock
def test_r13_item3_successful_play_does_not_post_queue(data_dir, signed_in_tokens) -> None:
    artist_id = "eeeeeeeeeeeeeeeeeeeeee"
    track_ids = ["ffffffffffffffffffffff"]
    _mock_artist_top_track_search(artist_id, "Radiohead", track_ids)
    queue_posts: list[str] = []

    def capture_queue(_request: httpx.Request) -> httpx.Response:
        queue_posts.append("x")
        return httpx.Response(204)

    respx.post(url__regex=r"https://api\.spotify\.com/v1/me/player/queue.*").mock(
        side_effect=capture_queue
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
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
    respx.get("https://api.spotify.com/v1/me/player/queue").mock(
        return_value=httpx.Response(200, json={"queue": [{"uri": "spotify:track:9999999999999999999999"}]})
    )
    runner = SpotifyToolRunner(settings=Settings())
    runner.run("spotify_play_artist", {"artist_name": "Radiohead"})
    runner.close()
    assert queue_posts == []


@respx.mock
def test_r13_item4_fallback_was_paused_restore_stays_paused(data_dir, signed_in_tokens) -> None:
    artist_id = "1111111111111111111111"
    track_ids = ["2222222222222222222222"]
    _mock_artist_top_track_search(artist_id, "Radiohead", track_ids)
    respx.get("https://api.spotify.com/v1/albums/3333333333333333333333/tracks").mock(
        return_value=httpx.Response(
            200,
            json={"items": [{"uri": "spotify:track:4444444444444444444444"}]},
        )
    )
    pause_n = {"n": 0}

    def capture_pause(_request: httpx.Request) -> httpx.Response:
        pause_n["n"] += 1
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/pause.*").mock(
        side_effect=capture_pause
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player/queue").mock(
        return_value=httpx.Response(200, json={"queue": []})
    )
    player_calls = {"n": 0}

    def player_get(_request: httpx.Request) -> httpx.Response:
        player_calls["n"] += 1
        if player_calls["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "is_playing": False,
                    "progress_ms": 88000,
                    "context": {"uri": "spotify:album:3333333333333333333333"},
                    "item": {"uri": "spotify:track:4444444444444444444444", "name": "Paused Song"},
                },
            )
        return httpx.Response(200, json={"is_playing": False, "item": None})

    respx.get("https://api.spotify.com/v1/me/player").mock(side_effect=player_get)
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_play_artist", {"artist_name": "Radiohead"})
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is False
    playback = data.get("playback") if isinstance(data.get("playback"), dict) else {}
    restore = playback.get("restore_body")
    assert isinstance(restore, dict)
    assert restore.get("position_ms") == 88000
    assert pause_n["n"] >= 1
