"""Round-11 laptop retest — test_r11_itemN_* per PR requirements."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from unittest.mock import patch

from spot_backend.app import app
from spot_backend.chat_messages import prepare_user_visible_reply
from spot_backend.chat_shortcuts import try_deterministic_chat_reply, try_deterministic_recently_played_reply
from spot_backend.config import Settings
from spot_backend.library_mutation_store import clear_session, load_last_playlist_id
from spot_backend.spotify_tools import SpotifyToolRunner, _score_track_search_candidate
from tests.recheck_helpers import FakeOllamaStream
from tests.test_r10_items import _collect_sse_events, _mock_artist_top_track_search


def _large_recently_played(count: int = 50) -> dict[str, Any]:
    items = []
    for i in range(count):
        items.append(
            {
                "played_at": f"2026-01-{1 + (i % 28):02d}T12:00:00.000Z",
                "track": {
                    "id": f"{i:022d}",
                    "uri": f"spotify:track:{i:022d}",
                    "name": f"Song {i}",
                    "album": {"name": f"Album {i}"},
                    "artists": [{"name": f"Artist {i % 5}"}],
                    "external_urls": {"spotify": f"https://open.spotify.com/track/{i:022d}"},
                    "duration_ms": 200000,
                },
            }
        )
    return {"items": items, "next": None, "cursors": {"after": "x", "before": None}}


@respx.mock
def test_r11_item1_play_artist_payload_is_album_context_plus_offset(
    data_dir, signed_in_tokens,
) -> None:
    artist_id = "aaaaaaaaaaaaaaaaaaaaaa"
    track_ids = ["bbbbbbbbbbbbbbbbbbbbbb"]
    _mock_artist_top_track_search(artist_id, "Radiohead", track_ids)
    play_calls: list[dict[str, Any]] = []

    def capture_play(request: httpx.Request) -> httpx.Response:
        play_calls.append(json.loads(request.content or b"{}"))
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        side_effect=capture_play
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
    raw = runner.run("spotify_play_artist", {"artist_name": "Radiohead"})
    runner.close()
    assert json.loads(raw).get("ok") is True
    assert len(play_calls) == 1
    body = play_calls[0]
    assert body["context_uri"].startswith("spotify:album:")
    assert body["offset"]["uri"] == f"spotify:track:{track_ids[0]}"
    assert "uris" not in body


@respx.mock
def test_r11_item1_no_uris_on_track_play_entry_points(data_dir, signed_in_tokens) -> None:
    track_id = "cccccccccccccccccccccc"
    album_id = "dddddddddddddddddddddd"
    artist_id = "aaaaaaaaaaaaaaaaaaaaaa"
    track_ids = [track_id]
    _mock_artist_top_track_search(artist_id, "Radiohead", track_ids)
    respx.get(f"https://api.spotify.com/v1/tracks/{track_id}").mock(
        return_value=httpx.Response(
            200,
            json={"id": track_id, "uri": f"spotify:track:{track_id}", "album": {"id": album_id}},
        )
    )

    def capture_play(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        assert "uris" not in body
        assert body.get("context_uri") == f"spotify:album:{album_id}"
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        side_effect=capture_play
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "context": {"uri": f"spotify:album:{album_id}"},
                "item": {"uri": f"spotify:track:{track_id}"},
            },
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    for args in (
        {"uris": [f"spotify:track:{track_id}"]},
        {"playlist_id": f"spotify:track:{track_id}"},
    ):
        runner.run("spotify_start_resume_playback", args)
        runner.run("spotify_play_playlist", args)
    outcome = try_deterministic_chat_reply("play Radiohead", runner)
    assert outcome is not None
    runner.close()


@respx.mock
def test_r11_item1_at_most_two_play_puts_and_no_transfer(
    data_dir, signed_in_tokens,
) -> None:
    artist_id = "eeeeeeeeeeeeeeeeeeeeee"
    track_ids = ["ffffffffffffffffffffff"]
    _mock_artist_top_track_search(artist_id, "Radiohead", track_ids)
    play_n = {"n": 0}
    transfer_n = {"n": 0}

    def play_handler(_request: httpx.Request) -> httpx.Response:
        play_n["n"] += 1
        return httpx.Response(204)

    def transfer_handler(_request: httpx.Request) -> httpx.Response:
        transfer_n["n"] += 1
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        side_effect=play_handler
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player(?:\?.*)?$").mock(
        side_effect=transfer_handler
    )

    def player_handler(_request: httpx.Request) -> httpx.Response:
        if play_n["n"] < 2:
            return httpx.Response(200, json={"is_playing": False, "item": None})
        return httpx.Response(
            200,
            json={
                "is_playing": True,
                "context": {"uri": "spotify:album:0000000000000000000030"},
                "item": {"uri": f"spotify:track:{track_ids[0]}"},
            },
        )

    respx.get("https://api.spotify.com/v1/me/player").mock(side_effect=player_handler)
    runner = SpotifyToolRunner(settings=Settings())
    runner.run("spotify_play_artist", {"artist_name": "Radiohead"})
    runner.close()
    assert play_n["n"] <= 2
    assert transfer_n["n"] == 0


@respx.mock
def test_r11_item1_artist_play_makes_zero_queue_calls(data_dir, signed_in_tokens) -> None:
    artist_id = "1111111111111111111111"
    track_ids = [f"{i:022d}" for i in range(4)]
    _mock_artist_top_track_search(artist_id, "Radiohead", track_ids)
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
    runner.close()
    assert queue_n["n"] == 0


@respx.mock
def test_r11_item1_two_artist_plays_leave_queue_untouched(data_dir, signed_in_tokens) -> None:
    queue_n = {"n": 0}

    def queue_handler(_request: httpx.Request) -> httpx.Response:
        queue_n["n"] += 1
        return httpx.Response(204)

    respx.post(url__regex=r"https://api\.spotify\.com/v1/me/player/queue.*").mock(
        side_effect=queue_handler
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={"is_playing": True, "context": {"uri": "spotify:album:x"}, "item": {"uri": "spotify:track:y"}},
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    for name, aid in (("Radiohead", "aaaaaaaaaaaaaaaaaaaaaa"), ("Drake", "bbbbbbbbbbbbbbbbbbbbbb")):
        _mock_artist_top_track_search(aid, name, [f"{aid[0]}{'0' * 21}"])
        runner.run("spotify_play_artist", {"artist_name": name})
    runner.close()
    assert queue_n["n"] == 0


@respx.mock
def test_r11_item2_recently_played_slim_and_gemini_shortcut(data_dir, signed_in_tokens) -> None:
    payload = _large_recently_played(50)
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player/recently-played.*").mock(
        return_value=httpx.Response(200, json=payload)
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_recently_played", {"limit": 50})
    runner.close()
    assert len(raw) <= 8100
    data = json.loads(raw)
    assert isinstance(data.get("items"), list) and len(data["items"]) >= 20
    runner2 = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_recently_played_reply(
        "what have I been listening to lately?", runner2
    )
    runner2.close()
    assert outcome is not None
    assert "Song 0" in outcome.reply
    assert "couldn't find any recent listening history" not in outcome.reply.lower()


@respx.mock
def test_r11_item2_ollama_path_recently_played_large_payload(
    data_dir, signed_in_tokens,
) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player/recently-played.*").mock(
        return_value=httpx.Response(200, json=_large_recently_played(50))
    )
    streams = [
        FakeOllamaStream(
            [json.dumps({"message": {"role": "assistant", "content": "summary"}, "done": True})]
        )
    ]

    def fake_stream(_self, _method, _url, **kwargs):
        return streams[0]

    client = TestClient(app)
    with patch("httpx.Client.stream", fake_stream):
        resp = client.post(
            "/api/chat",
            json={"message": "what have I been listening to lately?"},
        )
    assert resp.status_code == 200
    assert "couldn't find any recent listening history" not in resp.json()["reply"].lower()


@respx.mock
def test_r11_item3_create_then_private_via_stream_conversation_id(
    data_dir, signed_in_tokens,
) -> None:
    conv = "conv-r11-playlist"
    clear_session(conv)
    created_id = "pppppppppppppppppppppp"
    guessed_id = "gggggggggggggggggggggg"
    respx.post("https://api.spotify.com/v1/me/playlists").mock(
        return_value=httpx.Response(
            200,
            json={"id": created_id, "name": "Mix", "uri": f"spotify:playlist:{created_id}"},
        )
    )
    respx.get(f"https://api.spotify.com/v1/playlists/{created_id}").mock(
        return_value=httpx.Response(200, json={"id": created_id, "public": False, "name": "Mix"})
    )
    update_calls: list[str] = []

    def update_playlist(request: httpx.Request) -> httpx.Response:
        pid = request.url.path.split("/")[-1]
        update_calls.append(pid)
        return httpx.Response(200, json={"id": pid})

    respx.put(url__regex=r"https://api\.spotify\.com/v1/playlists/.*").mock(
        side_effect=update_playlist
    )

    def fake_iter_events(user_text, settings, history=None, *, conversation_id=None):
        runner = SpotifyToolRunner(settings=settings, conversation_id=conversation_id)
        if "create" in user_text.lower():
            name = "spotify_create_playlist"
            args = {"name": "Mix"}
            raw = runner.run(name, args)
            created = json.loads(raw)
            pid = created.get("id")
            if isinstance(pid, str):
                runner.note_session_playlist_id(pid)
            yield {"type": "tool_start", "name": name}
            yield {"type": "tool_done", "name": name, "preview": raw[:200]}
            yield {"type": "final", "text": "Created your private playlist Mix."}
        else:
            name = "spotify_update_playlist"
            args = {"playlist_id": guessed_id, "public": False}
            raw = runner.run(name, args)
            data = json.loads(raw)
            assert data.get("ok") is True
            yield {"type": "tool_start", "name": name}
            yield {"type": "tool_done", "name": name, "preview": raw[:200]}
            yield {"type": "final", "text": "Your playlist is private now."}
        runner.close()

    client = TestClient(app)
    with patch("spot_backend.app.iter_chat_events", fake_iter_events):
        r1 = client.post(
            "/api/chat/stream",
            json={"message": "create a playlist called Mix", "conversation_id": conv},
        )
        _ = r1.text
        r2 = client.post(
            "/api/chat/stream",
            json={
                "message": "make it private",
                "conversation_id": conv,
                "history": [
                    {"role": "user", "content": "create a playlist called Mix"},
                    {"role": "assistant", "content": "Created your private playlist Mix."},
                ],
            },
        )
        _ = r2.text
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert load_last_playlist_id(conv) == created_id
    assert created_id in update_calls
    assert guessed_id not in update_calls


@respx.mock
def test_r11_item3_guessed_playlist_id_rejected_without_fallback(
    data_dir, signed_in_tokens,
) -> None:
    guessed = "zzzzzzzzzzzzzzzzzzzzzz"
    runner = SpotifyToolRunner(settings=Settings(), conversation_id="empty-conv")
    raw = runner.run("spotify_update_playlist", {"playlist_id": guessed, "public": False})
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is False
    assert "not returned by Spotify tools" in data.get("error", "")


def test_r11_item3_create_visibility_note_once() -> None:
    assistant = "Created your playlist Mix."
    tool_json = json.dumps(
        {
            "id": "aaaaaaaaaaaaaaaaaaaaaa",
            "public": False,
            "visibility_change_requested": True,
            "visibility_warning": (
                "Spotify still reports this playlist as public after creation. "
                "You may need to set visibility manually in the Spotify app."
            ),
        }
    )
    reply = prepare_user_visible_reply(assistant, [tool_json])
    assert reply.count("still reports this playlist as public") == 1


def test_r11_item4_queue_search_prefers_original_over_karaoke() -> None:
    karaoke = {
        "name": "Karma Police (Karaoke Version)",
        "popularity": 40,
        "artists": [{"name": "Radiohead Tribute"}],
        "uri": "spotify:track:1111111111111111111111",
    }
    original = {
        "name": "Karma Police",
        "popularity": 70,
        "artists": [{"name": "Radiohead"}],
        "uri": "spotify:track:2222222222222222222222",
    }
    assert _score_track_search_candidate(
        karaoke, want_title="Karma Police", want_artist="Radiohead"
    ) < _score_track_search_candidate(
        original, want_title="Karma Police", want_artist="Radiohead"
    )


@respx.mock
def test_r11_item4_direct_queue_karma_police(data_dir, signed_in_tokens) -> None:
    original_uri = "spotify:track:2222222222222222222222"
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [
                        {
                            "name": "Karma Police (Karaoke Version)",
                            "popularity": 40,
                            "artists": [{"name": "Radiohead Tribute"}],
                            "uri": "spotify:track:1111111111111111111111",
                        },
                        {
                            "name": "Karma Police",
                            "popularity": 70,
                            "artists": [{"name": "Radiohead"}],
                            "uri": original_uri,
                        },
                    ]
                }
            },
        )
    )
    queue_calls: list[str] = []

    def capture_queue(request: httpx.Request) -> httpx.Response:
        queue_calls.append(request.url.params.get("uri", ""))
        return httpx.Response(204)

    respx.post(url__regex=r"https://api\.spotify\.com/v1/me/player/queue.*").mock(
        side_effect=capture_queue
    )
    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply(
        "queue Karma Police by Radiohead", runner
    )
    runner.close()
    assert outcome is not None
    assert outcome.tool_names() == ["spotify_add_to_queue"]
    assert queue_calls == [original_uri]
    assert "Queued Karma Police by Radiohead" in outcome.reply
