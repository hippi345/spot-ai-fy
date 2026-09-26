"""Round-10 laptop retest — test_r10_itemN_* per PR requirements."""

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
from spot_backend.chat_shortcuts import try_deterministic_chat_reply
from spot_backend.config import Settings
from spot_backend.library_mutation_store import clear_session
from spot_backend.spotify_tools import SpotifyToolRunner
from tests.recheck_helpers import FakeOllamaStream, mock_artist_name_search


def _mock_artist_top_track_search(artist_id: str, name: str, track_ids: list[str]) -> None:
    respx.get(f"https://api.spotify.com/v1/artists/{artist_id}").mock(
        return_value=httpx.Response(200, json={"id": artist_id, "name": name})
    )
    items = [
        {
            "uri": f"spotify:track:{tid}",
            "name": f"Track {i}",
            "id": tid,
            "album": {"id": f"{30 + i:022d}"},
        }
        for i, tid in enumerate(track_ids)
    ]
    for tid in track_ids:
        respx.get(f"https://api.spotify.com/v1/tracks/{tid}").mock(
            return_value=httpx.Response(200, json={"id": tid})
        )

    def search_handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("type") == "track":
            return httpx.Response(200, json={"tracks": {"items": items}})
        return httpx.Response(
            200,
            json={"artists": {"items": [{"id": artist_id, "name": name}]}},
        )

    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(side_effect=search_handler)


def _collect_sse_events(stream_resp) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in stream_resp.text.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    events.append(json.loads(payload))
    return events


@respx.mock
@pytest.mark.parametrize(
    ("phrase", "artist_id", "artist_name"),
    (
        ("play Radiohead", "aaaaaaaaaaaaaaaaaaaaaa", "Radiohead"),
        ("play Taylor Swift", "06HL4z0CvFAxyc27GXpf02", "Taylor Swift"),
    ),
)
def test_r10_item1_play_artist_uses_top_track_uris_not_playlist(
    data_dir,
    signed_in_tokens,
    phrase: str,
    artist_id: str,
    artist_name: str,
) -> None:
    track_ids = [f"t{i:022d}" for i in range(3)]
    _mock_artist_top_track_search(artist_id, artist_name, track_ids)
    play_calls: list[bytes] = []

    def capture_play(request: httpx.Request) -> httpx.Response:
        play_calls.append(request.content or b"")
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        side_effect=capture_play
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "item": {"uri": f"spotify:track:{track_ids[0]}"},
            },
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_chat_reply(phrase, runner)
    runner.close()
    assert outcome is not None
    assert outcome.tool_names() == ["spotify_play_artist"]
    assert "spotify_play_playlist" not in outcome.tool_names()
    assert play_calls
    body = json.loads(play_calls[0].decode() or "{}")
    assert body.get("context_uri", "").startswith("spotify:album:")
    assert body.get("offset", {}).get("uri") == f"spotify:track:{track_ids[0]}"
    assert not body.get("uris")


@respx.mock
def test_r10_item1_play_artist_retries_when_player_has_no_item(
    data_dir, signed_in_tokens,
) -> None:
    artist_id = "bbbbbbbbbbbbbbbbbbbbbb"
    track_ids = ["cccccccccccccccccccccc"]
    _mock_artist_top_track_search(artist_id, "Radiohead", track_ids)
    play_n = {"n": 0}

    def play_handler(_request: httpx.Request) -> httpx.Response:
        play_n["n"] += 1
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        side_effect=play_handler
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
    raw = runner.run("spotify_play_artist", {"artist_name": "Radiohead"})
    runner.close()
    data = json.loads(raw)
    assert play_n["n"] >= 2
    assert data.get("ok") is True


@respx.mock
def test_r10_item1_friendly_failure_message_has_no_internal_fields(
    data_dir, signed_in_tokens,
) -> None:
    pid = "pppppppppppppppppppppp"
    respx.get(f"https://api.spotify.com/v1/playlists/{pid}").mock(
        return_value=httpx.Response(200, json={"id": pid, "name": "Mix"})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(403, json={"error": {"message": "Restriction violated"}})
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(200, json={"is_playing": False})
    )
    runner = SpotifyToolRunner(settings=Settings())
    runner._session_known_ids.add(pid)
    raw = runner.run(
        "spotify_play_playlist",
        {"playlist_id": pid, "repeat": "context"},
    )
    runner.close()
    data = json.loads(raw)
    err = str(data.get("error") or "")
    assert "playback.error" not in err
    assert "playback.detail" not in err
    assert "couldn't start playback" in err.lower()


@respx.mock
def test_r10_item2_like_then_undo_via_chat_api_same_conversation(
    data_dir, signed_in_tokens,
) -> None:
    track_id = "ffffffffffffffffffffff"
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(200, json={"item": {"id": track_id, "name": "Hit"}})
    )
    respx.get(f"https://api.spotify.com/v1/tracks/{track_id}").mock(
        return_value=httpx.Response(200, json={"id": track_id})
    )
    respx.put("https://api.spotify.com/v1/me/library").mock(return_value=httpx.Response(200))
    delete_calls: list[str] = []

    def capture_delete(request: httpx.Request) -> httpx.Response:
        delete_calls.append(request.url.params.get("uris", ""))
        return httpx.Response(200)

    respx.delete(url__regex=r"https://api\.spotify\.com/v1/me/library.*").mock(
        side_effect=capture_delete
    )
    streams = [
        FakeOllamaStream(
            [json.dumps({"message": {"role": "assistant", "content": "ok"}, "done": True})]
        )
    ]

    def fake_stream(_self, _method, _url, **kwargs):
        return streams[0]

    client = TestClient(app)
    conv = "conv-like-undo-a"
    with patch("httpx.Client.stream", fake_stream):
        r1 = client.post(
            "/api/chat",
            json={"message": "like this", "conversation_id": conv},
        )
    assert r1.status_code == 200
    with patch("httpx.Client.stream", fake_stream):
        r2 = client.post(
            "/api/chat",
            json={
                "message": "undo that",
                "conversation_id": conv,
                "history": [
                    {"role": "user", "content": "like this"},
                    {"role": "assistant", "content": r1.json()["reply"]},
                ],
            },
        )
    assert r2.status_code == 200
    assert delete_calls
    assert f"spotify:track:{track_id}" in delete_calls[-1]
    assert "that" not in delete_calls[-1]


@respx.mock
def test_r10_item2_undo_isolated_between_conversations(
    data_dir, signed_in_tokens,
) -> None:
    track_id = "ababababababababababab"
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(200, json={"item": {"id": track_id, "name": "One"}})
    )
    respx.get(f"https://api.spotify.com/v1/tracks/{track_id}").mock(
        return_value=httpx.Response(200, json={"id": track_id})
    )
    respx.put("https://api.spotify.com/v1/me/library").mock(return_value=httpx.Response(200))
    delete_route = respx.delete(url__regex=r"https://api\.spotify\.com/v1/me/library.*").mock(
        return_value=httpx.Response(200)
    )
    streams = [
        FakeOllamaStream(
            [json.dumps({"message": {"role": "assistant", "content": "ok"}, "done": True})]
        )
    ]

    def fake_stream(_self, _method, _url, **kwargs):
        return streams[0]

    client = TestClient(app)
    clear_session("conv-a")
    clear_session("conv-b")
    with patch("httpx.Client.stream", fake_stream):
        client.post("/api/chat", json={"message": "like this", "conversation_id": "conv-a"})
    delete_route.calls.clear()
    with patch("httpx.Client.stream", fake_stream):
        r = client.post(
            "/api/chat",
            json={"message": "undo that", "conversation_id": "conv-b"},
        )
    assert r.status_code == 200
    assert "nothing to undo" in r.json()["reply"].lower()
    assert delete_route.call_count == 0


@respx.mock
def test_r10_item3_shortcut_tools_appear_in_sse_trace(
    data_dir, signed_in_tokens,
) -> None:
    artist_id = "gggggggggggggggggggggg"
    _mock_artist_top_track_search(artist_id, "Radiohead", ["hhhhhhhhhhhhhhhhhhhhhh"])
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={"is_playing": True, "item": {"uri": "spotify:track:hhhhhhhhhhhhhhhhhhhhhh"}},
        )
    )
    streams = [
        FakeOllamaStream(
            [json.dumps({"message": {"role": "assistant", "content": "x"}, "done": True})]
        )
    ]

    def fake_stream(_self, _method, _url, **kwargs):
        return streams[0]

    client = TestClient(app)
    with patch("httpx.Client.stream", fake_stream):
        resp = client.post("/api/chat/stream", json={"message": "play Radiohead"})
    events = _collect_sse_events(resp)
    tool_starts = [e for e in events if e.get("type") == "tool_start"]
    assert len(tool_starts) == 1
    assert tool_starts[0].get("name") == "spotify_play_artist"
    final = next(e for e in events if e.get("type") == "final")
    assert final.get("text")


@respx.mock
def test_r10_item4_ollama_chat_path_calls_recently_played_not_top_tracks(
    data_dir, signed_in_tokens,
) -> None:
    recent = respx.get(
        url__regex=r"https://api\.spotify\.com/v1/me/player/recently-played.*"
    ).mock(
        return_value=httpx.Response(
            200,
            json={"items": [{"track": {"name": "A Song", "artists": [{"name": "Band"}]}}]},
        )
    )
    top = respx.get(url__regex=r"https://api\.spotify\.com/v1/me/top/tracks.*").mock(
        return_value=httpx.Response(200, json={"items": []})
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
        client.post(
            "/api/chat",
            json={"message": "what have I been listening to lately?"},
        )
    assert recent.called
    assert top.call_count == 0


def test_r10_item5_make_private_visibility_note_once() -> None:
    assistant = "I've updated your playlist to be private."
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
    assert "I asked Spotify to make it private" in reply


@respx.mock
def test_r10_item6_undo_nothing_honest_wording_via_chat_api(
    data_dir, signed_in_tokens,
) -> None:
    streams = [
        FakeOllamaStream(
            [json.dumps({"message": {"role": "assistant", "content": "x"}, "done": True})]
        )
    ]

    def fake_stream(_self, _method, _url, **kwargs):
        return streams[0]

    client = TestClient(app)
    clear_session("fresh-conv")
    with patch("httpx.Client.stream", fake_stream):
        r = client.post(
            "/api/chat",
            json={"message": "undo that", "conversation_id": "fresh-conv"},
        )
    assert r.status_code == 200
    assert "nothing to undo" in r.json()["reply"].lower()
