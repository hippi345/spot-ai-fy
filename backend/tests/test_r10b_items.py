"""Round-10b: conversation_id isolation for cross-chat undo."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from spot_backend.app import app
from spot_backend.library_mutation_store import clear_session
from tests.recheck_helpers import FakeOllamaStream


@respx.mock
def test_r10b_item2_undo_in_chat_b_does_not_touch_chat_a_then_a_unsaves(
    data_dir, signed_in_tokens,
) -> None:
    track_a = "aaaaaaaaaaaaaaaaaaaaaa"
    track_b = "bbbbbbbbbbbbbbbbbbbbbb"
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(200, json={"item": {"id": track_a, "name": "A"}})
    )
    respx.get(f"https://api.spotify.com/v1/tracks/{track_a}").mock(
        return_value=httpx.Response(200, json={"id": track_a})
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
    conv_a = "chat-a-uuid"
    conv_b = "chat-b-uuid"
    clear_session(conv_a)
    clear_session(conv_b)

    with patch("httpx.Client.stream", fake_stream):
        r_like = client.post(
            "/api/chat",
            json={"message": "like this", "conversation_id": conv_a},
        )
    assert r_like.status_code == 200

    delete_calls.clear()
    with patch("httpx.Client.stream", fake_stream):
        r_undo_b = client.post(
            "/api/chat",
            json={"message": "undo that", "conversation_id": conv_b},
        )
    assert r_undo_b.status_code == 200
    assert "nothing to undo" in r_undo_b.json()["reply"].lower()
    assert delete_calls == []

    with patch("httpx.Client.stream", fake_stream):
        r_undo_a = client.post(
            "/api/chat",
            json={"message": "undo that", "conversation_id": conv_a},
        )
    assert r_undo_a.status_code == 200
    assert delete_calls
    assert f"spotify:track:{track_a}" in delete_calls[-1]
    assert track_b not in delete_calls[-1]


@respx.mock
def test_r10b_item2_requests_without_conversation_id_do_not_share_undo_state(
    data_dir, signed_in_tokens,
) -> None:
    track_id = "cccccccccccccccccccccc"
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(200, json={"item": {"id": track_id, "name": "T"}})
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
    with patch("httpx.Client.stream", fake_stream):
        client.post("/api/chat", json={"message": "like this"})
    delete_route.calls.clear()
    with patch("httpx.Client.stream", fake_stream):
        r = client.post("/api/chat", json={"message": "undo that"})
    assert r.status_code == 200
    assert "nothing to undo" in r.json()["reply"].lower()
    assert delete_route.call_count == 0
