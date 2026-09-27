from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from fastapi.testclient import TestClient

from spot_backend.app import app
from spot_backend.now_playing import reset_now_playing_cache_for_tests


def _track(tid: str, name: str) -> dict[str, Any]:
    return {
        "id": tid,
        "name": name,
        "duration_ms": 200_000,
        "artists": [{"name": "Artist"}],
        "album": {"name": "Album", "images": [{"url": "https://i.test/a.jpg", "width": 64}]},
    }


@pytest.fixture(autouse=True)
def _clear_np_cache() -> None:
    reset_now_playing_cache_for_tests()


@respx.mock
def test_player_next_returns_updated_track(data_dir, signed_in_tokens) -> None:
    calls = {"n": 0}

    def player_handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"is_playing": True, "item": _track("aaaa", "First")})
        return httpx.Response(200, json={"is_playing": True, "item": _track("bbbb", "Second")})

    respx.get("https://api.spotify.com/v1/me/player").mock(side_effect=player_handler)
    respx.post("https://api.spotify.com/v1/me/player/next").mock(return_value=httpx.Response(204))
    respx.get("https://api.spotify.com/v1/me/player/queue").mock(
        return_value=httpx.Response(200, json={"queue": []})
    )
    client = TestClient(app)
    body = client.post("/api/player/next").json()
    assert body["track"]["name"] == "Second"
