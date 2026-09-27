from __future__ import annotations

import time
from typing import Any

import httpx
import pytest
import respx

from fastapi.testclient import TestClient

from spot_backend.app import app
from spot_backend.now_playing import reset_now_playing_cache_for_tests
from spot_backend.token_store import DeviceSelection, save_device


def _track_payload(
    tid: str = "aaaaaaaaaaaaaaaaaaaaaa",
    name: str = "Sample River Song",
    artist: str = "John Mayer",
) -> dict[str, Any]:
    return {
        "id": tid,
        "name": name,
        "duration_ms": 200_000,
        "artists": [{"name": artist}],
        "album": {
            "name": "Continuum",
            "images": [
                {"url": "https://i.test/large.jpg", "width": 640, "height": 640},
                {"url": "https://i.test/small.jpg", "width": 64, "height": 64},
            ],
        },
    }


def _player_json(**overrides: Any) -> dict[str, Any]:
    base = {
        "is_playing": True,
        "progress_ms": 12_000,
        "item": _track_payload(),
        "device": {"name": "Test Speaker", "type": "Speaker"},
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _clear_np_cache() -> None:
    reset_now_playing_cache_for_tests()


def test_now_playing_signed_out(data_dir) -> None:
    client = TestClient(app)
    r = client.get("/api/now-playing")
    assert r.status_code == 401
    assert r.json() == {"signed_in": False}


@respx.mock
def test_now_playing_playing(data_dir, signed_in_tokens) -> None:
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(200, json=_player_json())
    )
    respx.get("https://api.spotify.com/v1/me/player/queue").mock(
        return_value=httpx.Response(
            200,
            json={
                "queue": [
                    {"track": _track_payload("bbbbbbbbbbbbbbbbbbbbbb", "Gravity", "John Mayer")},
                ]
            },
        )
    )
    client = TestClient(app)
    body = client.get("/api/now-playing").json()
    assert body["is_playing"] is True
    assert body["track"]["name"] == "Sample River Song"
    assert body["track"]["art_url"] == "https://i.test/large.jpg"
    assert body["queue"][0]["name"] == "Gravity"
    assert body["queue"][0]["art_url"] == "https://i.test/small.jpg"
    assert "fetched_at" in body


@respx.mock
def test_now_playing_idle(data_dir, signed_in_tokens) -> None:
    respx.get("https://api.spotify.com/v1/me/player").mock(return_value=httpx.Response(204))
    respx.get("https://api.spotify.com/v1/me/player/queue").mock(
        return_value=httpx.Response(200, json={"queue": []})
    )
    client = TestClient(app)
    body = client.get("/api/now-playing").json()
    assert body == {
        "is_playing": False,
        "track": None,
        "progress_ms": 0,
        "device": None,
        "queue": [],
        "fetched_at": body["fetched_at"],
    }


@respx.mock
def test_now_playing_429_returns_stale(data_dir, signed_in_tokens) -> None:
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(200, json=_player_json())
    )
    respx.get("https://api.spotify.com/v1/me/player/queue").mock(
        return_value=httpx.Response(200, json={"queue": []})
    )
    client = TestClient(app)
    first = client.get("/api/now-playing").json()
    assert first.get("stale") is not True

    respx.get("https://api.spotify.com/v1/me/player").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "30"}, json={"error": {"status": 429}}),
            httpx.Response(429, headers={"Retry-After": "30"}, json={"error": {"status": 429}}),
        ]
    )
    stale = client.get("/api/now-playing").json()
    assert stale.get("stale") is True
    assert stale["track"]["name"] == first["track"]["name"]


@respx.mock
def test_now_playing_queue_trimmed_to_ten(data_dir, signed_in_tokens) -> None:
    queue_items = [
        {"track": _track_payload(f"{i:022d}", f"Track {i}", "Artist")}
        for i in range(15)
    ]
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(200, json=_player_json())
    )
    respx.get("https://api.spotify.com/v1/me/player/queue").mock(
        return_value=httpx.Response(200, json={"queue": queue_items})
    )
    client = TestClient(app)
    body = client.get("/api/now-playing").json()
    assert len(body["queue"]) == 10
    assert body["queue"][0]["name"] == "Track 0"
    assert body["queue"][-1]["name"] == "Track 9"


@respx.mock
def test_player_toggle_pause(data_dir, signed_in_tokens) -> None:
    save_device(data_dir / "device.json", DeviceSelection(device_id="dev-abc"))
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(200, json=_player_json(is_playing=True))
    )
    pause = respx.put("https://api.spotify.com/v1/me/player/pause").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player/queue").mock(
        return_value=httpx.Response(200, json={"queue": []})
    )
    client = TestClient(app)
    after_pause = client.post("/api/player/toggle").json()
    assert pause.called
    assert pause.calls.last.request.url.params.get("device_id") == "dev-abc"
    assert "track" in after_pause


@respx.mock
def test_player_next_and_previous(data_dir, signed_in_tokens) -> None:
    next_route = respx.post("https://api.spotify.com/v1/me/player/next").mock(
        return_value=httpx.Response(204)
    )
    prev_route = respx.post("https://api.spotify.com/v1/me/player/previous").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(200, json=_player_json())
    )
    respx.get("https://api.spotify.com/v1/me/player/queue").mock(
        return_value=httpx.Response(200, json={"queue": []})
    )
    client = TestClient(app)
    client.post("/api/player/next")
    client.post("/api/player/previous")
    assert next_route.called
    assert prev_route.called
