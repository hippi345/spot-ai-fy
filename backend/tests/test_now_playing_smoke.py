"""Smoke flow for docs/SMOKE_TEST.md steps 1–7 via /api/chat + /api/now-playing."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from unittest.mock import patch

from spot_backend.app import app
from spot_backend.now_playing import reset_now_playing_cache_for_tests
from tests.recheck_helpers import (
    gemini_candidates_payload,
    gemini_stop_candidate,
    make_gemini_post_recorder,
)


ARTIST_ID = "jjjjjjjjjjjjjjjjjjjjjj"
TRACK_A = "aaaaaaaaaaaaaaaaaaaaa1"
TRACK_B = "aaaaaaaaaaaaaaaaaaaaa2"
GRAVITY_ID = "ggggggggggggggggggggg1"


def _track(
    tid: str,
    name: str,
    album_id: str = "albumcontinuum00000001",
) -> dict[str, Any]:
    return {
        "id": tid,
        "uri": f"spotify:track:{tid}",
        "name": name,
        "duration_ms": 210_000,
        "artists": [{"name": "John Mayer"}],
        "album": {
            "id": album_id,
            "name": "Continuum",
            "images": [
                {"url": "https://i.test/l.jpg", "width": 300, "height": 300},
                {"url": "https://i.test/s.jpg", "width": 64, "height": 64},
            ],
        },
    }


class SpotifyPlaybackState:
    def __init__(self) -> None:
        self.tracks = [
            _track(TRACK_A, "Slow Dancing in a Burning Room"),
            _track(TRACK_B, "Waiting on the World to Change"),
        ]
        self.gravity = _track(GRAVITY_ID, "Gravity")
        self.index = 0
        self.is_playing = False
        self.queue: list[dict[str, Any]] = []
        self.playlists = [
            {"id": "plist000000000000000001", "name": "Evening Acoustic"},
            {"id": "plist000000000000000002", "name": "Road Trip Mix"},
        ]
        self._started = False

    def current(self) -> dict[str, Any]:
        return self.tracks[self.index]

    def player_payload(self) -> dict[str, Any] | None:
        if not self._started:
            return None
        return {
            "is_playing": self.is_playing,
            "progress_ms": 5000,
            "item": self.current(),
            "device": {"id": "dev-smoke", "name": "Smoke Speaker", "type": "Speaker"},
        }

    def start_playback(self) -> None:
        self._started = True
        self.is_playing = True

    def skip_next(self) -> None:
        self.index = min(self.index + 1, len(self.tracks) - 1)
        self.is_playing = True

    def pause(self) -> None:
        self.is_playing = False

    def resume(self) -> None:
        self.is_playing = True

    def add_gravity_to_queue(self) -> None:
        self.queue.append(self.gravity)


def _install_stateful_spotify(state: SpotifyPlaybackState) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/search"):
            q = request.url.params.get("q", "")
            if request.url.params.get("type") == "track" or "Gravity" in q:
                items = [state.gravity] if "Gravity" in q else state.tracks
                return httpx.Response(200, json={"tracks": {"items": items}})
            return httpx.Response(
                200,
                json={"artists": {"items": [{"id": ARTIST_ID, "name": "John Mayer"}]}},
            )
        if f"/artists/{ARTIST_ID}" in path and path.endswith(f"/artists/{ARTIST_ID}"):
            return httpx.Response(200, json={"id": ARTIST_ID, "name": "John Mayer"})
        if f"/artists/{ARTIST_ID}/top-tracks" in path:
            return httpx.Response(
                200,
                json={"tracks": state.tracks},
            )
        for tr in state.tracks + [state.gravity]:
            if path.endswith(f"/tracks/{tr['id']}"):
                return httpx.Response(200, json=tr)
            if path.endswith(f"/albums/{tr['album']['id']}/tracks"):
                return httpx.Response(200, json={"items": [{"uri": tr["uri"]}]})
            if path.endswith(f"/albums/{tr['album']['id']}"):
                return httpx.Response(200, json={"id": tr["album"]["id"], "uri": f"spotify:album:{tr['album']['id']}"})

        if path.endswith("/me/player/devices"):
            return httpx.Response(
                200,
                json={
                    "devices": [
                        {
                            "id": "dev-smoke",
                            "is_active": True,
                            "is_restricted": False,
                            "name": "Smoke Speaker",
                        }
                    ]
                },
            )
        if path.endswith("/me/player/play") or path.endswith("/me/player"):
            if request.method == "PUT" and "play" in path:
                state.start_playback()
                return httpx.Response(204)
        if path.endswith("/me/player/pause"):
            state.pause()
            return httpx.Response(204)
        if path.endswith("/me/player/next"):
            state.skip_next()
            return httpx.Response(204)
        if path.endswith("/me/player/queue") and request.method == "POST":
            state.add_gravity_to_queue()
            return httpx.Response(204)
        if path.endswith("/me/player/queue") and request.method == "GET":
            return httpx.Response(
                200,
                json={"queue": [{"track": t} for t in state.queue]},
            )
        if path.endswith("/me/player"):
            payload = state.player_payload()
            if payload is None:
                return httpx.Response(204)
            return httpx.Response(200, json=payload)
        if path.endswith("/me/playlists"):
            return httpx.Response(200, json={"items": state.playlists})
        return httpx.Response(200, json={})

    respx.route(url__regex=r"https://api\.spotify\.com/.*").mock(side_effect=handler)


_GEMINI_TOOLS: dict[str, str] = {
    "what's playing?": "spotify_playback_state",
    "skip": "spotify_skip_next",
    "pause": "spotify_pause",
    "resume": "spotify_start_resume_playback",
    "what are my playlists?": "spotify_user_playlists",
}


def _gemini_handler(state: SpotifyPlaybackState, user_text: str):
    key = user_text.strip().lower()
    tool = _GEMINI_TOOLS.get(key)

    def handler(_body: dict[str, Any], n: int, req: httpx.Request) -> httpx.Response:
        if n == 1 and tool:
            part = {"functionCall": {"name": tool, "args": {}}}
        elif n == 1:
            part = {"text": "OK."}
        else:
            if tool == "spotify_playback_state":
                part = {"text": f"You're listening to {state.current()['name']} by John Mayer."}
            elif tool == "spotify_user_playlists":
                names = ", ".join(p["name"] for p in state.playlists)
                part = {"text": f"Your playlists include: {names}."}
            else:
                part = {"text": "Done."}
        payload = gemini_candidates_payload(gemini_stop_candidate(part))
        return httpx.Response(200, json=payload, request=req)

    return handler


@pytest.fixture(autouse=True)
def _clear_np_cache() -> None:
    reset_now_playing_cache_for_tests()


@respx.mock
def test_smoke_steps_1_through_7(data_dir, signed_in_tokens, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    state = SpotifyPlaybackState()
    _install_stateful_spotify(state)
    client = TestClient(app)

    def chat(message: str) -> dict[str, Any]:
        key = message.strip().lower()
        if key in _GEMINI_TOOLS:
            handler = _gemini_handler(state, message)
            _, fake_post = make_gemini_post_recorder(handler)
            with patch("httpx.Client.post", fake_post):
                r = client.post("/api/chat", json={"message": message})
        else:
            r = client.post("/api/chat", json={"message": message})
        assert r.status_code == 200
        return r.json()

    def np() -> dict[str, Any]:
        r = client.get("/api/now-playing")
        assert r.status_code == 200
        return r.json()

    # 1 — Play John Mayer
    r1 = chat("Play John Mayer")
    assert "playing" in r1["reply"].lower()
    np1 = np()
    assert np1["is_playing"] is True
    assert "John Mayer" in " ".join(np1["track"]["artists"])
    current_name = np1["track"]["name"]

    # 2 — What's playing?
    r2 = chat("What's playing?")
    assert current_name.lower() in r2["reply"].lower()
    assert np()["track"]["name"] == current_name

    # 3 — Skip
    chat("Skip")
    np3 = np()
    assert np3["track"]["name"] != current_name
    current_name = np3["track"]["name"]

    # 4 — Pause
    chat("Pause")
    assert np()["is_playing"] is False

    # 5 — Resume
    chat("Resume")
    assert np()["is_playing"] is True

    # 6 — Playlists
    r6 = chat("What are my playlists?")
    assert "Evening Acoustic" in r6["reply"]
    assert "Road Trip Mix" in r6["reply"]

    # 7 — Queue Gravity
    r7 = chat("Queue Gravity by John Mayer")
    assert "gravity" in r7["reply"].lower()
    queue_names = [q["name"] for q in np()["queue"]]
    assert "Gravity" in queue_names
