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
from tests.smoke_spotify_state import SpotifyPlaybackState, install_stateful_spotify_mock


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
    install_stateful_spotify_mock(state)
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
