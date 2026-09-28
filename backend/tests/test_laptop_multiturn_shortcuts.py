"""Laptop repro: vague multi-turn prompts must not hit deterministic play shortcuts."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from spot_backend.app import app
from spot_backend.chat_shortcuts import try_deterministic_chat_reply
from spot_backend.config import Settings
from spot_backend.deterministic_chat import resolve_deterministic_chat_outcome
from spot_backend.play_artist import format_play_artist_reply
from spot_backend.reply_tool_trace import tool_trace_log_path
from spot_backend.spotify_tools import SpotifyToolRunner

LAPTOP_VAGUE_PROMPTS = (
    "Play something by The Weeknd",
    "play his latest single",
    "Can you play podcasts via this interface?",
    "Play one of my playlists",
)


def _collect_sse_events(stream_resp) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in stream_resp.text.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    events.append(json.loads(payload))
    return events


def test_vague_laptop_prompts_skip_deterministic_shortcuts(signed_in_tokens) -> None:
    runner = SpotifyToolRunner(settings=Settings())
    try:
        for phrase in LAPTOP_VAGUE_PROMPTS:
            assert try_deterministic_chat_reply(phrase, runner) is None
            assert resolve_deterministic_chat_outcome(phrase, runner, conversation_id="lap-1") is None
    finally:
        runner.close()


@respx.mock
def test_sse_multiturn_vague_prompts_reach_gemini_not_shortcut(
    data_dir, signed_in_tokens, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key-laptop-multiturn-01")
    gemini_calls: list[str] = []

    def fake_gemini_turn(
        user_text: str,
        settings: Settings,
        history=None,
        *,
        emit=None,
        conversation_id: str | None = None,
    ) -> str:
        gemini_calls.append(user_text)
        return f"LLM handled: {user_text[:40]}"

    client = TestClient(app)
    history: list[dict[str, str]] = []
    for phrase in LAPTOP_VAGUE_PROMPTS:
        with patch("spot_backend.gemini_llm.run_chat_turn_gemini", side_effect=fake_gemini_turn):
            resp = client.post(
                "/api/chat/stream",
                json={"message": phrase, "history": history, "conversation_id": "sse-lap"},
            )
        assert resp.status_code == 200
        events = _collect_sse_events(resp)
        tool_starts = [e for e in events if e.get("type") == "tool_start"]
        assert not tool_starts, f"Shortcut tools fired for {phrase!r}: {tool_starts}"
        final = next((e for e in events if e.get("type") == "final"), None)
        assert final
        text = str(final.get("text") or "")
        assert " on Spotify." not in text
        history.append({"role": "user", "content": phrase})
        history.append({"role": "assistant", "content": text})

    assert gemini_calls == list(LAPTOP_VAGUE_PROMPTS)


@respx.mock
def test_shortcut_play_artist_writes_trace_via_sse(data_dir, signed_in_tokens) -> None:
    from tests.test_r10_items import _mock_artist_top_track_search

    artist_id = "aaaaaaaaaaaaaaaaaaaaaa"
    track_id = "bbbbbbbbbbbbbbbbbbbbbb"
    album_id = "cccccccccccccccccccccc"
    _mock_artist_top_track_search(artist_id, "Radiohead", [track_id])
    respx.get(f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks").mock(
        return_value=httpx.Response(
            200,
            json={
                "tracks": [
                    {
                        "id": track_id,
                        "uri": f"spotify:track:{track_id}",
                        "name": "Creep",
                        "album": {"id": album_id},
                    }
                ],
                "artist_name": "Radiohead",
            },
        )
    )
    respx.get(f"https://api.spotify.com/v1/tracks/{track_id}").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": track_id,
                "uri": f"spotify:track:{track_id}",
                "album": {"id": album_id},
            },
        )
    )
    respx.get(f"https://api.spotify.com/v1/albums/{album_id}/tracks").mock(
        return_value=httpx.Response(200, json={"items": [{"uri": f"spotify:track:{track_id}"}]})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "item": {
                    "uri": f"spotify:track:{track_id}",
                    "name": "Creep",
                    "artists": [{"name": "Radiohead"}],
                },
            },
        )
    )
    client = TestClient(app)
    resp = client.post(
        "/api/chat/stream",
        json={"message": "play Radiohead", "conversation_id": "trace-shortcut"},
    )
    assert resp.status_code == 200
    path = tool_trace_log_path(data_dir)
    assert path.is_file()
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert any("spotify_play_artist" in ln for ln in lines)
    final = next(
        e for e in _collect_sse_events(resp) if e.get("type") == "final"
    )
    assert "Creep" in str(final.get("text") or "")
    assert "Playing Radiohead on Spotify" not in str(final.get("text") or "")


def test_play_artist_shortcut_reply_requires_verified_playback() -> None:
    raw_ok_unverified = json.dumps(
        {
            "ok": True,
            "artist_name": "Radiohead",
            "playback_verified": False,
            "playback": {"ok": True, "playback_verified": False},
        }
    )
    reply = format_play_artist_reply("Radiohead", raw_ok_unverified)
    assert "could not confirm" in reply.lower()
    assert "Playing Radiohead on Spotify" not in reply

    raw_verified = json.dumps(
        {
            "ok": True,
            "artist_name": "Radiohead",
            "playback_verified": True,
            "player_after": {
                "item": {
                    "name": "Creep",
                    "artists": [{"name": "Radiohead"}],
                }
            },
        }
    )
    reply2 = format_play_artist_reply("Radiohead", raw_verified)
    assert "Creep" in reply2
    assert "Radiohead" in reply2
    assert " on Spotify." not in reply2
