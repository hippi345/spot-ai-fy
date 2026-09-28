"""Regression tests for laptop multi-turn chat issues (mocked Spotify + Gemini)."""

from __future__ import annotations

import json
from datetime import date
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from spot_backend.action_claim_guard import reply_claims_unbacked_action
from spot_backend.app import app
from spot_backend.config import Settings
from spot_backend.gemini_history import validate_gemini_contents
from spot_backend.gemini_llm import run_chat_turn_gemini
from spot_backend.play_artist import format_play_artist_reply
from spot_backend.spotify_tools import SpotifyToolRunner, pick_latest_album_release
from tests.test_r10_items import _mock_artist_top_track_search


def _collect_sse_events(stream_resp) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in stream_resp.text.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    events.append(json.loads(payload))
    return events


def test_play_artist_reply_accepts_featured_artist_credit() -> None:
    raw = json.dumps(
        {
            "ok": True,
            "artist_name": "The Weeknd",
            "playback_verified": True,
            "player_after": {
                "item": {
                    "name": "Or Nah",
                    "artists": [
                        {"name": "DJ Khaled"},
                        {"name": "The Weeknd"},
                    ],
                }
            },
        }
    )
    reply = format_play_artist_reply("The Weeknd", raw)
    assert "Weeknd" in reply
    assert "wasn't able" not in reply.lower()


def test_play_artist_ok_without_verify_uses_soft_wording() -> None:
    raw = json.dumps(
        {
            "ok": True,
            "artist_name": "The Weeknd",
            "playback_verified": False,
            "playback": {"ok": True, "playback_verified": False},
        }
    )
    reply = format_play_artist_reply("The Weeknd", raw)
    assert "could not confirm" in reply.lower()
    assert "wasn't able to run the spotify action" not in reply


def test_latest_release_skips_future_dates() -> None:
    today_key = (2026, 9, 28)
    items = [
        {"name": "Future Album", "release_date": "2027-01-31", "album_type": "album"},
        {"name": "Out Now Single", "release_date": "2026-06-01", "album_type": "single"},
    ]
    latest = pick_latest_album_release(items, on_or_before=today_key)
    assert latest is not None
    assert latest["name"] == "Out Now Single"


@respx.mock
def test_artist_latest_album_ignores_future_hurry_up(data_dir, signed_in_tokens) -> None:
    artist_id = "0aHjOrDlHWSXDSF1DXJuUY"
    respx.get(url__regex=rf"https://api\.spotify\.com/v1/artists/{artist_id}/albums.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "name": "Hurry Up Tomorrow",
                        "release_date": "2027-01-31",
                        "album_type": "album",
                    },
                    {
                        "name": "Live Single",
                        "release_date": "2026-05-01",
                        "album_type": "single",
                    },
                ],
                "total": 2,
            },
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    try:
        raw = runner.run(
            "spotify_artist_latest_album",
            {"artist_id": artist_id, "prefer": "release"},
        )
        data = json.loads(raw)
        latest = data.get("latest_release") or {}
        assert latest.get("name") == "Live Single"
        assert data.get("reference_date_utc") == date.today().isoformat()
    finally:
        runner.close()


def test_playing_claim_without_play_tool_flagged_for_vague_playlist() -> None:
    assert reply_claims_unbacked_action(
        "Playing 'Jamz — My Artists — 2026-09-25'. Would you like a different one?",
        {"spotify_user_playlists"},
        user_text="Play one of my playlists",
        turn_tool_calls=[("spotify_user_playlists", '{"ok": true}')],
    )


@respx.mock
def test_sse_utf8_em_dash_in_final_text(data_dir, signed_in_tokens, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key-utf8-emdash")

    def fake_gemini_turn(
        user_text: str,
        settings: Settings,
        history=None,
        *,
        emit=None,
        conversation_id: str | None = None,
    ) -> str:
        return "Playing Jamz — My Artists — 2026-09-25."

    client = TestClient(app)
    with patch("spot_backend.gemini_llm.run_chat_turn_gemini", side_effect=fake_gemini_turn):
        resp = client.post(
            "/api/chat/stream",
            json={"message": "play one of my playlists", "conversation_id": "utf8-em"},
        )
    assert resp.status_code == 200
    assert "charset=utf-8" in (resp.headers.get("content-type") or "").lower()
    assert "Jamz — My Artists" in resp.text
    assert "\u00e2\u0080\u0094" not in resp.text


@respx.mock
def test_gemini_seven_turn_history_passes_validation(data_dir, signed_in_tokens) -> None:
    """Simulate 7 SSE turns with tool calls; each Gemini body must validate."""
    settings = Settings(gemini_api_key="gemini-test-key-seven-turn", agent_max_steps=6)
    bodies_seen: list[dict[str, Any]] = []
    call_idx = {"n": 0}

    prompts = [
        "Play something by The Weeknd",
        "play his latest single",
        "Can you play podcasts via this interface?",
        "Play one of my playlists",
        "How many albums does Drake have?",
        "Who are my top artists this month?",
        "What's playing?",
    ]

    def gemini_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        bodies_seen.append(body)
        contents = body.get("contents") or []
        errors = validate_gemini_contents(contents)
        assert not errors, f"Invalid Gemini history: {errors}; contents={json.dumps(contents)[:1200]}"
        n = call_idx["n"]
        call_idx["n"] += 1
        if n % 2 == 0:
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {
                                "parts": [{"text": f"Answer for turn {n // 2 + 1}."}],
                            },
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [
                                {
                                    "thoughtSignature": "sig_test_abc123",
                                    "functionCall": {
                                        "name": "spotify_me",
                                        "args": {},
                                    },
                                }
                            ],
                        },
                    }
                ]
            },
        )

    respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/.*").mock(
        side_effect=gemini_handler
    )

    history: list[dict[str, str]] = []
    with patch(
        "spot_backend.gemini_llm.gemini_deterministic_shortcut_reply",
        return_value=None,
    ), patch(
        "spot_backend.gemini_llm.should_send_gemini_tool_nudge",
        return_value=False,
    ):
        for phrase in prompts:
            run_chat_turn_gemini(phrase, settings, history=history)
            history.append({"role": "user", "content": phrase})
            history.append({"role": "assistant", "content": f"Handled {phrase[:20]}"})

    assert len(bodies_seen) >= 7


@respx.mock
def test_podcast_capability_turn_has_no_tools(data_dir, signed_in_tokens) -> None:
    settings = Settings(gemini_api_key="gemini-test-key-podcast-cap", agent_max_steps=2)
    seen_tools: list[bool] = []

    def gemini_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen_tools.append(bool(body.get("tools")))
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [{"text": "No — podcast playback is not supported here."}],
                        },
                    }
                ]
            },
        )

    respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/.*").mock(
        side_effect=gemini_handler
    )
    history = [
        {"role": "user", "content": "play his latest single"},
        {"role": "assistant", "content": "Playing the latest single."},
    ]
    with patch(
        "spot_backend.gemini_llm.gemini_deterministic_shortcut_reply",
        return_value=None,
    ):
        reply = run_chat_turn_gemini(
            "Can you play podcasts via this interface?",
            settings,
            history=history,
        )
    assert "podcast" in reply.lower() or "no" in reply.lower()
    assert seen_tools and seen_tools[0] is False


@respx.mock
def test_play_artist_featured_verification_mock(data_dir, signed_in_tokens) -> None:
    artist_id = "aaaaaaaaaaaaaaaaaaaaaa"
    track_id = "bbbbbbbbbbbbbbbbbbbbbb"
    album_id = "cccccccccccccccccccccc"
    _mock_artist_top_track_search(artist_id, "The Weeknd", [track_id])
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
        return_value=httpx.Response(
            200,
            json={"items": [{"uri": f"spotify:track:{track_id}"}]},
        )
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )

    poll = {"count": 0}

    def player_handler(_request: httpx.Request) -> httpx.Response:
        poll["count"] += 1
        if poll["count"] < 2:
            return httpx.Response(204)
        return httpx.Response(
            200,
            json={
                "is_playing": True,
                "item": {
                    "uri": f"spotify:track:{track_id}",
                    "name": "Or Nah",
                    "artists": [{"name": "DJ Khaled"}, {"name": "The Weeknd"}],
                },
            },
        )

    respx.get("https://api.spotify.com/v1/me/player").mock(side_effect=player_handler)
    runner = SpotifyToolRunner(settings=Settings())
    try:
        raw = runner.run("spotify_play_artist", {"artist_name": "The Weeknd"})
        data = json.loads(raw)
        assert data.get("ok") is True
        assert data.get("playback_verified") is True
    finally:
        runner.close()
