"""Unit tests for action-claim guard read-only backing and queue device sanitization."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from spot_backend.action_claim_guard import reply_claims_unbacked_action
from spot_backend.config import Settings
from spot_backend.spotify_tools import SpotifyToolRunner


def test_whats_playing_reply_not_flagged_after_playback_state_tool() -> None:
    tools = {"spotify_playback_state"}
    reply = "Playing Slow Dancing in a Burning Room by John Mayer on your speaker."
    assert (
        reply_claims_unbacked_action(
            reply,
            tools,
            user_text="What's playing?",
        )
        is False
    )


def test_playlist_list_not_flagged_after_user_playlists_tool() -> None:
    tools = {"spotify_user_playlists"}
    reply = "Your playlists include Evening Acoustic and Road Trip Mix."
    assert (
        reply_claims_unbacked_action(
            reply,
            tools,
            user_text="What are my playlists?",
        )
        is False
    )


@respx.mock
def test_add_to_queue_strips_default_device_id(data_dir, signed_in_tokens) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [
                        {
                            "id": "tid001",
                            "uri": "spotify:track:tid001",
                            "name": "Gravity",
                            "artists": [{"name": "John Mayer"}],
                        }
                    ]
                }
            },
        )
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(
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
    )
    queue_route = respx.post(url__regex=r"https://api\.spotify\.com/v1/me/player/queue.*").mock(
        return_value=httpx.Response(204)
    )
    runner = SpotifyToolRunner(settings=Settings(data_dir=data_dir))
    raw = runner.run(
        "spotify_add_to_queue",
        {"track_name": "Gravity", "artist_name": "John Mayer", "device_id": "default"},
    )
    runner.close()
    assert json.loads(raw)["ok"] is True
    assert queue_route.called
    assert "device_id" not in (queue_route.calls.last.request.url.params or {})
