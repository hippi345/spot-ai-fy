"""Optional real-Gemini replay eval with mocked Spotify HTTP (never hits Spotify)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest
import respx

from spot_backend.config import Settings
from spot_backend.gemini_llm import run_chat_turn_gemini

_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "gemini_eval_latest.json"

pytestmark = pytest.mark.gemini_eval


def _seed_spotify_mocks() -> None:
    live_album = "00GCAlaaaaaaaaaaaaaaab"
    stale_album = "48YIv8aaaaaaaaaaaaaaab"
    show_id = "4rOoJ6Egrf8K2IrywzwOMy"
    pl_exact = "2HfFccisPxQfprhgIHM7XH"
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "item": {"type": "track", "name": "Track", "album": {"id": live_album, "name": "Live Album"}},
            },
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/library/contains.*").mock(
        return_value=httpx.Response(200, json=[True])
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/albums.*").mock(
        return_value=httpx.Response(
            200,
            json={"items": [{"album": {"id": stale_album, "name": "Old"}}], "total": 1},
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*type=show.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "shows": {
                    "items": [
                        {
                            "id": show_id,
                            "name": "StarTalk Radio",
                            "publisher": "StarTalk",
                            "uri": f"spotify:show:{show_id}",
                        }
                    ],
                    "total": 1,
                }
            },
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*type=playlist.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "playlists": {
                    "items": [
                        {"id": "wrong0000000000000001", "name": "90s HITS | TOP 100 SONGS"},
                        {"id": pl_exact, "name": "90s Rock Classics"},
                    ],
                    "total": 2,
                }
            },
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*type=track.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [
                        {
                            "id": "t" * 22,
                            "uri": f"spotify:track:{'t' * 22}",
                            "name": "Chill 90",
                            "album": {"name": "90s Chill", "release_date": "1994-01-01"},
                            "artists": [{"name": "Artist"}],
                        }
                    ]
                }
            },
        )
    )
    respx.post(url__regex=r"https://api\.spotify\.com/v1/me/playlists.*").mock(
        return_value=httpx.Response(200, json={"id": "newpl" + "x" * 17, "name": "spot-ai-fy test"})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/playlists/.*").mock(
        return_value=httpx.Response(200, json={"id": "newpl" + "x" * 17, "public": True, "name": "spot-ai-fy test"})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/playlists/.*").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.post(url__regex=r"https://api\.spotify\.com/v1/playlists/.*/items").mock(
        return_value=httpx.Response(200, json={})
    )


@pytest.fixture
def gemini_eval_settings(data_dir, signed_in_tokens):
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        pytest.skip("GEMINI_API_KEY not set")
    s = Settings()
    s.gemini_api_key = key
    return s


@respx.mock
def test_gemini_eval_replay_prompts(gemini_eval_settings, data_dir) -> None:
    _seed_spotify_mocks()
    prompts = [
        "do I already have this album saved?",
        "what albums do I have saved?",
        "build a chill 90s playlist called spot-ai-fy test",
        "drop track 3",
        "yes make it",
        "find podcasts about astronomy",
        "play the latest episode",
        "save this show",
        "is this show saved?",
        "remove it",
        "save the playlist 90s Rock Classics",
        "is it saved?",
    ]
    rows: list[dict] = []
    for prompt in prompts:
        reply = run_chat_turn_gemini(prompt, gemini_eval_settings, conversation_id="gemini-eval")
        rows.append({"prompt": prompt, "reply": reply[:500]})
    _ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    _ARTIFACT.write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")
    assert rows[0]["reply"]
