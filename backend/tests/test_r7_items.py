"""Round-7 PR items — dedicated test_r7_itemN_* per requirement."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from spot_backend.agent import iter_ollama_chat_events
from spot_backend.config import Settings
from spot_backend.gemini_llm import apply_gemini_function_calling_tools, run_chat_turn_gemini
from spot_backend.prompt_intent import (
    INFORMATIONAL_REPLY_SYSTEM_SUFFIX,
    prompt_is_informational,
)
from spot_backend.spotify_tools import SpotifyToolRunner
from tests.recheck_helpers import (
    FakeOllamaStream,
    gemini_candidates_payload,
    gemini_stop_candidate,
    make_gemini_post_recorder,
)

R7_NEW_HOW_TO_PHRASES = (
    "where do I find my liked songs?",
    "could you walk me through creating a playlist?",
    "what's the best way to queue a song?",
    "help me understand repeat modes",
    "any tips for making a playlist private?",
    "how do I share a playlist with a friend?",
    "what should I tell Spotify to skip a track?",
    "can you explain how collaborative playlists work?",
    "is there a way to hide my listening activity?",
    "tell me about crossfade in the Spotify app",
    "how would I change the order of songs in a playlist?",
    "what is the way to follow an artist on mobile?",
    "how can I download podcasts for offline listening?",
    "explain how the queue differs from Up Next",
    "where can I see lyrics in Spotify?",
)

R7_ACTION_NOT_INFORMATIONAL = (
    "make my playlist Workout private",
    "delete playlist Old Mix",
    "like this",
    "save this album",
    "play Radiohead",
    "shuffle on",
    "skip",
    "turn it up",
    "add this to Workout",
    "follow this artist",
)


def _gemini_payload_has_valid_tooling(body: dict[str, Any]) -> bool:
    tools = body.get("tools")
    decls: list[Any] = []
    if isinstance(tools, list) and tools:
        first = tools[0]
        if isinstance(first, dict):
            decls = first.get("functionDeclarations") or []
    has_decls = bool(decls)
    has_tool_config = "toolConfig" in body
    if has_tool_config and not has_decls:
        return False
    if has_decls and not has_tool_config:
        return False
    return True


def _capture_gemini_first_body(user_text: str) -> dict[str, Any]:
    settings = Settings(gemini_api_key="k")

    def handler(body: dict[str, Any], _n: int, req: httpx.Request) -> httpx.Response:
        payload = gemini_candidates_payload(
            gemini_stop_candidate({"text": "Here is the answer."})
        )
        return httpx.Response(200, json=payload, request=req)

    bodies, fake_post = make_gemini_post_recorder(handler)
    with patch("httpx.Client.post", fake_post):
        run_chat_turn_gemini(user_text, settings)
    assert bodies
    return bodies[0]


@pytest.mark.parametrize(
    "user_text",
    (
        "how do I like a song in Spotify?",
        "pause",
        "add a track then play my workout playlist",
    ),
)
def test_r7_item1_gemini_never_tool_config_without_declarations(
    data_dir, signed_in_tokens, user_text: str,
) -> None:
    body = _capture_gemini_first_body(user_text)
    assert _gemini_payload_has_valid_tooling(body)


def test_r7_item1_apply_gemini_function_calling_tools_unit() -> None:
    body: dict[str, Any] = {}
    apply_gemini_function_calling_tools(body, decls=[], fc_cfg={"mode": "AUTO"})
    assert "toolConfig" not in body
    assert "tools" not in body
    decl = [{"name": "spotify_pause", "description": "x", "parameters": {"type": "object"}}]
    apply_gemini_function_calling_tools(body, decls=decl, fc_cfg={"mode": "ANY"})
    assert body["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    assert body["tools"][0]["functionDeclarations"] == decl


@respx.mock
def test_r7_item2_like_then_undo_and_invalid_id_errors(data_dir, signed_in_tokens) -> None:
    track_id = "4uLU6hMCjMI75M1A2tKUQC"
    bogus_id = "3NpeMjx6VwX05JmI83G0Fw"
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={"item": {"id": track_id, "name": "Nude", "album": {"id": "aaaaaaaaaaaaaaaaaaaaaa"}}},
        )
    )
    respx.get(f"https://api.spotify.com/v1/tracks/{track_id}").mock(
        return_value=httpx.Response(200, json={"id": track_id})
    )
    respx.put("https://api.spotify.com/v1/me/library").mock(return_value=httpx.Response(200))
    respx.get(f"https://api.spotify.com/v1/tracks/{bogus_id}").mock(
        return_value=httpx.Response(404, json={"error": {"status": 404, "message": "Not found"}})
    )
    delete_route = respx.delete("https://api.spotify.com/v1/me/library").mock(
        return_value=httpx.Response(200)
    )

    runner = SpotifyToolRunner(settings=Settings())
    save_raw = runner.run("spotify_save_tracks", {"track_id": "this"})
    save_data = json.loads(save_raw)
    assert save_data.get("ok") is True

    bad_raw = runner.run("spotify_unsave_tracks", {"track_id": bogus_id})
    bad_data = json.loads(bad_raw)
    assert bad_data.get("ok") is False
    assert "error" in bad_data
    assert delete_route.call_count == 0

    undo_raw = runner.run("spotify_unsave_tracks", {"track_id": "that"})
    undo_data = json.loads(undo_raw)
    assert undo_data.get("ok") is True
    assert undo_data.get("removed_track_ids") == [track_id]
    assert delete_route.call_count == 1
    runner.close()


@respx.mock
def test_r7_item3_playlist_visibility_readback_mismatch_and_match(
    data_dir, signed_in_tokens,
) -> None:
    pid = "playlist1111111111111"
    respx.put(f"https://api.spotify.com/v1/playlists/{pid}").mock(
        return_value=httpx.Response(200)
    )

    respx.get(f"https://api.spotify.com/v1/playlists/{pid}").mock(
        return_value=httpx.Response(200, json={"id": pid, "public": True, "name": "Mix"})
    )
    runner = SpotifyToolRunner(settings=Settings())
    runner._session_known_ids.add(pid)
    mismatch_raw = runner.run(
        "spotify_update_playlist", {"playlist_id": pid, "public": False}
    )
    mismatch = json.loads(mismatch_raw)
    assert mismatch.get("ok") is False
    assert mismatch.get("visibility_mismatch") is True
    assert mismatch.get("visibility_warning")

    respx.get(f"https://api.spotify.com/v1/playlists/{pid}").mock(
        return_value=httpx.Response(200, json={"id": pid, "public": False, "name": "Mix"})
    )
    match_raw = runner.run(
        "spotify_update_playlist", {"playlist_id": pid, "public": False}
    )
    match = json.loads(match_raw)
    assert match.get("ok") is True
    assert "visibility_warning" not in match
    runner.close()


@pytest.mark.parametrize("phrase", R7_NEW_HOW_TO_PHRASES)
def test_r7_item4_new_how_to_phrasings_are_informational(phrase: str) -> None:
    assert prompt_is_informational(phrase) is True


@pytest.mark.parametrize("phrase", R7_ACTION_NOT_INFORMATIONAL)
def test_r7_item4_action_phrasings_still_not_informational(phrase: str) -> None:
    assert prompt_is_informational(phrase) is False


R7_STRUCTURAL_HELD_OUT_INFORMATIONAL = (
    "why does shuffle keep repeating the same songs?",
    "is it possible to share a playlist with a friend?",
    "when should I use smart shuffle?",
    "does Spotify let me hide a song?",
    "who can see my private playlists?",
    "tell me about crossfade",
    "should I use the queue or a playlist for a party?",
    "explain the difference between liking and saving an album",
    "can playlists be collaborative?",
    "is there any limit on playlist size?",
)

R7_POLITE_ACTION_NOT_INFORMATIONAL = (
    "can you play Radiohead?",
    "could you pause the music?",
    "would you skip this song?",
    "will you add this to my Road Trip playlist?",
    "please like this song",
    "can you make my playlist Chill private?",
)


@pytest.mark.parametrize(
    "phrase,expect_informational",
    tuple((p, True) for p in R7_STRUCTURAL_HELD_OUT_INFORMATIONAL)
    + tuple((p, False) for p in R7_POLITE_ACTION_NOT_INFORMATIONAL),
)
def test_r7_item4_structural_held_out(phrase: str, expect_informational: bool) -> None:
    assert prompt_is_informational(phrase) is expect_informational


def test_r7_item5_informational_guidance_mentions_heart_like_not_contradiction() -> None:
    low = INFORMATIONAL_REPLY_SYSTEM_SUFFIX.lower()
    assert "heart" in low
    assert "liked songs" in low
    assert "add to liked songs" in low or "+" in INFORMATIONAL_REPLY_SYSTEM_SUFFIX
    assert "impossible" in low or "never tell" in low
    assert "doesn't have a like button" not in low
    assert "you can't like" not in low


@respx.mock
def test_r7_item6_get_album_returns_all_ten_tracks(data_dir, signed_in_tokens) -> None:
    album_id = "bbbbbbbbbbbbbbbbbbbbbb"
    tracks = [{"id": f"t{i:02d}", "name": f"Track {i}", "track_number": i} for i in range(1, 11)]

    def album_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/tracks"):
            offset = int(request.url.params.get("offset", 0))
            limit = int(request.url.params.get("limit", 50))
            batch = tracks[offset : offset + limit]
            next_offset = offset + len(batch)
            payload: dict[str, Any] = {
                "items": batch,
                "total": 10,
                "limit": limit,
                "offset": offset,
                "next": (
                    f"https://api.spotify.com/v1/albums/{album_id}/tracks?offset={next_offset}&limit=7"
                    if next_offset < 10
                    else None
                ),
            }
            return httpx.Response(200, json=payload)
        return httpx.Response(
            200,
            json={
                "id": album_id,
                "name": "In Rainbows",
                "total_tracks": 10,
                "tracks": {"items": tracks[:7], "total": 10},
            },
        )

    respx.route(url__regex=rf"https://api\.spotify\.com/v1/albums/{album_id}.*").mock(
        side_effect=album_handler
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_get_album", {"album_id": album_id})
    runner.close()
    data = json.loads(raw)
    listed = data.get("tracks", {}).get("items") or []
    assert len(listed) == 10
    assert data.get("tracks", {}).get("total") == 10


@respx.mock
def test_r7_item7_ollama_path_unknown_device_id_fallback(data_dir, signed_in_tokens) -> None:
    from spot_backend.token_store import DeviceSelection, save_device
    from tests.recheck_helpers import mock_spotify_active_device, mock_spotify_idle_player_state

    settings = Settings()
    save_device_id = "ollama_saved_device_99"
    save_device(settings.resolved_device_path, DeviceSelection(device_id=save_device_id))
    mock_spotify_active_device(save_device_id)
    mock_spotify_idle_player_state()

    streams = [
        FakeOllamaStream(
            [
                json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "spotify_start_resume_playback",
                                        "arguments": {"device_id": "device_123"},
                                    }
                                }
                            ],
                        },
                        "done": True,
                    }
                ),
                json.dumps(
                    {"message": {"role": "assistant", "content": "Playing now."}, "done": True}
                ),
            ]
        )
    ]
    idx = {"i": 0}

    def fake_stream(_self, _method, _url, **kwargs):
        stream = streams[min(idx["i"], len(streams) - 1)]
        idx["i"] += 1
        return stream

    with patch("httpx.Client.stream", fake_stream):
        events = list(iter_ollama_chat_events("play something", settings))
    tool_dones = [e for e in events if e.get("type") == "tool_done"]
    assert tool_dones
    assert tool_dones[0].get("name") == "spotify_start_resume_playback"
    preview = tool_dones[0].get("preview") or ""
    assert "device_123" in preview or '"ok": true' in preview.lower()
    play_requests = [c.request for c in respx.calls if "player/play" in str(c.request.url)]
    assert play_requests
    assert save_device_id in str(play_requests[-1].url)
