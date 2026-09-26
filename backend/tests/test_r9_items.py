"""Round-9 laptop retest items — test_r9_itemN_* per requirement."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from spot_backend.action_claim_guard import reply_claims_unbacked_action
from spot_backend.gemini_llm import run_chat_turn_gemini
from spot_backend.chat_messages import prepare_user_visible_reply, strip_internal_correction_leaks
from spot_backend.chat_shortcuts import try_deterministic_chat_reply
from spot_backend.chat_tool_state import last_playlist_id_from_chat_history, seed_runner_from_chat_history
from spot_backend.config import Settings
from spot_backend.gemini_llm import gemini_intent_allowed_function_names
from spot_backend.prompt_intent import informational_system_suffix, prompt_is_pure_how_to
from spot_backend.spotify_tools import SpotifyToolRunner
from tests.recheck_helpers import (
    gemini_candidates_payload,
    gemini_stop_candidate,
    make_gemini_post_recorder,
    run_playback_restriction_violated,
)


def _gemini_tool_names_from_body(body: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for part in (body.get("contents") or []):
        if not isinstance(part, dict) or part.get("role") != "model":
            continue
        for p in part.get("parts") or []:
            if isinstance(p, dict):
                fc = p.get("functionCall")
                if isinstance(fc, dict) and fc.get("name"):
                    names.append(str(fc["name"]))
    return names


def _run_gemini_how_to_turn(
    user_text: str,
    *,
    history: list[dict[str, str]] | None = None,
    first_answer: str = "Open the playlist menu and choose Delete.",
) -> tuple[str, list[dict[str, Any]], list[str]]:
    settings = Settings(gemini_api_key="k", agent_max_steps=8)
    bodies: list[dict[str, Any]] = []
    call_n = {"n": 0}

    def handler(body: dict[str, Any], _n: int, req: httpx.Request) -> httpx.Response:
        bodies.append(body)
        call_n["n"] += 1
        if call_n["n"] == 1:
            part = {"text": first_answer}
        else:
            part = {
                "functionCall": {
                    "name": "spotify_user_playlists",
                    "args": {},
                }
            }
        payload = gemini_candidates_payload(gemini_stop_candidate(part))
        return httpx.Response(200, json=payload, request=req)

    _, fake_post = make_gemini_post_recorder(handler)
    with patch("httpx.Client.post", fake_post):
        with patch.object(SpotifyToolRunner, "run", lambda *a, **k: json.dumps({"items": []})):
            text = run_chat_turn_gemini(user_text, settings, history=history)
    tool_names: list[str] = []
    for b in bodies:
        tool_names.extend(_gemini_tool_names_from_body(b))
    return text, bodies, tool_names


@pytest.mark.parametrize(
    "phrase",
    (
        "how can I delete a playlist?",
        "how do I make a playlist collaborative?",
        "how do I play an artist?",
    ),
)
def test_r9_item1_how_to_skips_tool_nudge_keeps_first_answer(phrase: str) -> None:
    assert prompt_is_pure_how_to(phrase)
    text, bodies, tool_names = _run_gemini_how_to_turn(phrase)
    assert "Open the playlist menu" in text
    nudge_hits = sum(
        1
        for b in bodies
        for c in b.get("contents") or []
        if c.get("role") == "user"
        for p in c.get("parts") or []
        if isinstance(p, dict)
        and "did not call any Spotify tools" in str(p.get("text") or "")
    )
    assert nudge_hits == 0
    assert "spotify_user_playlists" not in tool_names


def test_r9_item1_how_to_with_history_still_no_nudge() -> None:
    history = [
        {"role": "user", "content": "what are my playlists?"},
        {"role": "assistant", "content": "You have several playlists in your library."},
    ]
    text, _bodies, tool_names = _run_gemini_how_to_turn(
        "how do I make a playlist private?",
        history=history,
    )
    assert "playlist" in text.lower()
    assert not tool_names


@respx.mock
def test_r9_item2_artist_play_uses_context_uri_not_single_track(
    data_dir, signed_in_tokens,
) -> None:
    artist_id = "aaaaaaaaaaaaaaaaaaaaaa"
    track_id = "bbbbbbbbbbbbbbbbbbbbbb"
    play_calls: list[dict[str, Any]] = []

    def capture_play(request: httpx.Request) -> httpx.Response:
        play_calls.append({"url": str(request.url), "body": request.content})
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(side_effect=capture_play)
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/pause.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": [{"id": "dev1", "is_active": True}]})
    )
    respx.put("https://api.spotify.com/v1/me/player").mock(return_value=httpx.Response(204))
    respx.get(f"https://api.spotify.com/v1/tracks/{track_id}").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": track_id,
                "artists": [{"id": artist_id, "name": "Radiohead"}],
                "album": {"id": "cccccccccccccccccccccc"},
            },
        )
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "context": {"uri": "spotify:album:cccccccccccccccccccccc"},
                "item": {"uri": "spotify:track:ffffffffffffffffffffff"},
            },
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    runner._last_primary_artist_id = artist_id
    raw = runner.run(
        "spotify_start_resume_playback",
        {"uris": [f"spotify:track:{track_id}"]},
    )
    runner.close()
    data = json.loads(raw)
    assert play_calls
    sent = json.loads(play_calls[0]["body"].decode() or "{}")
    assert sent.get("context_uri") == f"spotify:artist:{artist_id}"
    assert "uris" not in sent or sent.get("uris") is None
    assert data.get("requested_body", {}).get("context_uri") == f"spotify:artist:{artist_id}"


@respx.mock
def test_r9_item2_restriction_403_no_transfer_retry(data_dir, signed_in_tokens) -> None:
    track_id = "dddddddddddddddddddddd"
    transfer = respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player(\?.*)?$").mock(
        return_value=httpx.Response(204)
    )
    play = respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(
            403,
            json={"error": {"status": 403, "message": "Restriction violated"}},
        )
    )
    data = run_playback_restriction_violated(track_id)
    assert play.call_count == 1
    assert not transfer.called
    assert "stuck state" in data.get("error", "").lower()


@respx.mock
def test_r9_item3_like_this_saves_current_track(data_dir, signed_in_tokens) -> None:
    track_id = "eeeeeeeeeeeeeeeeeeeeee"
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={"item": {"id": track_id, "name": "Kill Bill"}},
        )
    )
    respx.get(f"https://api.spotify.com/v1/tracks/{track_id}").mock(
        return_value=httpx.Response(200, json={"id": track_id})
    )
    lib = respx.put("https://api.spotify.com/v1/me/library").mock(return_value=httpx.Response(200))
    runner = SpotifyToolRunner(settings=Settings())
    reply = try_deterministic_chat_reply("like this", runner)
    runner.close()
    assert reply and "Kill Bill" in reply
    assert lib.called


@respx.mock
def test_r9_item3_undo_without_prior_save_is_honest(data_dir, signed_in_tokens) -> None:
    runner = SpotifyToolRunner(settings=Settings())
    reply = try_deterministic_chat_reply("undo that", runner)
    runner.close()
    assert reply
    assert "nothing" in reply.lower()


def test_r9_item3_false_liked_claim_detected_without_save_tool() -> None:
    assert reply_claims_unbacked_action("I've liked 'Kill Bill'.", set())


def test_r9_item3_raw_json_sanitized_from_reply() -> None:
    raw = '{"error": "Nothing is playing"}'
    assert prepare_user_visible_reply(raw) == "Nothing is playing"


def test_r9_item4_playing_claim_without_tool() -> None:
    assert reply_claims_unbacked_action("Playing Radiohead's top tracks: Creep, Let Down…", set())


def test_r9_item4_playing_claim_backed_by_play_tool() -> None:
    assert not reply_claims_unbacked_action(
        "Playing Radiohead now.",
        {"spotify_start_resume_playback"},
    )


@respx.mock
def test_r9_item5_create_then_make_private_resolves_it(
    data_dir, signed_in_tokens,
) -> None:
    pid = "aaaaaaaaaaaaaaaaaaaaaa"
    history = [
        {
            "role": "assistant",
            "content": f"Created your playlist (spotify:playlist:{pid}).",
        },
    ]
    assert last_playlist_id_from_chat_history(history) == pid
    respx.put(f"https://api.spotify.com/v1/playlists/{pid}").mock(return_value=httpx.Response(200))
    respx.get(f"https://api.spotify.com/v1/playlists/{pid}").mock(
        return_value=httpx.Response(200, json={"id": pid, "public": False, "name": "Mix"})
    )
    runner = SpotifyToolRunner(settings=Settings())
    seed_runner_from_chat_history(runner, history)
    raw = runner.run("spotify_update_playlist", {"playlist_id": "it", "public": False})
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is True


def test_r9_item6_correction_preamble_stripped() -> None:
    messy = (
        "My apologies — in my previous response I stated I could not skip.\n\n"
        "Playback is paused now."
    )
    cleaned = strip_internal_correction_leaks(messy)
    assert "apologies" not in cleaned.lower()
    assert "previous response" not in cleaned.lower()
    assert "paused" in cleaned.lower()


def test_r9_item7_gemini_intent_routes_lately_to_recently_played() -> None:
    names = gemini_intent_allowed_function_names("what have I been listening to lately?")
    assert names == ["spotify_recently_played"]


def test_r9_item8_how_to_guidance_mentions_crossfade() -> None:
    phrase = "how do I turn on crossfade in the app?"
    assert prompt_is_pure_how_to(phrase)
    guidance = informational_system_suffix(phrase).lower()
    assert "crossfade" in guidance
    assert "settings" in guidance and "playback" in guidance
