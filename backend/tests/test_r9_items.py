"""Round-9 laptop retest items — test_r9_itemN_* per requirement."""

from __future__ import annotations

import copy
import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from spot_backend.action_claim_guard import reply_claims_unbacked_action
from spot_backend.agent import _forced_json_tool_calls_for_question
from spot_backend.chat_messages import prepare_user_visible_reply, strip_internal_correction_leaks
from spot_backend.chat_shortcuts import try_deterministic_chat_reply
from spot_backend.chat_tool_state import last_playlist_id_from_chat_history, seed_runner_from_chat_history
from spot_backend.config import Settings
from spot_backend.gemini_llm import _GEMINI_TOOL_NUDGE, gemini_intent_allowed_function_names, run_chat_turn_gemini
from spot_backend.prompt_intent import informational_system_suffix, prompt_is_pure_how_to
from spot_backend.reply_grounding import (
    ground_reply_artist_credits,
    reply_mentions_artists_outside_tool_data,
)
from spot_backend.spotify_tools import PLAY_DEVICE_404_RETRY_DELAY_SECONDS, SpotifyToolRunner
from tests.recheck_helpers import (
    FakeOllamaStream,
    gemini_candidates_payload,
    gemini_stop_candidate,
    make_gemini_post_recorder,
    run_playback_restriction_violated,
)


def _gemini_tool_names_from_body(body: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for part in body.get("contents") or []:
        if not isinstance(part, dict) or part.get("role") != "model":
            continue
        for p in part.get("parts") or []:
            if isinstance(p, dict):
                fc = p.get("functionCall")
                if isinstance(fc, dict) and fc.get("name"):
                    names.append(str(fc["name"]))
    return names


def _count_gemini_nudges(bodies: list[dict[str, Any]]) -> int:
    return sum(
        1
        for b in bodies
        for c in b.get("contents") or []
        if c.get("role") == "user"
        for p in c.get("parts") or []
        if isinstance(p, dict) and _GEMINI_TOOL_NUDGE in str(p.get("text") or "")
    )


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


def _run_gemini_non_how_to_nudge_scenario() -> tuple[str, list[dict[str, Any]]]:
    settings = Settings(gemini_api_key="k", agent_max_steps=6)
    bodies: list[dict[str, Any]] = []
    call_n = {"n": 0}
    first = "Here is a helpful overview of your library."
    second = "Still summarizing without calling tools."

    def handler(_body: dict[str, Any], _n: int, req: httpx.Request) -> httpx.Response:
        bodies.append(copy.deepcopy(_body))
        call_n["n"] += 1
        part = {"text": first if call_n["n"] == 1 else second}
        payload = gemini_candidates_payload(gemini_stop_candidate(part))
        return httpx.Response(200, json=payload, request=req)

    _, fake_post = make_gemini_post_recorder(handler)
    with patch("httpx.Client.post", fake_post):
        with patch.object(SpotifyToolRunner, "run", lambda *a, **k: json.dumps({"items": []})):
            text = run_chat_turn_gemini("list my playlists", settings)
    return text, bodies


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
    assert _count_gemini_nudges(bodies) == 0
    assert "spotify_user_playlists" not in tool_names


def test_r9_item1_how_to_with_history_still_no_nudge() -> None:
    history = [
        {"role": "user", "content": "what are my playlists?"},
        {"role": "assistant", "content": "You have several playlists in your library."},
    ]
    text, bodies, tool_names = _run_gemini_how_to_turn(
        "how do I make a playlist private?",
        history=history,
    )
    assert "playlist" in text.lower()
    assert not tool_names
    assert _count_gemini_nudges(bodies) == 0


def test_r9_item1_non_how_to_nudge_at_most_once_keeps_first_text() -> None:
    text, bodies = _run_gemini_non_how_to_nudge_scenario()
    assert "helpful overview" in text
    per_request = [_count_gemini_nudges([b]) for b in bodies]
    assert per_request[0] == 0
    assert per_request[-1] == 1
    assert sum(per_request) == 1
    assert "spotify_user_playlists" not in _gemini_tool_names_from_body(bodies[-1])


@respx.mock
def test_r9_item2_play_artist_fresh_session_uses_context_uri(
    data_dir, signed_in_tokens,
) -> None:
    artist_id = "aaaaaaaaaaaaaaaaaaaaaa"
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(
            200,
            json={"artists": {"items": [{"id": artist_id, "name": "Radiohead"}]}},
        )
    )
    respx.get(f"https://api.spotify.com/v1/artists/{artist_id}").mock(
        return_value=httpx.Response(200, json={"id": artist_id, "name": "Radiohead"})
    )
    play_calls: list[bytes] = []

    def capture_play(request: httpx.Request) -> httpx.Response:
        play_calls.append(request.content or b"")
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(side_effect=capture_play)
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "context": {"uri": f"spotify:artist:{artist_id}"},
                "item": {"uri": "spotify:track:bbbbbbbbbbbbbbbbbbbbbb"},
            },
        )
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": [{"id": "dev1", "is_active": True}]})
    )
    runner = SpotifyToolRunner(settings=Settings())
    reply = try_deterministic_chat_reply("can you play Radiohead?", runner)
    runner.close()
    assert reply and "Radiohead" in reply
    assert play_calls
    sent = json.loads(play_calls[0].decode() or "{}")
    assert sent.get("context_uri") == f"spotify:artist:{artist_id}"
    assert not sent.get("uris")


@respx.mock
def test_r9_item2_play_404_one_delayed_retry(data_dir, signed_in_tokens) -> None:
    artist_id = "cccccccccccccccccccccc"
    _mock_artist_catalog_get(artist_id)
    play_count = {"n": 0}
    sleeps: list[float] = []

    def play_handler(request: httpx.Request) -> httpx.Response:
        play_count["n"] += 1
        if play_count["n"] == 1:
            return httpx.Response(404, json={"error": {"status": 404, "message": "No active device"}})
        return httpx.Response(204)

    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(side_effect=play_handler)
    respx.put("https://api.spotify.com/v1/me/player").mock(return_value=httpx.Response(204))
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(
            200,
            json={"devices": [{"id": "dev404", "is_restricted": False, "is_active": True}]},
        )
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "context": {"uri": f"spotify:artist:{artist_id}"},
                "item": {"uri": "spotify:track:dddddddddddddddddddddd"},
            },
        )
    )

    with patch("spot_backend.spotify_tools.time.sleep", side_effect=lambda s: sleeps.append(s)):
        runner = SpotifyToolRunner(settings=Settings())
        raw = runner.run(
            "spotify_start_resume_playback",
            {"context_uri": f"spotify:artist:{artist_id}"},
        )
        runner.close()

    data = json.loads(raw)
    assert play_count["n"] == 2
    assert PLAY_DEVICE_404_RETRY_DELAY_SECONDS in sleeps
    assert data.get("ok") is True
    assert data.get("playback_verified") is True


def _mock_artist_catalog_get(artist_id: str) -> None:
    respx.get(f"https://api.spotify.com/v1/artists/{artist_id}").mock(
        return_value=httpx.Response(200, json={"id": artist_id, "name": "Artist"})
    )


@respx.mock
def test_r9_item2_post_play_verified_success_path(data_dir, signed_in_tokens) -> None:
    artist_id = "eeeeeeeeeeeeeeeeeeeeee"
    _mock_artist_catalog_get(artist_id)
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "context": {"uri": f"spotify:artist:{artist_id}"},
                "item": {"uri": "spotify:track:ffffffffffffffffffffff"},
            },
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run(
        "spotify_start_resume_playback",
        {"context_uri": f"spotify:artist:{artist_id}"},
    )
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is True
    assert data.get("playback_verified") is True


@respx.mock
def test_r9_item2_post_play_clear_report_when_stuck(data_dir, signed_in_tokens) -> None:
    artist_id = "1111111111111111111111"
    _mock_artist_catalog_get(artist_id)
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.put("https://api.spotify.com/v1/me/player").mock(return_value=httpx.Response(204))
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "context": {"uri": "spotify:album:2222222222222222222222"},
                "item": {"uri": "spotify:track:3333333333333333333333"},
            },
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run(
        "spotify_start_resume_playback",
        {"context_uri": f"spotify:artist:{artist_id}"},
    )
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is False
    assert data.get("playback_verified") is False
    assert "did not switch" in data.get("error", "").lower()


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
@pytest.mark.parametrize("phrase", ("save this song", "heart this"))
def test_r9_item3_save_and_heart_phrases_save_track(
    data_dir, signed_in_tokens, phrase: str,
) -> None:
    track_id = "ababababababababababab"
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(200, json={"item": {"id": track_id, "name": "SO GOOD"}})
    )
    respx.get(f"https://api.spotify.com/v1/tracks/{track_id}").mock(
        return_value=httpx.Response(200, json={"id": track_id})
    )
    lib = respx.put("https://api.spotify.com/v1/me/library").mock(return_value=httpx.Response(200))
    runner = SpotifyToolRunner(settings=Settings())
    reply = try_deterministic_chat_reply(phrase, runner)
    runner.close()
    assert reply and "SO GOOD" in reply
    assert lib.called


@respx.mock
def test_r9_item3_undo_after_save_unsaves_exact_track(data_dir, signed_in_tokens) -> None:
    track_id = "bcbcbcbcbcbcbcbcbcbcbc"
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(200, json={"item": {"id": track_id, "name": "Track"}})
    )
    respx.get(f"https://api.spotify.com/v1/tracks/{track_id}").mock(
        return_value=httpx.Response(200, json={"id": track_id})
    )
    respx.put("https://api.spotify.com/v1/me/library").mock(return_value=httpx.Response(200))
    delete_route = respx.delete(url__regex=r"https://api\.spotify\.com/v1/me/library.*").mock(
        return_value=httpx.Response(200)
    )
    queue_route = respx.get("https://api.spotify.com/v1/me/player/queue").mock(
        return_value=httpx.Response(200, json={"queue": []})
    )
    runner = SpotifyToolRunner(settings=Settings())
    try_deterministic_chat_reply("like this", runner)
    undo = try_deterministic_chat_reply("undo that", runner)
    runner.close()
    assert undo and "removed" in undo.lower()
    assert delete_route.called
    assert queue_route.call_count == 0


@respx.mock
def test_r9_item3_undo_without_save_does_not_touch_queue(data_dir, signed_in_tokens) -> None:
    delete_route = respx.delete("https://api.spotify.com/v1/me/library").mock(
        return_value=httpx.Response(200)
    )
    queue_route = respx.delete(url__regex=r"https://api\.spotify\.com/v1/me/player/queue.*").mock(
        return_value=httpx.Response(200)
    )
    runner = SpotifyToolRunner(settings=Settings())
    reply = try_deterministic_chat_reply("undo that", runner)
    runner.close()
    assert reply and "nothing" in reply.lower()
    assert not delete_route.called
    assert not queue_route.called


@respx.mock
def test_r9_item3_like_nothing_playing_clear_message_no_json(data_dir, signed_in_tokens) -> None:
    respx.get("https://api.spotify.com/v1/me/player").mock(return_value=httpx.Response(200, json={}))
    runner = SpotifyToolRunner(settings=Settings())
    reply = try_deterministic_chat_reply("like this", runner)
    runner.close()
    assert reply
    assert "something is playing" in reply.lower()
    assert "{" not in reply


def test_r9_item3_false_liked_claim_detected_without_save_tool() -> None:
    assert reply_claims_unbacked_action("I've liked 'Kill Bill'.", set())


def test_r9_item3_raw_json_sanitized_from_reply() -> None:
    raw = '{"error": "Nothing is playing"}'
    assert prepare_user_visible_reply(raw) == "Nothing is playing"


@pytest.mark.parametrize(
    "phrase,tools,expect_flagged",
    [
        ("Playing Radiohead now.", set(), True),
        ("Now playing Creep.", set(), True),
        ("Liked that track for you.", set(), True),
        ("Saved it to your library.", set(), True),
        ("Added it to your liked songs.", set(), True),
        ("Skipped to the next song.", set(), True),
        ("Now playing Creep.", {"spotify_start_resume_playback"}, False),
        ("Saved it to your library.", {"spotify_save_tracks"}, False),
        ("Skipped to the next song.", {"spotify_skip_next"}, False),
    ],
)
def test_r9_item4_present_tense_claim_guard(
    phrase: str, tools: set[str], expect_flagged: bool,
) -> None:
    flagged = reply_claims_unbacked_action(phrase, tools)
    assert flagged is expect_flagged


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


def test_r9_item8_gemini_intent_routes_lately_to_recently_played() -> None:
    names = gemini_intent_allowed_function_names("what have I been listening to lately?")
    assert names == ["spotify_recently_played"]


def test_r9_item8_how_to_guidance_mentions_crossfade() -> None:
    phrase = "how do I turn on crossfade in the app?"
    assert prompt_is_pure_how_to(phrase)
    guidance = informational_system_suffix(phrase).lower()
    assert "crossfade" in guidance
    assert "settings" in guidance and "playback" in guidance


def test_r9_item8_grounding_fixes_so_good_misattribution() -> None:
    tool_json = json.dumps(
        {
            "tracks": {
                "items": [
                    {
                        "name": "SO GOOD",
                        "artists": [{"name": "Doechii"}],
                    }
                ]
            }
        }
    )
    reply = '"SO GOOD" by SZA is in your recent rotation.'
    assert reply_mentions_artists_outside_tool_data(reply, [tool_json]) == ["SZA"]
    fixed = ground_reply_artist_credits(reply, [tool_json])
    assert "SZA" not in fixed
    assert "Doechii" in fixed


def test_r9_item8_ollama_json_mode_lately_forces_recently_played() -> None:
    from spot_backend.agent import _forced_json_tool_calls_for_question

    forced = _forced_json_tool_calls_for_question("what have I been listening to lately?")
    assert forced is not None
    assert forced[0]["function"]["name"] == "spotify_recently_played"
    assert "spotify_top_tracks" not in str(forced)
