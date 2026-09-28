"""PR #9 round-5: intent-based finalize replies + Gemini-path sim sequences."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from spot_backend.chat_messages import prepare_user_visible_reply
from spot_backend.config import Settings
from spot_backend.gemini_llm import run_chat_turn_gemini
from spot_backend.library_mutation_store import record_library_mutation
from spot_backend.llm_tool_loop import ToolLoopState, finalize_assistant_text, run_tool_calls
from spot_backend.prompt_intent import refused_mutating_tool_result
from spot_backend.reply_tool_trace import append_tool_trace_record, tool_trace_outcome
from spot_backend.spotify_tools import SpotifyToolRunner
from spot_backend.tool_server_enforcement import enforce_tool_arguments_for_turn


def _finalize(user: str, model: str, names: list[str], results: list[str]) -> str:
    state = ToolLoopState()
    state.tool_results = list(results)
    state.turn_tool_calls = list(zip(names, results))
    for n, r in state.turn_tool_calls:
        if json.loads(r).get("ok") is not False and not json.loads(r).get("failure_reason") == "guard_refused":
            state.successful_tools.add(n)
    action = finalize_assistant_text(model, state, user_text=user)
    assert action.kind == "return"
    return prepare_user_visible_reply(action.text, state.tool_results)


def test_t22_saved_shows_beats_guard_boilerplate() -> None:
    saved = json.dumps(
        {
            "ok": True,
            "user_message": "1. StarTalk — Neil\n2. Other — Pub\n(2 saved shows total — and 0 more not listed here.)",
        }
    )
    blocked = refused_mutating_tool_result("spotify_start_resume_playback")
    guard_text = (
        "That was a question-only turn, so I didn't run a playback change. "
        "Ask me to play something if you want me to start it."
    )
    reply = _finalize(
        "what podcasts do I follow?",
        guard_text,
        ["spotify_user_saved_shows", "spotify_start_resume_playback"],
        [saved, blocked],
    )
    assert "StarTalk" in reply
    assert "question-only" not in reply.lower()


def test_t21_play_failure_beats_truncated_search_line() -> None:
    search = json.dumps({"ok": True, "user_message": "1. StarTalk with Neil deGrasse Tyson —"})
    play_fail = json.dumps(
        {
            "ok": False,
            "failure_reason": "playback_not_verified",
            "episode_name": "A Universe of Possibilities",
            "user_message": (
                "I found the latest episode, 'A Universe of Possibilities', but Spotify didn't confirm "
                "it started playing. Try tapping play on your device or ask me to transfer playback."
            ),
        }
    )
    reply = _finalize(
        "Play the latest StarTalk episode",
        search.split("\n", 1)[0],
        ["spotify_search", "spotify_play_show_latest_episode"],
        [search, play_fail],
    )
    assert "didn't confirm" in reply.lower() or "didn't confirm" in reply
    assert "Possibilities" in reply


def test_t23_save_show_beats_search_line() -> None:
    search = json.dumps({"ok": True, "user_message": "1. StarTalk with Neil deGrasse Tyson —"})
    saved = json.dumps(
        {
            "ok": True,
            "saved_uris": ["spotify:show:4rOoJ6Egrf8K2IrywzwOMy"],
            "item_name": "StarTalk with Neil deGrasse Tyson",
            "user_message": "Saved StarTalk with Neil deGrasse Tyson to your library.",
        }
    )
    reply = _finalize(
        "Save this show",
        search.split("\n", 1)[0],
        ["spotify_search", "spotify_library_save"],
        [search, saved],
    )
    assert "Saved StarTalk" in reply


def test_t25_remove_beats_false_contains() -> None:
    verify = json.dumps(
        {
            "ok": True,
            "saved_single": False,
            "user_message": "No — that's not in your library.",
        }
    )
    removed = json.dumps(
        {
            "ok": True,
            "removed_uris": ["spotify:show:4rOoJ6Egrf8K2IrywzwOMy"],
            "item_name": "StarTalk with Neil deGrasse Tyson",
            "user_message": "Removed StarTalk with Neil deGrasse Tyson from your library.",
        }
    )
    reply = _finalize(
        "Remove it",
        "No — that's not in your library.",
        ["spotify_library_contains", "spotify_library_remove"],
        [verify, removed],
    )
    assert reply.startswith("Removed")


def test_t11_strips_offset_pagination_hint() -> None:
    saved = json.dumps(
        {
            "ok": True,
            "user_message": "1. Album — Artist\n(More saved albums available — pass offset=10.)",
        }
    )
    reply = _finalize(
        "what albums do I have saved?",
        "1. Album — Artist",
        ["spotify_saved_albums"],
        [saved],
    )
    assert "offset=" not in reply.lower()


def test_multiline_preview_lines_preserved() -> None:
    preview = "1. Nine — A\n2. Ten — B"
    out = prepare_user_visible_reply(preview)
    assert "1. Nine" in out
    assert "\n2. Ten" in out


@respx.mock
def test_t2a_enforcement_rewrites_track_to_album(data_dir, signed_in_tokens) -> None:
    stale = "48YIv8aaaaaaaaaaaaaaab"
    live = "00GCAlaaaaaaaaaaaaaaab"
    track = "2bCl59aaaaaaaaaaaaaaab"
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/library/contains.*").mock(
        return_value=httpx.Response(200, json=[True])
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/albums/.*").mock(
        return_value=httpx.Response(200, json={"id": live, "name": "Live Album"})
    )
    player = {"is_playing": True, "item": {"type": "track", "album": {"id": live, "name": "Live Album"}}}
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player.*").mock(
        return_value=httpx.Response(200, json=player)
    )
    runner = SpotifyToolRunner(settings=Settings(), conversation_id="t2a")
    record_library_mutation("t2a", "album", [stale])
    runner._last_library_mutation = {"segment": "album", "ids": [stale]}
    args = enforce_tool_arguments_for_turn(
        "spotify_library_contains",
        {"uris": [f"spotify:track:{track}"]},
        user_text="Is this album saved?",
        runner=runner,
    )
    assert live in json.dumps(args)
    raw = runner.run("spotify_library_contains", args)
    runner.close()
    data = json.loads(raw)
    assert "Live Album" in data.get("user_message", "")
    assert data.get("saved_single") is True


@respx.mock
def test_t26b_it_uses_last_playlist_mutation(data_dir, signed_in_tokens) -> None:
    pl_id = "2HfFccisPxQfprhgIHM7XH"
    wrong_track = "1aVYJKaaaaaaaaaaaaaaab"
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/library/contains.*").mock(
        return_value=httpx.Response(200, json=[True])
    )
    respx.get(f"https://api.spotify.com/v1/playlists/{pl_id}").mock(
        return_value=httpx.Response(200, json={"id": pl_id, "name": "90s Rock Classics"})
    )
    runner = SpotifyToolRunner(settings=Settings(), conversation_id="t26b")
    runner._last_library_mutation = {"segment": "playlist", "ids": [pl_id]}
    args = enforce_tool_arguments_for_turn(
        "spotify_library_contains",
        {"uris": [f"spotify:track:{wrong_track}"]},
        user_text="Is it saved?",
        runner=runner,
    )
    assert pl_id in json.dumps(args)
    raw = runner.run("spotify_library_contains", args)
    runner.close()
    data = json.loads(raw)
    assert "90s Rock Classics" in data.get("user_message", "")


@respx.mock
def test_track_search_types_list_not_unknown_error(data_dir, signed_in_tokens, tmp_path) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(
            200,
            json={"tracks": {"items": [{"id": "t" * 22, "name": "Hit", "uri": f"spotify:track:{'t' * 22}"}]}},
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run(
        "spotify_search",
        {"query": "live track", "types": ["track"], "limit": 5},
    )
    runner.close()
    append_tool_trace_record(
        tmp_path,
        conversation_id="c",
        tool_name="spotify_search",
        args_summary="{}",
        outcome=tool_trace_outcome(raw),
        raw_result=raw,
    )
    row = json.loads((tmp_path / "chat_tool_traces.jsonl").read_text(encoding="utf-8").strip())
    assert row.get("outcome") == "ok"


@respx.mock
def test_gemini_sim_t2a_extra_tools_chain(data_dir, signed_in_tokens) -> None:
    """Model: search fail, library_contains(track), blocked play — reply is album yes/no."""
    settings = Settings(gemini_api_key="gemini-test-pr9-intent", agent_max_steps=4)
    live = "00GCAlaaaaaaaaaaaaaaab"
    track = "2bCl59aaaaaaaaaaaaaaab"
    stale = "48YIv8aaaaaaaaaaaaaaab"

    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player.*").mock(
        return_value=httpx.Response(
            200,
            json={"is_playing": True, "item": {"type": "track", "album": {"id": live, "name": "Live Album"}}},
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/library/contains.*").mock(
        return_value=httpx.Response(200, json=[True])
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/albums/.*").mock(
        return_value=httpx.Response(200, json={"id": live, "name": "Live Album"})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(400, json={"error": {"status": 400, "message": "Invalid type"}})
    )

    calls = {"n": 0}

    def gemini_handler(_request: httpx.Request) -> httpx.Response:
        n = calls["n"]
        calls["n"] += 1
        if n == 0:
            parts = [
                {"functionCall": {"name": "spotify_search", "args": {"query": "x", "types": "tracks"}}},
                {
                    "functionCall": {
                        "name": "spotify_library_contains",
                        "args": {"uris": [f"spotify:track:{track}"]},
                    }
                },
                {
                    "functionCall": {
                        "name": "spotify_start_resume_playback",
                        "args": {"context_uri": f"spotify:album:{stale}"},
                    }
                },
            ]
            return httpx.Response(
                200,
                json={"candidates": [{"finishReason": "STOP", "content": {"parts": parts}}]},
            )
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": "Done."}]},
                    }
                ]
            },
        )

    respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/.*").mock(
        side_effect=gemini_handler
    )

    with patch(
        "spot_backend.gemini_llm.gemini_deterministic_shortcut_reply",
        return_value=None,
    ), patch(
        "spot_backend.gemini_llm.should_send_gemini_tool_nudge",
        return_value=False,
    ):
        reply = run_chat_turn_gemini(
            "Is this album saved?",
            settings,
            conversation_id="sim-t2a",
        )
    assert "Live Album" in reply
    assert "Yes" in reply or "saved" in reply.lower()
