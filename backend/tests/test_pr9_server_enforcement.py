"""Server-side enforcement: wrong model args/replies corrected without prompt compliance."""

from __future__ import annotations

import json
import httpx
import pytest
import respx

from spot_backend.config import Settings
from spot_backend.llm_tool_loop import ToolLoopState, finalize_assistant_text
from spot_backend.library_mutation_store import record_library_mutation
from spot_backend.reply_tool_fallback import apply_tool_grounded_reply, humanize_failure_reason
from spot_backend.reply_tool_trace import append_tool_trace_record, tool_trace_outcome
from spot_backend.spotify_tools import SpotifyToolRunner
from spot_backend.tool_server_enforcement import enforce_tool_arguments_for_turn


@respx.mock
def test_enforce_stale_album_id_on_this_album_question(data_dir, signed_in_tokens) -> None:
    stale = "48YIv8aaaaaaaaaaaaaaab"
    live = "00GCAlaaaaaaaaaaaaaaab"
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/library/contains.*").mock(
        return_value=httpx.Response(200, json=[True])
    )
    player = {
        "is_playing": True,
        "item": {"type": "track", "album": {"id": live}},
    }
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player.*").mock(
        return_value=httpx.Response(200, json=player)
    )
    runner = SpotifyToolRunner(settings=Settings(), conversation_id="enf-album")
    record_library_mutation("enf-album", "album", [stale])
    runner._last_library_mutation = {"segment": "album", "ids": [stale]}
    args = enforce_tool_arguments_for_turn(
        "spotify_library_contains",
        {"uris": [f"spotify:album:{stale}"]},
        user_text="do I already have this album saved?",
        runner=runner,
    )
    assert live in json.dumps(args)
    raw = runner.run("spotify_library_contains", args)
    runner.close()
    data = json.loads(raw)
    assert live in data["uris"][0]


def test_failure_boilerplate_replaced_by_saved_albums_summary() -> None:
    saved = json.dumps({"ok": True, "user_message": "1. Album A — Artist"})
    blocked = json.dumps({"ok": False, "failure_reason": "guard_refused", "error": "refused"})
    state = ToolLoopState()
    state.tool_results = [saved, blocked]
    state.turn_tool_calls = [("spotify_saved_albums", saved), ("spotify_start_resume_playback", blocked)]
    state.successful_tools.add("spotify_saved_albums")
    honest = (
        "I wasn't able to run the Spotify action that turn, so I can't confirm anything changed. "
        "Please try again or rephrase the request."
    )
    action = finalize_assistant_text(honest, state, user_text="what albums do I have saved?")
    assert action.kind == "return"
    assert "Album A" in action.text
    assert "wasn't able" not in action.text.lower()


def test_builder_edit_success_not_reported_as_failure() -> None:
    preview = json.dumps(
        {
            "ok": True,
            "preview_text": "Proposed playlist \"spot-ai-fy test\" (11 tracks):\n1. T1 — A",
        }
    )
    grounded = apply_tool_grounded_reply(
        "I wasn't able to run the Spotify action that turn, so I can't confirm anything changed.",
        [preview],
        user_text="drop track 3",
    )
    assert "1. T1" in grounded
    assert "wasn't able" not in grounded.lower()


@respx.mock
def test_save_albums_show_id_invalid_uri_type(data_dir, signed_in_tokens) -> None:
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_save_albums", {"album_id": "spotify:show:4uJ4S0xxxxxxxxxxxxxx"})
    runner.close()
    data = json.loads(raw)
    assert data.get("failure_reason") == "invalid_uri_type"


@respx.mock
def test_show_search_trace_not_unknown_error(data_dir, signed_in_tokens, tmp_path) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(
            400,
            json={"error": {"status": 400, "message": "Invalid market"}},
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_search", {"query": "astronomy podcast", "types": "show", "limit": 5})
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
    assert row.get("failure_reason") != "unknown_error"
    assert row.get("spotify_error_body_redacted")


def test_playback_not_verified_humanized() -> None:
    msg = humanize_failure_reason("playback_not_verified", "Spotify lookup failed (playback_not_verified)")
    assert "playback_not_verified" not in msg
    assert "confirm" in msg.lower()


@respx.mock
def test_search_playlists_limit_clamped_in_enforcement(data_dir, signed_in_tokens) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(200, json={"playlists": {"items": []}})
    )
    runner = SpotifyToolRunner(settings=Settings())
    args = enforce_tool_arguments_for_turn(
        "spotify_search_playlists",
        {"query": "90s Rock Classics", "limit": 1},
        user_text="save the playlist 90s Rock Classics",
        runner=runner,
    )
    assert args.get("limit", 0) >= 5
    runner.close()


def test_gemini_module_uses_shared_finalize() -> None:
    import spot_backend.gemini_llm as gem

    assert "finalize_assistant_text" in gem.run_chat_turn_gemini.__code__.co_names
