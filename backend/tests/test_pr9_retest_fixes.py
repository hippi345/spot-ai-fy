"""PR #9 Gemini laptop retest regressions (items 1–8)."""

from __future__ import annotations

import json

import httpx
import respx

from spot_backend.chat_messages import prepare_user_visible_reply, sanitize_raw_tool_json_in_reply
from spot_backend.llm_tool_loop import finalize_assistant_text, ToolLoopState
from spot_backend.playlist_builder_store import clear_playlist_preview, save_playlist_preview
from spot_backend.prompt_intent import refused_mutating_tool_result
from spot_backend.reply_tool_fallback import apply_tool_grounded_reply
from spot_backend.reply_tool_trace import append_tool_trace_record, tool_trace_outcome
from spot_backend.spotify_tools import SpotifyToolRunner
from spot_backend.config import Settings
from spot_backend.library_mutation_store import record_library_mutation


@respx.mock
def test_this_album_uses_playback_not_stale_mutation(data_dir, signed_in_tokens) -> None:
    stale_album = "48YIv8aaaaaaaaaaaaaaab"
    live_album = "00GCAlaaaaaaaaaaaaaaab"
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/library/contains.*").mock(
        return_value=httpx.Response(200, json=[True])
    )
    player_json = {
        "is_playing": True,
        "item": {
            "type": "track",
            "album": {"id": live_album},
        },
    }
    respx.get(
        "https://api.spotify.com/v1/me/player",
        params={"additional_types": "episode"},
    ).mock(return_value=httpx.Response(200, json=player_json))
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player.*").mock(
        return_value=httpx.Response(200, json=player_json)
    )
    runner = SpotifyToolRunner(settings=Settings(), conversation_id="album-pronoun")
    record_library_mutation("album-pronoun", "album", [stale_album])
    runner._last_library_mutation = {"segment": "album", "ids": [stale_album]}
    snap = runner._player_state_snapshot()
    assert snap and snap.get("item", {}).get("album", {}).get("id") == live_album
    resolved = runner._resolve_pronoun_to_uris("this album")
    assert resolved and live_album in resolved[0]
    raw = runner.run("spotify_library_contains", {"uris": ["this album"]})
    runner.close()
    data = json.loads(raw)
    assert live_album in data["uris"][0]
    assert stale_album not in data["uris"][0]


def test_library_contains_user_message_not_done() -> None:
    tool_json = json.dumps({"ok": True, "saved_single": True, "uris": ["spotify:album:x"]})
    assert "Yes" in sanitize_raw_tool_json_in_reply(tool_json)
    apology = "You are right, I apologize, I incorrectly claimed I played the song."
    saved_albums = json.dumps({"ok": True, "user_message": "1. Album A — Artist"})
    grounded = apply_tool_grounded_reply(apology, [saved_albums])
    assert "Album A" in grounded
    assert "apolog" not in grounded.lower()


def test_refused_mutating_tool_trace_reason(tmp_path) -> None:
    raw = refused_mutating_tool_result("spotify_start_resume_playback")
    data = json.loads(raw)
    assert data.get("failure_reason") == "guard_refused"
    append_tool_trace_record(
        tmp_path,
        conversation_id="c1",
        tool_name="spotify_start_resume_playback",
        args_summary="{}",
        outcome=tool_trace_outcome(raw),
        raw_result=raw,
    )
    line = (tmp_path / "chat_tool_traces.jsonl").read_text(encoding="utf-8")
    row = json.loads(line.strip())
    assert row["failure_reason"] == "guard_refused"


@respx.mock
def test_save_tracks_rejects_show_uri(data_dir, signed_in_tokens) -> None:
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_save_tracks", {"track_id": "spotify:show:3d0uC0xxxxxxxxxxxxxx"})
    runner.close()
    data = json.loads(raw)
    assert data.get("failure_reason") == "invalid_uri_type"


@respx.mock
def test_search_playlists_exact_match_prefers_name(data_dir, signed_in_tokens) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "playlists": {
                    "items": [
                        {"id": "wrong0000000000000001", "name": "Similar Rock"},
                        {"id": "2HfFccisPxQfprhgIHM7XH", "name": "90s Rock Classics"},
                    ]
                }
            },
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    data = json.loads(runner.run("spotify_search_playlists", {"query": "90s Rock Classics", "limit": 5}))
    runner.close()
    assert data["items"][0]["id"] == "2HfFccisPxQfprhgIHM7XH"
    assert data["best_match"]["id"] == "2HfFccisPxQfprhgIHM7XH"


@respx.mock
def test_builder_edit_drop_track_shrinks_preview(data_dir, signed_in_tokens) -> None:
    cid = "builder-edit"
    clear_playlist_preview(cid)
    tracks = [
        {"n": i, "uri": f"spotify:track:{i:022d}", "name": f"T{i}", "artist": "A"}
        for i in range(1, 13)
    ]
    save_playlist_preview(cid, {"proposed_name": "spot-ai-fy test", "tracks": tracks})
    runner = SpotifyToolRunner(settings=Settings(), conversation_id=cid)
    raw = runner.run("spotify_playlist_builder_edit", {"remove_indices": [3]})
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is True
    assert len(data["preview"]["tracks"]) == 11
    assert "1. T1" in data["preview_text"]
    assert "3. T4" in data["preview_text"]


def test_finalize_question_gets_yes_no_not_done() -> None:
    tool_json = json.dumps(
        {
            "ok": True,
            "saved_single": False,
            "user_message": "No — that's not in your library.",
        }
    )
    state = ToolLoopState()
    state.tool_results = [tool_json]
    state.turn_tool_calls = [("spotify_library_contains", tool_json)]
    action = finalize_assistant_text("Done.", state, user_text="is it saved?")
    assert action.kind == "return"
    assert "not in your library" in action.text.lower()


@respx.mock
def test_unfollow_editorial_blocked(data_dir, signed_in_tokens) -> None:
    editorial = "37i9dQZF1Ept8KpxrQJ0R9"
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_unfollow_playlist", {"playlist_id": editorial})
    runner.close()
    data = json.loads(raw)
    assert data.get("failure_reason") == "editorial_playlist_blocked"
