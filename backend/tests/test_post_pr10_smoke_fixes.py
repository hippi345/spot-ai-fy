"""Post-PR #10 smoke fixes: owned playlist edits, audiobooks, session resolve, playback."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import httpx
import pytest
import respx

from spot_backend.chat_messages import prepare_user_visible_reply
from spot_backend.config import Settings
from spot_backend.llm_tool_loop import ToolLoopState, finalize_assistant_text, rewrite_tool_call_for_turn
from spot_backend.reply_tool_fallback import apply_tool_grounded_reply, reply_hallucinates_after_tool_failure
from spot_backend.reply_tool_trace import append_tool_trace_record, summarize_tool_args, tool_trace_outcome
from spot_backend.spotify_tools import SpotifyToolRunner, _track_has_primary_artist
from spot_backend.tool_server_enforcement import enforce_tool_arguments_for_turn


def _finalize(user: str, model: str, names: list[str], results: list[str]) -> str:
    state = ToolLoopState()
    state.tool_results = list(results)
    state.turn_tool_calls = list(zip(names, results))
    for n, r in state.turn_tool_calls:
        parsed = json.loads(r)
        if parsed.get("ok") is not False and parsed.get("failure_reason") != "guard_refused":
            state.successful_tools.add(n)
    action = finalize_assistant_text(model, state, user_text=user)
    assert action.kind == "return"
    return prepare_user_visible_reply(action.text, state.tool_results)


@respx.mock
def test_t18_remove_track_from_owned_builder_playlist(data_dir, signed_in_tokens) -> None:
    owned = "2hIbTEmTOz1XrjUxSOcoq4"
    me_id = "meuser0000000000000001"
    track_id = "t" * 22
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": me_id})
    )
    respx.get(f"https://api.spotify.com/v1/playlists/{owned}").mock(
        return_value=httpx.Response(
            200,
            json={"id": owned, "name": "spot-ai-fy test", "owner": {"id": me_id}},
        )
    )
    respx.delete(f"https://api.spotify.com/v1/playlists/{owned}/items").mock(
        return_value=httpx.Response(200, json={"snapshot_id": "snap"})
    )
    runner = SpotifyToolRunner(settings=Settings(), conversation_id="t18")
    runner.note_session_playlist_id(owned)
    args = enforce_tool_arguments_for_turn(
        "spotify_remove_playlist_tracks",
        {"playlist_id": owned, "track_uris": [f"spotify:track:{track_id}"]},
        user_text="Remove it from that playlist",
        runner=runner,
    )
    raw = runner.run("spotify_remove_playlist_tracks", args)
    runner.close()
    data = json.loads(raw)
    assert data.get("failure_reason") != "owned_playlist_protected"
    assert "snapshot_id" in json.dumps(data) or data.get("ok") is not False


@respx.mock
def test_t19_rename_owned_builder_playlist(data_dir, signed_in_tokens) -> None:
    owned = "2hIbTEmTOz1XrjUxSOcoq4"
    me_id = "meuser0000000000000001"
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": me_id})
    )
    respx.get(f"https://api.spotify.com/v1/playlists/{owned}").mock(
        return_value=httpx.Response(
            200,
            json={"id": owned, "name": "spot-ai-fy test", "owner": {"id": me_id}},
        )
    )
    respx.put(f"https://api.spotify.com/v1/playlists/{owned}").mock(
        return_value=httpx.Response(200, json={"id": owned})
    )
    runner = SpotifyToolRunner(settings=Settings(), conversation_id="t19")
    runner.note_session_playlist_id(owned)
    args = enforce_tool_arguments_for_turn(
        "spotify_update_playlist",
        {"playlist_id": "it", "name": "spot-ai-fy test 2"},
        user_text="Rename it to spot-ai-fy test 2",
        runner=runner,
    )
    raw = runner.run("spotify_update_playlist", args)
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is True
    assert data.get("failure_reason") != "owned_playlist_protected"


@respx.mock
def test_owned_unfollow_still_protected(data_dir, signed_in_tokens) -> None:
    owned = "2hIbTEmTOz1XrjUxSOcoq4"
    me_id = "meuser0000000000000001"
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": me_id})
    )
    respx.get(f"https://api.spotify.com/v1/playlists/{owned}").mock(
        return_value=httpx.Response(
            200,
            json={"id": owned, "name": "spot-ai-fy test", "owner": {"id": me_id}},
        )
    )
    runner = SpotifyToolRunner(settings=Settings(), conversation_id="unfollow-owned")
    raw = runner.run(
        "spotify_unfollow_playlist",
        {"playlist_id": owned, "_turn_user_text": "remove my new test playlist"},
    )
    runner.close()
    data = json.loads(raw)
    assert data.get("failure_reason") == "owned_playlist_protected"
    assert "remove spot-ai-fy test" not in data.get("error", "").lower()


@respx.mock
def test_audiobook_search_failure_not_unknown_error(data_dir, signed_in_tokens, tmp_path) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(
            400,
            json={"error": {"status": 400, "message": "Invalid market for audiobook"}},
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run(
        "spotify_search",
        {"query": "Brandon Sanderson", "types": "audiobook", "limit": 5},
    )
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is False
    assert data.get("failure_reason") == "audiobooks_unavailable_in_market"
    append_tool_trace_record(
        tmp_path,
        conversation_id="c",
        tool_name="spotify_search",
        args_summary="{}",
        outcome=tool_trace_outcome(raw),
        raw_result=raw,
    )
    row = json.loads((tmp_path / "chat_tool_traces.jsonl").read_text(encoding="utf-8").strip())
    assert row.get("failure_reason") == "audiobooks_unavailable_in_market"


def test_audiobook_hallucination_blocked() -> None:
    fail = json.dumps(
        {
            "ok": False,
            "failure_reason": "audiobooks_unavailable_in_market",
            "error": "Invalid market",
        }
    )
    model = "1. Mistborn — narrated by Michael Kramer"
    assert reply_hallucinates_after_tool_failure(model, [fail])
    grounded = apply_tool_grounded_reply(model, [fail], user_text="Find an audiobook by Brandon Sanderson")
    assert "mistborn" not in grounded.lower()
    assert "narrated" not in grounded.lower()


@respx.mock
def test_t23_rewrite_save_tracks_fake_id_to_show(data_dir, signed_in_tokens) -> None:
    show_id = "1mNsuXbbbbbbbbbbbbbbbb"
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player.*").mock(
        return_value=httpx.Response(204)
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/library.*").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/library/contains.*").mock(
        return_value=httpx.Response(200, json=[True])
    )
    respx.get(f"https://api.spotify.com/v1/shows/{show_id}").mock(
        return_value=httpx.Response(200, json={"id": show_id, "name": "StarTalk"})
    )
    runner = SpotifyToolRunner(settings=Settings(), conversation_id="t23-rewrite")
    runner._record_library_mutation("show", [show_id])
    name, args = rewrite_tool_call_for_turn(
        "spotify_save_tracks",
        {"uris": [f"spotify:track:{'f' * 22}"]},
        user_text="Save this show",
        runner=runner,
    )
    assert name == "spotify_library_save"
    args = enforce_tool_arguments_for_turn(name, args, user_text="Save this show", runner=runner)
    raw = runner.run(name, args)
    runner.close()
    data = json.loads(raw)
    assert "StarTalk" in data.get("user_message", "")


@respx.mock
def test_t27_unfollow_invented_playlist_id_resolves_session(data_dir, signed_in_tokens) -> None:
    owned = "2hIbTEmTOz1XrjUxSOcoq4"
    fake = "x" * 22
    me_id = "meuser0000000000000001"
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": me_id})
    )
    respx.get(f"https://api.spotify.com/v1/playlists/{owned}").mock(
        return_value=httpx.Response(
            200,
            json={"id": owned, "name": "spot-ai-fy test", "owner": {"id": me_id}},
        )
    )
    runner = SpotifyToolRunner(settings=Settings(), conversation_id="t27")
    runner.note_session_playlist_id(owned)
    args = enforce_tool_arguments_for_turn(
        "spotify_unfollow_playlist",
        {"playlist_id": fake},
        user_text="Remove it",
        runner=runner,
    )
    assert args.get("playlist_id") == owned
    raw = runner.run("spotify_unfollow_playlist", args)
    runner.close()
    data = json.loads(raw)
    assert data.get("failure_reason") == "owned_playlist_protected"


def test_play_artist_prefers_primary_credit() -> None:
    weeknd = "weekndartistid00000001"
    remix = {
        "id": "r" * 22,
        "uri": f"spotify:track:{'r' * 22}",
        "name": "Remix",
        "popularity": 99,
        "artists": [
            {"id": "other0000000000000001", "name": "DJ"},
            {"id": weeknd, "name": "The Weeknd"},
        ],
    }
    original = {
        "id": "o" * 22,
        "uri": f"spotify:track:{'o' * 22}",
        "name": "Blinding Lights",
        "popularity": 90,
        "artists": [{"id": weeknd, "name": "The Weeknd"}],
    }
    assert not _track_has_primary_artist(remix, artist_id=weeknd, artist_name="The Weeknd")
    assert _track_has_primary_artist(original, artist_id=weeknd, artist_name="The Weeknd")


def test_t21_still_playing_after_playback_not_verified() -> None:
    from spot_backend.turn_reply_intent import _play_failure_user_message

    msg = _play_failure_user_message(
        {
            "ok": False,
            "failure_reason": "playback_not_verified",
            "player_after": {
                "item": {
                    "type": "track",
                    "name": "Old Song",
                    "artists": [{"name": "Old Artist"}],
                }
            },
        }
    )
    assert "still playing Old Song by Old Artist" in msg


def test_trace_args_utf8_emoji_safe() -> None:
    summary = summarize_tool_args(
        {"query": "café 🎵" + "é" * 120},
        max_len=80,
    )
    assert "\ufffd" not in summary
    assert "…" in summary
    summary.encode("utf-8")


@respx.mock
@respx.mock
def test_t23_real_sequence_invented_show_id_not_saved(data_dir, signed_in_tokens) -> None:
    """Reproduce laptop T23: invented show id must not yield a false Saved reply."""
    session_show = "1mNsuXbbbbbbbbbbbbbbbb"
    invented = "4o8R8J8jPqRj2m9L7m4L4M"
    invented_uri = f"spotify:show:{invented}"
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get(f"https://api.spotify.com/v1/shows/{session_show}").mock(
        return_value=httpx.Response(200, json={"id": session_show, "name": "StarTalk"})
    )
    respx.get(f"https://api.spotify.com/v1/shows/{invented}").mock(
        return_value=httpx.Response(404, json={"error": {"status": 404, "message": "Not found"}})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/library.*").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/library/contains.*").mock(
        return_value=httpx.Response(200, json=[False])
    )
    runner = SpotifyToolRunner(settings=Settings(), conversation_id="t23-seq")
    runner.note_session_show(session_show, "StarTalk")
    user = "Save this show"
    name1, args1 = rewrite_tool_call_for_turn(
        "spotify_save_tracks",
        {"track_ids": [invented_uri]},
        user_text=user,
        runner=runner,
    )
    assert name1 == "spotify_library_save"
    args1 = enforce_tool_arguments_for_turn(name1, args1, user_text=user, runner=runner)
    raw1 = runner.run(name1, args1)
    data1 = json.loads(raw1)
    assert session_show in json.dumps(data1)
    args2 = enforce_tool_arguments_for_turn(
        "spotify_library_save",
        {"uris": ["this show"]},
        user_text=user,
        runner=runner,
    )
    assert session_show in json.dumps(args2)
    raw2 = runner.run(
        "spotify_library_save",
        enforce_tool_arguments_for_turn(
            "spotify_library_save",
            {"uris": [invented_uri]},
            user_text=user,
            runner=runner,
        ),
    )
    data2 = json.loads(raw2)
    assert data2.get("ok") is not True
    assert data2.get("failure_reason") in ("save_not_verified", "show_not_found")
    assert "Saved" not in (data2.get("user_message") or "")
    runner.close()


@respx.mock
def test_t21_play_show_latest_still_playing_reply(data_dir, signed_in_tokens) -> None:
    show_id = "1mNsuXbbbbbbbbbbbbbbbb"
    respx.get(f"https://api.spotify.com/v1/shows/{show_id}/episodes").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "e" * 22,
                        "uri": f"spotify:episode:{'e' * 22}",
                        "name": "Latest ep",
                    }
                ]
            },
        )
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "item": {
                    "type": "track",
                    "name": "Enjoy The Show",
                    "artists": [{"name": "The Weeknd"}],
                },
            },
        )
    )
    runner = SpotifyToolRunner(settings=Settings(), conversation_id="t21-show")
    runner.note_session_show(show_id, "StarTalk")
    raw = runner.run(
        "spotify_play_show_latest_episode",
        {"show_id": show_id, "_turn_user_text": "Play latest StarTalk"},
    )
    runner.close()
    data = json.loads(raw)
    assert data.get("failure_reason") == "playback_not_verified"
    assert "Enjoy The Show" in data.get("user_message", "")
    assert "Weeknd" in data.get("user_message", "")


def test_p1_play_artist_reply_uses_verified_player_track() -> None:
    from spot_backend.turn_reply_intent import primary_tool_user_reply

    raw = json.dumps(
        {
            "ok": True,
            "playback_verified": True,
            "user_message": "Now playing The Party & The After Party by The Weeknd.",
            "track": {"name": "Enjoy The Show", "id": "wrong" * 4},
            "player_after": {
                "item": {
                    "type": "track",
                    "name": "The Party & The After Party",
                    "artists": [{"name": "The Weeknd"}],
                }
            },
        }
    )
    reply = primary_tool_user_reply("spotify_play_artist", raw, user_text="play something by The Weeknd")
    assert reply is not None
    assert "Party" in reply
    assert "Enjoy The Show" not in reply


def test_builder_q_trace_when_debug_flag(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SPOT_DEBUG_BUILDER_Q", "1")
    settings = Settings().model_copy(update={"data_dir": tmp_path})
    runner = SpotifyToolRunner(settings=settings, conversation_id="builder-q")
    track = {
        "id": "a" * 22,
        "uri": f"spotify:track:{'a' * 22}",
        "name": "Hit",
        "artists": [{"name": "Band"}],
        "album": {"name": "Al", "release_date": "1995-01-01"},
    }
    with patch.object(
        runner.client,
        "api_get",
        return_value={"tracks": {"items": [track]}},
    ):
        runner._resolve_track_query("90s pop", "from_token", theme_blob="chill 90s")
    runner.close()
    lines = (tmp_path / "chat_tool_traces.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert any("playlist_builder_search" in ln and '"q"' in ln for ln in lines)
