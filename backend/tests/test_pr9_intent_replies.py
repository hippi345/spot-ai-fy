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
    assert calls["n"] == 1


@respx.mock
def test_t26c_unfollow_wrong_id_overrides_to_last_followed_playlist(data_dir, signed_in_tokens) -> None:
    followed = "2HfFccisPxQfprhgIHM7XH"
    owned_wrong = "2hIbTEmTOz1XrjUxSOcoq4"
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": "meuser0000000000000001"})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/library.*").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/library/contains.*").mock(
        return_value=httpx.Response(200, json=[False])
    )
    respx.delete(url__regex=r"https://api\.spotify\.com/v1/me/library.*").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.get(f"https://api.spotify.com/v1/playlists/{followed}").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": followed,
                "name": "90s Rock Classics",
                "owner": {"id": "otheruser00000000000001"},
            },
        )
    )
    runner = SpotifyToolRunner(settings=Settings(), conversation_id="t26c")
    runner._record_library_mutation("playlist", [followed])
    args = enforce_tool_arguments_for_turn(
        "spotify_unfollow_playlist",
        {"playlist_id": owned_wrong},
        user_text="Remove it",
        runner=runner,
    )
    assert args["playlist_id"] == followed
    raw = runner.run("spotify_unfollow_playlist", args)
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is True
    assert "90s Rock Classics" in data.get("user_message", "")


@respx.mock
def test_t26c_owned_playlist_unfollow_refused_no_delete(data_dir, signed_in_tokens) -> None:
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
    delete_calls = 0

    def _delete_library(request: httpx.Request) -> httpx.Response:
        nonlocal delete_calls
        delete_calls += 1
        return httpx.Response(200, json={})

    respx.delete(url__regex=r"https://api\.spotify\.com/v1/me/library.*").mock(
        side_effect=_delete_library
    )
    runner = SpotifyToolRunner(settings=Settings(), conversation_id="t26c-owned")
    raw = runner.run(
        "spotify_unfollow_playlist",
        {"playlist_id": owned, "_turn_user_text": "Remove it"},
    )
    runner.close()
    data = json.loads(raw)
    assert data.get("failure_reason") == "owned_playlist_protected"
    assert data.get("needs_confirmation") is True
    assert delete_calls == 0


@respx.mock
def test_t23_save_this_show_uses_last_show_not_track_playback(data_dir, signed_in_tokens) -> None:
    show_id = "1mNsuXbbbbbbbbbbbbbbbb"
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "item": {"type": "track", "id": "t" * 22, "name": "Some Song"},
            },
        )
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
    runner = SpotifyToolRunner(settings=Settings(), conversation_id="t23")
    runner._record_library_mutation("show", [show_id])
    args = enforce_tool_arguments_for_turn(
        "spotify_library_save",
        {"uris": ["spotify:show:4s0y8C1xxxxxxxxxxxxxx"]},
        user_text="Save this show",
        runner=runner,
    )
    assert show_id in json.dumps(args)
    raw = runner.run("spotify_library_save", args)
    runner.close()
    data = json.loads(raw)
    assert "StarTalk" in data.get("user_message", "")


@respx.mock
def test_builder_nineties_search_q_and_filters_off_decade(data_dir, signed_in_tokens) -> None:
    captured_q: list[str] = []
    call_n = 0

    def search_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_n
        captured_q.append(str(request.url.params.get("q")))
        call_n += 1
        good_id = f"{call_n:022d}"
        bad_id = f"{100 + call_n:022d}"
        return httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [
                        {
                            "id": good_id,
                            "uri": f"spotify:track:{good_id}",
                            "name": f"90s Hit {call_n}",
                            "artists": [{"name": "Band"}],
                            "album": {"name": "Album", "release_date": "1995-06-01"},
                        },
                        {
                            "id": bad_id,
                            "uri": f"spotify:track:{bad_id}",
                            "name": f"2000s Hit {call_n}",
                            "artists": [{"name": "Band"}],
                            "album": {"name": "Later", "release_date": "2003-01-01"},
                        },
                    ]
                }
            },
        )

    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(side_effect=search_handler)
    runner = SpotifyToolRunner(settings=Settings(), conversation_id="90s-theme")
    seeds = [f"rock anthem {i}" for i in range(12)]
    raw = runner.run(
        "spotify_playlist_builder_preview",
        {
            "name": "90s mix",
            "theme": "90s rock",
            "track_queries": seeds,
        },
    )
    runner.close()
    data = json.loads(raw)
    assert captured_q
    assert any("year:1990-1999" in q for q in captured_q)
    assert data.get("ok") is True
    preview = data.get("preview", {}).get("tracks", [])
    assert preview
    assert all("2000s" not in str(t.get("name") or "") for t in preview if isinstance(t, dict))


def test_t15_privacy_warning_once_in_finalize() -> None:
    note = (
        "Spotify still shows this playlist as public. To make it private, open it in the Spotify app."
    )
    tool_json = json.dumps(
        {
            "ok": True,
            "privacy_warning": note,
            "user_message": f'Created playlist "Mix" with 10 tracks. {note}',
        }
    )
    reply = _finalize(
        "make it",
        f"Created your playlist. {note}",
        ["spotify_playlist_builder_commit"],
        [tool_json],
    )
    assert reply.count("still shows") == 1 or reply.lower().count("public") <= 2
    assert reply.count(note) <= 1


def test_reply_spotify_only_replaced_by_tool_user_message() -> None:
    play_fail = json.dumps(
        {
            "ok": False,
            "failure_reason": "show_not_found",
            "user_message": "I couldn't find that show's latest episode.",
        }
    )
    reply = prepare_user_visible_reply(
        "spotify_play_show_latest_episode",
        [play_fail],
        tool_names=["spotify_play_show_latest_episode"],
        user_text="Play the latest episode of StarTalk",
    )
    assert reply == "I couldn't find that show's latest episode."
    assert reply.lower() != "spotify"


@respx.mock
def test_t21_invented_show_id_overridden_to_session_startalk(data_dir, signed_in_tokens) -> None:
    real_show = "1mNsuXbbbbbbbbbbbbbbbb"
    invented = "2rD20sDja2xP5t890C36gL"
    runner = SpotifyToolRunner(settings=Settings(), conversation_id="t21")
    runner.note_session_show(real_show, "StarTalk with Neil deGrasse Tyson")
    args = enforce_tool_arguments_for_turn(
        "spotify_play_show_latest_episode",
        {"show_id": invented},
        user_text="Play the latest episode of StarTalk",
        runner=runner,
    )
    assert args["show_id"] == real_show
    runner.close()


@respx.mock
def test_t25_remove_unsaved_show_skips_delete(data_dir, signed_in_tokens) -> None:
    show_id = "1mNsuXbbbbbbbbbbbbbbbb"
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/library/contains.*").mock(
        return_value=httpx.Response(200, json=[False])
    )
    respx.get(f"https://api.spotify.com/v1/shows/{show_id}").mock(
        return_value=httpx.Response(200, json={"id": show_id, "name": "StarTalk"})
    )
    delete_calls = 0

    def _delete(_request: httpx.Request) -> httpx.Response:
        nonlocal delete_calls
        delete_calls += 1
        return httpx.Response(200, json={})

    respx.delete(url__regex=r"https://api\.spotify\.com/v1/me/library.*").mock(side_effect=_delete)
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_library_remove", {"uris": [f"spotify:show:{show_id}"]})
    runner.close()
    data = json.loads(raw)
    assert delete_calls == 0
    assert "wasn't in your library" in data.get("user_message", "")


@respx.mock
def test_saved_shows_list_no_dangling_em_dash(data_dir, signed_in_tokens) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/shows.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {"show": {"id": "s" * 22, "name": "StarTalk", "publisher": ""}},
                ],
                "total": 1,
            },
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_user_saved_shows", {})
    runner.close()
    data = json.loads(raw)
    line = data.get("summary_lines", [""])[0]
    assert line == "1. StarTalk"
    assert not line.endswith("—")


@respx.mock
def test_chill_90s_music_theme_search_and_filter(data_dir, signed_in_tokens) -> None:
    captured_q: list[str] = []
    call_n = 0

    def search_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_n
        captured_q.append(str(request.url.params.get("q")))
        call_n += 1
        good_id = f"{call_n:022d}"
        bad_id = f"{100 + call_n:022d}"
        return httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [
                        {
                            "id": good_id,
                            "uri": f"spotify:track:{good_id}",
                            "name": f"Chill {call_n}",
                            "artists": [{"name": "Band"}],
                            "album": {"name": "Album", "release_date": "1994-03-01"},
                        },
                        {
                            "id": bad_id,
                            "uri": f"spotify:track:{bad_id}",
                            "name": f"Modern {call_n}",
                            "artists": [{"name": "Band"}],
                            "album": {"name": "Later", "release_date": "2008-03-01"},
                        },
                    ]
                }
            },
        )

    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(side_effect=search_handler)
    runner = SpotifyToolRunner(settings=Settings(), conversation_id="chill-90s")
    raw = runner.run(
        "spotify_playlist_builder_preview",
        {"name": "mix", "theme": "chill 90s music", "track_queries": ["chill 90s music"]},
    )
    runner.close()
    data = json.loads(raw)
    assert any("year:1990-1999" in q for q in captured_q)
    preview = data.get("preview", {}).get("tracks", [])
    assert preview
    assert all(
        (t.get("release_date") or "")[:4].isdigit() and 1990 <= int((t.get("release_date") or "1990")[:4]) <= 1999
        for t in preview
        if isinstance(t, dict) and t.get("release_date")
    )


def test_trace_args_strip_turn_user_text(tmp_path) -> None:
    from spot_backend.reply_tool_trace import append_tool_trace_record, summarize_tool_args

    summary = summarize_tool_args(
        {"show_id": "abc", "_turn_user_text": "Save this show please with secret details"}
    )
    parsed = json.loads(summary)
    assert "_turn_user_text" not in parsed
    assert parsed.get("_turn_user_text_len") == len("Save this show please with secret details")
    assert "secret" not in summary
    append_tool_trace_record(
        tmp_path,
        conversation_id="c",
        tool_name="spotify_library_save",
        args_summary=summary,
        outcome="ok",
        raw_result='{"ok": true}',
    )
    row = json.loads((tmp_path / "chat_tool_traces.jsonl").read_text(encoding="utf-8").strip())
    traced_args = json.loads(row["args"])
    assert "_turn_user_text" not in traced_args


def test_numbered_list_breaks_after_scrub_glued_model_echo() -> None:
    glued = "Proposed playlist \"x\" (11 tracks):\n" + " ".join(
        f"{i}. Track{i} — Artist" for i in range(1, 12)
    )
    reply = prepare_user_visible_reply(glued)
    for i in range(1, 12):
        assert f"\n{i}. Track{i}" in reply or reply.strip().startswith(f"{i}. Track{i}")


@respx.mock
def test_gemini_t23_save_this_show_after_list(data_dir, signed_in_tokens) -> None:
    show_id = "1mNsuXbbbbbbbbbbbbbbbb"
    settings = Settings(gemini_api_key="gemini-test-t23", agent_max_steps=3)
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player.*").mock(
        return_value=httpx.Response(
            200,
            json={"is_playing": True, "item": {"type": "track", "name": "Song", "artists": [{"name": "A"}]}},
        )
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

    def gemini_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "spotify_playback_state",
                                        "args": {},
                                    }
                                }
                            ]
                        },
                    }
                ]
            },
        )

    respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/.*").mock(
        side_effect=gemini_handler
    )
    record_library_mutation("t23-seq", "show", [show_id])
    with patch(
        "spot_backend.gemini_llm.gemini_deterministic_shortcut_reply",
        return_value=None,
    ), patch(
        "spot_backend.gemini_llm.should_send_gemini_tool_nudge",
        return_value=False,
    ):
        reply = run_chat_turn_gemini(
            "Save this show",
            settings,
            conversation_id="t23-seq",
        )
    assert "Saved StarTalk" in reply


def test_builder_preview_newlines_through_finalize() -> None:
    lines = "\n".join([f"{i}. Track{i} — Artist" for i in range(1, 12)])
    preview = json.dumps({"ok": True, "preview_text": f'Proposed playlist "x" (11 tracks):\n{lines}'})
    reply = _finalize("yes make it", "1. Track1 — Artist " + " ".join(f"{i}. Track{i} — Artist" for i in range(2, 12)), ["spotify_playlist_builder_preview"], [preview])
    assert "\n9." in reply or "\n10." in reply
    assert "9. Track9" in reply
    assert "10. Track10" in reply
