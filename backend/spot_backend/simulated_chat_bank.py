"""Deterministic simulated chat bank (mocked Spotify HTTP via respx in tests)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from spot_backend.chat_shortcuts import try_deterministic_chat_reply
from spot_backend.deterministic_chat import resolve_deterministic_chat_outcome
from spot_backend.prompt_intent import prompt_is_surprise_me_request
from spot_backend.playlist_builder_store import save_playlist_preview
from spot_backend.spotify_tools import SpotifyToolRunner, OLLAMA_TOOLS


@dataclass
class SimulatedChatBankRow:
    prompt: str
    tools_called: list[str]
    reply: str
    passed: bool
    tool_args: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""
    flow: str = ""


@dataclass
class SimTurn:
    prompt: str
    script: list[tuple[str, dict[str, Any]]]
    check: Callable[[str, list[str], list[dict[str, Any]]], bool]
    setup: Callable[[SpotifyToolRunner], None] | None = None


@dataclass
class SimFlow:
    name: str
    turns: list[SimTurn]


def _tool_names_from_steps(steps: list[tuple[str, dict, str]]) -> list[str]:
    return [name for name, _, _ in steps]


def _run_shortcut(prompt: str, runner: SpotifyToolRunner, conversation_id: str) -> SimulatedChatBankRow:
    outcome = resolve_deterministic_chat_outcome(prompt, runner, conversation_id=conversation_id)
    if outcome is None:
        outcome = try_deterministic_chat_reply(prompt, runner, conversation_id=conversation_id)
    if outcome is None:
        return SimulatedChatBankRow(prompt, [], "", False, note="no deterministic handler")
    tools = outcome.tool_names()
    reply = (outcome.reply or "").strip()
    return SimulatedChatBankRow(prompt, tools, reply, bool(reply), [])


def _run_tool_script(
    prompt: str,
    runner: SpotifyToolRunner,
    script: list[tuple[str, dict[str, Any]]],
    *,
    pass_check: Callable[[str, list[str], list[dict[str, Any]]], bool],
    flow: str = "",
    setup: Callable[[SpotifyToolRunner], None] | None = None,
) -> SimulatedChatBankRow:
    if setup:
        setup(runner)
    tools: list[str] = []
    arg_rows: list[dict[str, Any]] = []
    last_raw = ""
    for name, args in script:
        tools.append(name)
        arg_rows.append(dict(args))
        last_raw = runner.run(name, args)
    data = json.loads(last_raw) if last_raw else {}
    reply = str(data.get("user_message") or data.get("message") or data.get("preview_text") or "")
    if not reply and data.get("error"):
        reply = str(data.get("error"))
    if not reply and data.get("ok"):
        reply = "ok"
    passed = pass_check(reply, tools, arg_rows)
    return SimulatedChatBankRow(prompt, tools, reply, passed, arg_rows, "", flow)


def _pr9_retest_flows(
    show_id: str,
    episode_id: str,
    track_id: str,
    album_id: str,
    stale_album_id: str,
    live_album_id: str,
) -> list[SimFlow]:
    pl_exact = "2HfFccisPxQfprhgIHM7XH"
    return [
        SimFlow(
            "pr9_this_album_playback_resolution",
            [
                SimTurn(
                    "do I already have this album saved?",
                    [("spotify_library_contains", {"uris": ["this album"]})],
                    lambda reply, tools, _args: tools == ["spotify_library_contains"]
                    and ("yes" in reply.lower() or "no" in reply.lower() or "saved" in reply.lower()),
                    setup=lambda r, live=live_album_id, stale=stale_album_id: (
                        setattr(r, "_last_library_mutation", {"segment": "album", "ids": [stale]}),
                        setattr(
                            r,
                            "_playback_catalog_id",
                            lambda segment, lid=live: lid if segment == "album" else None,
                        ),
                    ),
                ),
            ],
        ),
        SimFlow(
            "pr9_no_hallucination_after_search_error",
            [
                SimTurn(
                    "find podcasts about astronomy",
                    [("spotify_search", {"query": "astronomy podcast", "types": "show", "limit": 5})],
                    lambda reply, tools, _a: "spotify_search" in tools,
                ),
            ],
        ),
        SimFlow(
            "pr9_saved_albums_not_apology",
            [
                SimTurn(
                    "what albums do I have saved?",
                    [("spotify_saved_albums", {"limit": 5})],
                    lambda reply, tools, _a: tools == ["spotify_saved_albums"]
                    and "apolog" not in reply.lower(),
                ),
            ],
        ),
        SimFlow(
            "pr9_builder_drop_track_edit",
            [
                SimTurn(
                    "drop track 3",
                    [("spotify_playlist_builder_edit", {"remove_indices": [3]})],
                    lambda reply, tools, _a: tools == ["spotify_playlist_builder_edit"]
                    and "2." in reply
                    and "4." in reply,
                    setup=lambda r: save_playlist_preview(
                        r.conversation_id or "chat-bank",
                        {
                            "proposed_name": "spot-ai-fy test",
                            "tracks": [
                                {
                                    "n": i,
                                    "uri": f"spotify:track:{i:022d}",
                                    "name": f"Track {i}",
                                    "artist": "A",
                                }
                                for i in range(1, 13)
                            ],
                        },
                    ),
                ),
            ],
        ),
        SimFlow(
            "pr9_playlist_exact_match",
            [
                SimTurn(
                    "save playlist 90s Rock Classics",
                    [
                        (
                            "spotify_search_playlists",
                            {"query": "90s Rock Classics", "limit": 5},
                        ),
                        ("spotify_follow_playlist", {"playlist_id": pl_exact}),
                    ],
                    lambda _r, tools, args: tools[0] == "spotify_search_playlists"
                    and args[1].get("playlist_id") == pl_exact,
                ),
            ],
        ),
        SimFlow(
            "pr9_is_it_saved_yes_no",
            [
                SimTurn(
                    "is it saved?",
                    [("spotify_library_contains", {"uris": [f"spotify:track:{track_id}"]})],
                    lambda reply, tools, _a: tools == ["spotify_library_contains"]
                    and ("yes" in reply.lower() or "no" in reply.lower() or "saved" in reply.lower()),
                ),
            ],
        ),
        SimFlow(
            "pr9_save_show_rejects_save_tracks",
            [
                SimTurn(
                    "save this show",
                    [("spotify_save_tracks", {"track_id": f"spotify:show:{show_id}"})],
                    lambda reply, tools, _a: tools == ["spotify_save_tracks"]
                    and ("invalid_uri_type" in reply or "only accepts track" in reply.lower()),
                ),
            ],
        ),
    ]


def _podcast_flows(show_id: str, episode_id: str, track_id: str, album_id: str) -> list[SimFlow]:
    pl_saved = "2HfFccisPxQfprhgIHM7XH"
    pl_created = "newpl0000000000000001"
    return [
        SimFlow(
            "podcast_astronomy_to_remove",
            [
                SimTurn(
                    "find podcasts about astronomy",
                    [("spotify_search", {"query": "astronomy podcast", "types": "show", "limit": 5})],
                    lambda _r, tools, args: "spotify_search" in tools
                    and args[0].get("types") == "show",
                ),
                SimTurn(
                    "play the latest episode of that show",
                    [("spotify_play_show_latest_episode", {"show_id": show_id})],
                    lambda _r, tools, _a: tools == ["spotify_play_show_latest_episode"],
                ),
                SimTurn(
                    "what podcasts do I follow?",
                    [("spotify_user_saved_shows", {"limit": 10})],
                    lambda _r, tools, _a: tools == ["spotify_user_saved_shows"],
                ),
                SimTurn(
                    "save this show",
                    [("spotify_library_save", {"uris": [f"spotify:show:{show_id}"]})],
                    lambda _r, tools, args: tools == ["spotify_library_save"]
                    and f"spotify:show:{show_id}" in (args[0].get("uris") or []),
                    setup=lambda r: r._record_library_mutation("show", [show_id]),
                ),
                SimTurn(
                    "is this show saved?",
                    [("spotify_library_contains", {"uris": [f"spotify:show:{show_id}"]})],
                    lambda _r, tools, args: tools == ["spotify_library_contains"],
                ),
                SimTurn(
                    "remove it from my library",
                    [("spotify_library_remove", {"uri": "it"})],
                    lambda _r, tools, args: tools == ["spotify_library_remove"],
                    setup=lambda r: r._record_library_mutation("show", [show_id]),
                ),
            ],
        ),
        SimFlow(
            "library_contains_track_album",
            [
                SimTurn(
                    "is this song in my likes?",
                    [
                        (
                            "spotify_library_contains",
                            {"uris": [f"spotify:track:{track_id}"]},
                        )
                    ],
                    lambda _r, tools, _a: tools == ["spotify_library_contains"],
                ),
                SimTurn(
                    "do I already have this album saved?",
                    [
                        (
                            "spotify_library_contains",
                            {"uris": [f"spotify:album:{album_id}"]},
                        )
                    ],
                    lambda _r, tools, _a: tools == ["spotify_library_contains"],
                ),
                SimTurn(
                    "what albums do I have saved?",
                    [("spotify_saved_albums", {"limit": 5})],
                    lambda reply, tools, _a: tools == ["spotify_saved_albums"]
                    and "user_message" not in reply
                    and len(reply) < 500,
                ),
            ],
        ),
        SimFlow(
            "playlist_builder_rename",
            [
                SimTurn(
                    "build a chill 90s playlist called spot-ai-fy test",
                    [
                        (
                            "spotify_playlist_builder_preview",
                            {
                                "name": "spot-ai-fy test",
                                "track_queries": ["1990s chill"],
                                "theme": "chill 90s",
                            },
                        )
                    ],
                    lambda reply, tools, _a: tools == ["spotify_playlist_builder_preview"]
                    and "spotify_create_playlist" not in tools,
                ),
                SimTurn(
                    "yes make it",
                    [("spotify_playlist_builder_commit", {"approve": True, "name": "spot-ai-fy test"})],
                    lambda _r, tools, _a: tools == ["spotify_playlist_builder_commit"],
                    setup=lambda r: None,
                ),
                SimTurn(
                    "rename it to spot-ai-fy test renamed",
                    [
                        (
                            "spotify_update_playlist",
                            {"playlist_id": "it", "name": "spot-ai-fy test renamed"},
                        )
                    ],
                    lambda _r, tools, args: tools == ["spotify_update_playlist"]
                    and args[0].get("playlist_id") == "it",
                    setup=lambda r: r.note_session_playlist_id(pl_created),
                ),
            ],
        ),
        SimFlow(
            "reorder_playlist",
            [
                SimTurn(
                    "reorder track 2 before track 1",
                    [
                        (
                            "spotify_reorder_playlist_tracks",
                            {
                                "playlist_id": pl_created,
                                "range_start": 1,
                                "insert_before": 0,
                                "range_length": 1,
                            },
                        )
                    ],
                    lambda _r, tools, _a: tools == ["spotify_reorder_playlist_tracks"],
                ),
            ],
        ),
        SimFlow(
            "save_playlist_then_remove",
            [
                SimTurn(
                    "save playlist 90s Rock Classics",
                    [("spotify_follow_playlist", {"playlist_id": pl_saved})],
                    lambda _r, tools, args: tools == ["spotify_follow_playlist"]
                    and args[0].get("playlist_id") == pl_saved,
                ),
                SimTurn(
                    "remove it from my library",
                    [("spotify_unfollow_playlist", {"playlist_id": "it"})],
                    lambda _r, tools, args: tools == ["spotify_unfollow_playlist"],
                    setup=lambda r: r._record_library_mutation("playlist", [pl_saved]),
                ),
            ],
        ),
    ]


def run_simulated_chat_bank(
    runner: SpotifyToolRunner,
    *,
    conversation_id: str = "chat-bank",
    show_id: str = "ssssssssssssssssssssss",
    episode_id: str = "eeeeeeeeeeeeeeeeeeeeee",
    track_id: str = "1111111111111111111111",
    album_id: str = "aaaaaaaaaaaaaaaaaaaa",
) -> list[SimulatedChatBankRow]:
    rows: list[SimulatedChatBankRow] = []

    shortcut_prompts = [
        "Play one of my playlists",
        "Surprise me",
        "What's playing?",
        "What playlists do I have?",
    ]
    for prompt in shortcut_prompts:
        row = _run_shortcut(prompt, runner, f"{conversation_id}-{len(rows)}")
        if prompt.lower().startswith("surprise"):
            row.passed = row.passed and not any("37i9" in t for t in row.tools_called)
        rows.append(row)

    tool_scripts: list[
        tuple[
            str,
            list[tuple[str, dict[str, Any]]],
            Callable[[str, list[str], list[dict[str, Any]]], bool],
        ]
    ] = [
        (
            "shuffle on",
            [("spotify_set_shuffle", {"state": True})],
            lambda _r, tools, _a: "spotify_set_shuffle" in tools,
        ),
        (
            "find podcasts about astronomy",
            [("spotify_search", {"query": "astronomy podcast", "types": "show", "limit": 5})],
            lambda _r, tools, _a: "spotify_search" in tools and _a[0].get("types") == "show",
        ),
        (
            "is this song in my likes?",
            [("spotify_library_contains", {"uris": [f"spotify:track:{track_id}"]})],
            lambda _r, tools, _a: "spotify_library_contains" in tools,
        ),
    ]
    for prompt, script, check in tool_scripts:
        rows.append(_run_tool_script(prompt, runner, script, pass_check=check))

    stale_album = "48YIv8aaaaaaaaaaaaaaab"
    live_album = album_id
    for flow in _podcast_flows(show_id, episode_id, track_id, album_id):
        for turn in flow.turns:
            rows.append(
                _run_tool_script(
                    turn.prompt,
                    runner,
                    turn.script,
                    pass_check=turn.check,
                    flow=flow.name,
                    setup=turn.setup,
                )
            )

    for flow in _pr9_retest_flows(
        show_id, episode_id, track_id, album_id, stale_album, live_album
    ):
        for turn in flow.turns:
            rows.append(
                _run_tool_script(
                    turn.prompt,
                    runner,
                    turn.script,
                    pass_check=turn.check,
                    flow=flow.name,
                    setup=turn.setup,
                )
            )

    return rows


def podcast_tool_names_in_agent_payload() -> set[str]:
    names: set[str] = set()
    for entry in OLLAMA_TOOLS:
        fn = entry.get("function") if isinstance(entry, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.add(fn["name"])
    required = {
        "spotify_library_contains",
        "spotify_library_save",
        "spotify_library_remove",
        "spotify_get_show",
        "spotify_get_show_episodes",
        "spotify_get_episode",
        "spotify_user_saved_shows",
        "spotify_play_show_latest_episode",
        "spotify_playlist_builder_preview",
        "spotify_playlist_builder_commit",
    }
    return required & names


def format_chat_bank_table(rows: list[SimulatedChatBankRow]) -> str:
    lines = [
        "| Flow | Prompt | Tools called | Reply (excerpt) | Pass |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        tools = ", ".join(row.tools_called) if row.tools_called else "—"
        excerpt = (row.reply or row.note or "")[:80].replace("|", "/").replace("\n", " ")
        mark = "pass" if row.passed else "FAIL"
        flow = row.flow or "—"
        lines.append(f"| {flow} | {row.prompt} | {tools} | {excerpt} | {mark} |")
    return "\n".join(lines)
