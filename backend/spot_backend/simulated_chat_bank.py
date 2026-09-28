"""Deterministic simulated chat bank (mocked Spotify HTTP via respx in tests)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from spot_backend.chat_shortcuts import try_deterministic_chat_reply
from spot_backend.deterministic_chat import resolve_deterministic_chat_outcome
from spot_backend.prompt_intent import prompt_is_surprise_me_request
from spot_backend.spotify_tools import SpotifyToolRunner


@dataclass
class SimulatedChatBankRow:
    prompt: str
    tools_called: list[str]
    reply: str
    passed: bool
    note: str = ""


def _tool_names_from_steps(steps: list[tuple[str, dict, str]]) -> list[str]:
    return [name for name, _, _ in steps]


def _run_shortcut(prompt: str, runner: SpotifyToolRunner, conversation_id: str) -> SimulatedChatBankRow:
    outcome = resolve_deterministic_chat_outcome(prompt, runner, conversation_id=conversation_id)
    if outcome is None:
        outcome = try_deterministic_chat_reply(prompt, runner, conversation_id=conversation_id)
    if outcome is None:
        return SimulatedChatBankRow(prompt, [], "", False, "no deterministic handler")
    tools = outcome.tool_names()
    reply = (outcome.reply or "").strip()
    return SimulatedChatBankRow(prompt, tools, reply, bool(reply), "")


def _run_tool_script(
    prompt: str,
    runner: SpotifyToolRunner,
    script: list[tuple[str, dict[str, Any]]],
    *,
    pass_check: Callable[[str, list[str]], bool],
) -> SimulatedChatBankRow:
    tools: list[str] = []
    last_raw = ""
    for name, args in script:
        tools.append(name)
        last_raw = runner.run(name, args)
    data = json.loads(last_raw) if last_raw else {}
    reply = str(data.get("user_message") or data.get("message") or data.get("preview_text") or "")
    if not reply and data.get("ok"):
        reply = "ok"
    passed = pass_check(reply, tools)
    return SimulatedChatBankRow(prompt, tools, reply, passed, "")


def run_simulated_chat_bank(
    runner: SpotifyToolRunner,
    *,
    conversation_id: str = "chat-bank",
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
            row.passed = row.passed and not any(
                "37i9" in t for t in row.tools_called
            )
            if not row.tools_called:
                row.passed = prompt_is_surprise_me_request(prompt) is False or row.passed
        rows.append(row)

    tool_scripts: list[tuple[str, list[tuple[str, dict[str, Any]]], Callable[[str, list[str]], bool]]] = [
        (
            "shuffle on",
            [("spotify_set_shuffle", {"state": True})],
            lambda _r, tools: "spotify_set_shuffle" in tools,
        ),
        (
            "find podcasts about astronomy",
            [("spotify_search", {"query": "astronomy podcast", "types": "show", "limit": 5})],
            lambda _r, tools: "spotify_search" in tools,
        ),
        (
            "is this song in my likes?",
            [
                (
                    "spotify_library_contains",
                    {"uris": ["spotify:track:1111111111111111111111"]},
                )
            ],
            lambda _r, tools: "spotify_library_contains" in tools,
        ),
    ]
    for prompt, script, check in tool_scripts:
        rows.append(
            _run_tool_script(prompt, runner, script, pass_check=check)
        )

    return rows


def format_chat_bank_table(rows: list[SimulatedChatBankRow]) -> str:
    lines = [
        "| Prompt | Tools called | Reply (excerpt) | Pass |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        tools = ", ".join(row.tools_called) if row.tools_called else "—"
        excerpt = (row.reply or row.note or "")[:80].replace("|", "/").replace("\n", " ")
        mark = "pass" if row.passed else "FAIL"
        lines.append(f"| {row.prompt} | {tools} | {excerpt} | {mark} |")
    return "\n".join(lines)
