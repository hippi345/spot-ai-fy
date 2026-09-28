"""Multi-turn Gemini laptop replay with mocked Spotify and per-turn scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
import respx

from spot_backend.config import Settings
from spot_backend.gemini_llm import run_chat_turn_gemini
from spot_backend.spotify_tools import SpotifyToolRunner
from spot_backend.tool_server_enforcement import enforce_tool_arguments_for_turn

LIVE_ALBUM = "00GCAlaaaaaaaaaaaaaaab"
STALE_ALBUM = "48YIv8aaaaaaaaaaaaaaab"
STARTALK_ID = "4rOoJ6Egrf8K2IrywzwOMy"
PL_EXACT = "2HfFccisPxQfprhgIHM7XH"
PL_WRONG = "wrong0000000000000001"


@dataclass
class TurnExpectation:
    prompt: str
    check: Callable[[dict[str, Any]], tuple[bool, str]]


@dataclass
class EvalReport:
    turns: list[dict[str, Any]] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    crashed: bool = False
    crash_detail: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "crashed": self.crashed,
            "crash_detail": self.crash_detail,
            "turns": self.turns,
        }


def _install_spotify_mocks() -> None:
    player = {
        "is_playing": True,
        "item": {
            "type": "track",
            "name": "Live Track",
            "album": {"id": LIVE_ALBUM, "name": "Live Album"},
        },
    }
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player.*").mock(
        return_value=httpx.Response(200, json=player)
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/library/contains.*").mock(
        return_value=httpx.Response(200, json=[True])
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/albums.*").mock(
        return_value=httpx.Response(
            200,
            json={"items": [{"album": {"id": STALE_ALBUM, "name": "Old"}}], "total": 1},
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        side_effect=_search_router
    )
    respx.post(url__regex=r"https://api\.spotify\.com/v1/me/playlists.*").mock(
        return_value=httpx.Response(200, json={"id": "newpl" + "x" * 17, "name": "spot-ai-fy test"})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/playlists/.*").mock(
        return_value=httpx.Response(200, json={"id": PL_EXACT, "public": True, "name": "90s Rock Classics"})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/playlists/.*").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.post(url__regex=r"https://api\.spotify\.com/v1/playlists/.*/items").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204, text="")
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/player$").mock(
        return_value=httpx.Response(200, json=player)
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/shows/.*/episodes.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "ep" + "x" * 20,
                        "name": "Latest StarTalk",
                        "uri": "spotify:episode:" + "e" * 22,
                    }
                ]
            },
        )
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/library.*").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.delete(url__regex=r"https://api\.spotify\.com/v1/me/library.*").mock(
        return_value=httpx.Response(200, json={})
    )


def _search_router(request: httpx.Request) -> httpx.Response:
    params = dict(request.url.params)
    stype = params.get("type") or ""
    if "show" in stype:
        return httpx.Response(
            200,
            json={
                "shows": {
                    "items": [
                        None,
                        {
                            "id": STARTALK_ID,
                            "name": "StarTalk Radio",
                            "publisher": "StarTalk",
                            "uri": f"spotify:show:{STARTALK_ID}",
                        },
                    ],
                    "total": 1,
                }
            },
        )
    if "playlist" in stype:
        return httpx.Response(
            200,
            json={
                "playlists": {
                    "items": [
                        {"id": PL_WRONG, "name": "90s HITS | TOP 100 SONGS"},
                        {"id": PL_EXACT, "name": "90s Rock Classics"},
                    ],
                    "total": 2,
                }
            },
        )
    year = "1994"
    return httpx.Response(
        200,
        json={
            "tracks": {
                "items": [
                    {
                        "id": "t" * 22,
                        "uri": f"spotify:track:{'t' * 22}",
                        "name": f"Chill {i}",
                        "album": {"name": "90s Chill", "release_date": f"{year}-06-01"},
                        "artists": [{"name": "Artist"}],
                    }
                    for i in range(12)
                ]
            }
        },
    )


def _tool_calls_from_turn(recorded: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(recorded)


def run_multiturn_gemini_eval(settings: Settings) -> EvalReport:
    report = EvalReport()
    history: list[dict[str, str]] = []
    conv = "gemini-eval-multiturn"
    recorded: list[dict[str, Any]] = []
    real_run = SpotifyToolRunner.run

    def recording_run(self, name: str, arguments: dict[str, Any]) -> str:
        enforced = enforce_tool_arguments_for_turn(
            name, arguments, user_text=current_prompt, runner=self
        )
        recorded.append({"tool": name, "args": enforced})
        return real_run(self, name, enforced)

    script: list[TurnExpectation] = [
        TurnExpectation(
            "do I already have this album saved?",
            lambda ctx: (
                LIVE_ALBUM in json.dumps(ctx["tools"]),
                "library_contains should use live album id",
            ),
        ),
        TurnExpectation(
            "what albums do I have saved?",
            lambda ctx: (
                any(t["tool"] == "spotify_saved_albums" for t in ctx["tools"]),
                "should list saved albums",
            ),
        ),
        TurnExpectation(
            "build a chill 90s playlist called spot-ai-fy test",
            lambda ctx: (
                any(t["tool"] == "spotify_playlist_builder_preview" for t in ctx["tools"])
                and not any(t["tool"] == "spotify_playlist_builder_commit" for t in ctx["tools"]),
                "preview only before yes",
            ),
        ),
        TurnExpectation(
            "drop track 3",
            lambda ctx: (
                any(t["tool"] == "spotify_playlist_builder_edit" for t in ctx["tools"]),
                "builder edit for drop track 3",
            ),
        ),
        TurnExpectation(
            "yes make it",
            lambda ctx: (
                any(t["tool"] == "spotify_playlist_builder_commit" for t in ctx["tools"]),
                "commit after approval",
            ),
        ),
        TurnExpectation(
            "find podcasts about astronomy",
            lambda ctx: (
                any(
                    t["tool"] == "spotify_search"
                    and "show" in json.dumps(t.get("args", {}))
                    for t in ctx["tools"]
                ),
                "show search for podcasts",
            ),
        ),
        TurnExpectation(
            "play the latest episode of StarTalk",
            lambda ctx: (
                any(t["tool"] == "spotify_play_show_latest_episode" for t in ctx["tools"]),
                "play latest episode tool",
            ),
        ),
        TurnExpectation(
            "save this show",
            lambda ctx: (
                any(
                    t["tool"] == "spotify_library_save"
                    and STARTALK_ID in json.dumps(t.get("args", {}))
                    for t in ctx["tools"]
                ),
                "save show uses StarTalk id",
            ),
        ),
        TurnExpectation(
            "is this show saved?",
            lambda ctx: (
                any(t["tool"] == "spotify_library_contains" for t in ctx["tools"]),
                "contains check for show",
            ),
        ),
        TurnExpectation(
            "remove it from my library",
            lambda ctx: (
                any(t["tool"] == "spotify_library_remove" for t in ctx["tools"]),
                "library remove",
            ),
        ),
        TurnExpectation(
            "save the playlist 90s Rock Classics",
            lambda ctx: (
                any(
                    t["tool"] == "spotify_follow_playlist"
                    and t.get("args", {}).get("playlist_id") == PL_EXACT
                    for t in ctx["tools"]
                ),
                "follow exact 90s Rock Classics id",
            ),
        ),
        TurnExpectation(
            "is it saved?",
            lambda ctx: (
                any(t["tool"] == "spotify_library_contains" for t in ctx["tools"])
                or "saved" in (ctx.get("reply") or "").lower(),
                "saved yes/no answer",
            ),
        ),
    ]

    current_prompt = ""
    try:
        with respx.mock:
            _install_spotify_mocks()
            with __import__("unittest").mock.patch.object(SpotifyToolRunner, "run", recording_run):
                for step in script:
                    current_prompt = step.prompt
                    turn_tools_start = len(recorded)
                    reply = run_chat_turn_gemini(
                        step.prompt,
                        settings,
                        history=history or None,
                        conversation_id=conv,
                    )
                    tools = recorded[turn_tools_start:]
                    ctx = {"tools": tools, "reply": reply}
                    ok, note = step.check(ctx)
                    row = {
                        "prompt": step.prompt,
                        "pass": ok,
                        "note": note,
                        "tools": tools,
                        "reply": reply[:800],
                    }
                    report.turns.append(row)
                    if ok:
                        report.passed += 1
                    else:
                        report.failed += 1
                    history.append({"role": "user", "content": step.prompt})
                    history.append({"role": "assistant", "content": reply})
    except Exception as exc:  # noqa: BLE001 — eval must record crash
        report.crashed = True
        report.crash_detail = f"{type(exc).__name__}: {exc}"
    return report


def format_eval_table(report: EvalReport) -> str:
    lines = ["Gemini eval (multi-turn)", f"passed={report.passed} failed={report.failed} crashed={report.crashed}"]
    for i, row in enumerate(report.turns, 1):
        mark = "PASS" if row.get("pass") else "FAIL"
        lines.append(f"{i:02d} {mark} | {row['prompt'][:48]} | {row.get('note', '')}")
    return "\n".join(lines)
