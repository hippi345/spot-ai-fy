"""Helpers for the opt-in Ollama smoke test (docs/SMOKE_TEST.md steps 1–7)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import httpx


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]


@dataclass
class PromptSmokeResult:
    step: int
    prompt: str
    latency_s: float
    tools: list[ToolCallRecord] = field(default_factory=list)
    expected_tools_ok: bool = False
    expected_tools_note: str = ""
    reply_sensible: bool = False
    reply_note: str = ""
    reply_excerpt: str = ""
    spotify_state_ok: bool = False
    spotify_state_note: str = ""
    error: str | None = None

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        return self.expected_tools_ok and self.reply_sensible and self.spotify_state_ok


def tools_summary(tools: list[ToolCallRecord]) -> str:
    if not tools:
        return "(none)"
    parts: list[str] = []
    for tc in tools:
        args = json.dumps(tc.arguments, sort_keys=True) if tc.arguments else "{}"
        parts.append(f"{tc.name}({args})")
    return "; ".join(parts)


def _tool_names(tools: list[ToolCallRecord]) -> list[str]:
    return [t.name for t in tools]


def _has_any(names: list[str], candidates: set[str]) -> bool:
    return any(n in candidates for n in names)


def evaluate_expected_tools(step: int, tools: list[ToolCallRecord]) -> tuple[bool, str]:
    names = _tool_names(tools)
    if not names:
        return False, "Model did not call any Spotify tools"
    if step == 1:
        if "spotify_play_artist" in names:
            return True, "spotify_play_artist"
        if _has_any(names, {"spotify_start_resume_playback", "spotify_play_playlist"}):
            return True, "playback tool after search (acceptable alternate path)"
        return False, f"Expected spotify_play_artist or playback tool; got {names}"
    if step == 2:
        ok = "spotify_playback_state" in names
        return ok, "spotify_playback_state" if ok else f"Expected spotify_playback_state; got {names}"
    if step == 3:
        ok = "spotify_skip_next" in names
        return ok, "spotify_skip_next" if ok else f"Expected spotify_skip_next; got {names}"
    if step == 4:
        ok = "spotify_pause" in names
        return ok, "spotify_pause" if ok else f"Expected spotify_pause; got {names}"
    if step == 5:
        ok = "spotify_start_resume_playback" in names
        return ok, "spotify_start_resume_playback" if ok else f"Expected spotify_start_resume_playback; got {names}"
    if step == 6:
        ok = "spotify_user_playlists" in names
        return ok, "spotify_user_playlists" if ok else f"Expected spotify_user_playlists; got {names}"
    if step == 7:
        if "spotify_add_to_queue" in names:
            return True, "spotify_add_to_queue"
        return False, f"Expected spotify_add_to_queue; got {names}"
    return False, "Unknown step"


def results_to_markdown_table(results: list[PromptSmokeResult], *, model: str) -> str:
    lines = [
        f"## Ollama smoke results (`{model}`)",
        "",
        "| # | Prompt | Tools (args) | Tool OK? | Reply OK? | Spotify OK? | Latency (s) | Notes |",
        "|---|--------|--------------|----------|-----------|-------------|-------------|-------|",
    ]
    for row in results:
        notes = " | ".join(
            filter(
                None,
                [
                    row.expected_tools_note if not row.expected_tools_ok else "",
                    row.reply_note if not row.reply_sensible else "",
                    row.spotify_state_note if not row.spotify_state_ok else "",
                    row.error or "",
                ],
            )
        )
        lines.append(
            "| {step} | `{prompt}` | {tools} | {tool_ok} | {reply_ok} | {state_ok} | {lat:.1f} | {notes} |".format(
                step=row.step,
                prompt=row.prompt.replace("|", "\\|"),
                tools=tools_summary(row.tools).replace("|", "\\|"),
                tool_ok="yes" if row.expected_tools_ok else "**no**",
                reply_ok="yes" if row.reply_sensible else "**no**",
                state_ok="yes" if row.spotify_state_ok else "**no**",
                lat=row.latency_s,
                notes=notes or "—",
            )
        )
    passed = sum(1 for r in results if r.passed)
    lines.extend(["", f"**Summary:** {passed}/{len(results)} prompts fully passed.", ""])
    return "\n".join(lines)


def write_report(path: str, results: list[PromptSmokeResult], *, model: str, total_s: float) -> None:
    payload = {
        "model": model,
        "total_latency_s": total_s,
        "results": [
            {
                **asdict(r),
                "tools": [asdict(t) for t in r.tools],
                "passed": r.passed,
            }
            for r in results
        ],
        "markdown_table": results_to_markdown_table(results, model=model),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def ollama_model_tag_from_env(default: str = "qwen2.5:3b") -> str:
    return (os.environ.get("OLLAMA_MODEL") or default).strip() or default


def ollama_smoke_strict() -> bool:
    raw = os.environ.get("OLLAMA_SMOKE_STRICT", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def probe_ollama_model(host: str, model: str, *, timeout: float = 30.0) -> tuple[bool, str]:
    """Return (ready, detail). Checks /api/tags for the requested model tag."""
    base = host.rstrip("/")
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout, connect=5.0)) as client:
            tags = client.get(f"{base}/api/tags")
            tags.raise_for_status()
            names = [
                str(m.get("name") or "")
                for m in tags.json().get("models") or []
                if isinstance(m, dict)
            ]
            if model in names:
                return True, "ok"
            prefix = model.split(":")[0]
            fuzzy = [n for n in names if n.startswith(model) or n.startswith(f"{prefix}:")]
            if fuzzy:
                return True, f"ok (installed as {fuzzy[0]!r})"
            return False, f"Model {model!r} not in ollama list: {names}"
    except httpx.RequestError as exc:
        return False, f"Cannot reach Ollama at {base}: {exc}"


ReplyValidator = Callable[[str, dict[str, Any]], tuple[bool, str]]
StateValidator = Callable[[dict[str, Any]], tuple[bool, str]]


def build_reply_validators(state_holder: dict[str, Any]) -> dict[int, ReplyValidator]:
    """Validators keyed by smoke step; state_holder carries mutable keys like current_name."""

    def v1(reply: str, np: dict[str, Any]) -> tuple[bool, str]:
        if np.get("is_playing") is True:
            return True, ""
        low = reply.lower()
        if "try something like" in low or "how would you like" in low:
            return False, "Generic help reply instead of acting on the request"
        if "playing" in low or "started" in low:
            return True, ""
        return False, "Reply did not acknowledge playback"

    def v2(reply: str, np: dict[str, Any]) -> tuple[bool, str]:
        name = state_holder.get("current_name") or ""
        if name and name.lower() in reply.lower():
            return True, ""
        return False, f"Reply did not mention current track {name!r}"

    def v6(reply: str, _np: dict[str, Any]) -> tuple[bool, str]:
        if "Evening Acoustic" in reply and "Road Trip Mix" in reply:
            return True, ""
        return False, "Reply missing expected playlist names"

    def v7(reply: str, _np: dict[str, Any]) -> tuple[bool, str]:
        if "gravity" in reply.lower() or "queue" in reply.lower():
            return True, ""
        return False, "Reply did not mention queue/Gravity"

    return {
        1: v1,
        2: v2,
        6: v6,
        7: v7,
    }


def build_state_validators(state_holder: dict[str, Any]) -> dict[int, StateValidator]:
    def s1(_np: dict[str, Any]) -> tuple[bool, str]:
        if _np.get("is_playing") is True and "John Mayer" in " ".join(
            (_np.get("track") or {}).get("artists") or []
        ):
            state_holder["current_name"] = (_np.get("track") or {}).get("name")
            return True, ""
        return False, "Now-playing bar not playing John Mayer"

    def s2(np: dict[str, Any]) -> tuple[bool, str]:
        if (np.get("track") or {}).get("name") == state_holder.get("current_name"):
            return True, ""
        return False, "Track changed unexpectedly after what's playing?"

    def s3(np: dict[str, Any]) -> tuple[bool, str]:
        prev = state_holder.get("current_name")
        new = (np.get("track") or {}).get("name")
        if prev and new and new != prev:
            state_holder["current_name"] = new
            return True, ""
        return False, f"Skip did not change track (still {prev!r})"

    def s4(np: dict[str, Any]) -> tuple[bool, str]:
        if np.get("is_playing") is False:
            return True, ""
        return False, "Expected paused playback"

    def s5(np: dict[str, Any]) -> tuple[bool, str]:
        if np.get("is_playing") is True:
            return True, ""
        return False, "Expected resumed playback"

    def s7(np: dict[str, Any]) -> tuple[bool, str]:
        names = [q.get("name") for q in np.get("queue") or [] if isinstance(q, dict)]
        if "Gravity" in names:
            return True, ""
        return False, f"Gravity not in queue ({names})"

    return {1: s1, 2: s2, 3: s3, 4: s4, 5: s5, 7: s7}
