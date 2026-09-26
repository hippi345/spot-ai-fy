"""User-visible chat fallback copy shared by the agent and frontend."""

from __future__ import annotations

import json
import re

STOCK_NO_ASSISTANT_HINT = (
    "The model returned no assistant text and no tool calls (Ollama may stream reasoning "
    "without a final answer, or the run was cut short). Try: (1) a shorter, one-step question, "
    "(2) another Ollama tag if this one misbehaves with tools, (3) Gemini in Spot-AI-fy, or "
    "(4) concrete examples in backend/AGENT_CONTEXT.md."
)

FRIENDLY_SPOTIFY_GUIDANCE = (
    "I can help with Spotify things like playing music, searching, playlists, and queue "
    "management — try something like “play my workout playlist”, “search for Taylor Swift”, "
    "or “what’s playing?”"
)


def is_unpersisted_assistant_fallback(text: str) -> bool:
    """True when assistant text should not be stored in chat history."""
    t = (text or "").strip()
    if not t:
        return True
    if t == STOCK_NO_ASSISTANT_HINT or t == FRIENDLY_SPOTIFY_GUIDANCE:
        return True
    if t.startswith("The model returned no assistant text"):
        return True
    if t.startswith("No response from model."):
        return True
    return False


def friendly_reply_for_empty_model_output(user_text: str) -> str:
    """Return a helpful reply when the model produced no usable assistant text."""
    _ = user_text
    return FRIENDLY_SPOTIFY_GUIDANCE


_PROMISE_ONLY_REPLY = re.compile(
    r"^\s*(?:i(?:'ll|\s+will)|let me|give me a (?:moment|sec)|i(?:'m|\s+am)\s+going to)\b",
    re.I,
)

_TOOL_NAME_SCRUB = re.compile(
    r"\bspotify_[a-z0-9_]+\b(?:\s*\([^)]*\))?",
    re.I,
)
_FC_SYNTAX_SCRUB = re.compile(
    r"\bspotify_[a-z0-9_]+\s*\(\s*[^)]*\)",
    re.I,
)
_CODE_SPAN_SCRUB = re.compile(r"`[^`]*`")
_TOOL_PARAMETER_SCRUB = re.compile(
    r"\b(?:the\s+)?[`']?(?:Spotify|spotify)[`']?\s+tool\b",
    re.I,
)
_PARAMETER_WORD_SCRUB = re.compile(
    r"\b(?:set|change)\s+the\s+[`']?[a-z_]+[`']?\s+parameter\b",
    re.I,
)

PROMISE_AFTER_ID_ERROR_NUDGE = (
    "Spot-AI-fy: The last Spotify tool failed because an id/uri was invalid. "
    "Do not reply with only a promise — call spotify_search or another lookup tool now, "
    "then answer with what you found."
)


def assistant_reply_is_promise_only(text: str) -> bool:
    """True when the model defers action to a later step without calling tools."""
    t = (text or "").strip()
    if not t:
        return False
    return bool(_PROMISE_ONLY_REPLY.match(t))


def collect_visibility_warnings(tool_results: list[str]) -> list[str]:
    warnings: list[str] = []
    seen: set[str] = set()
    for raw in tool_results:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        note = data.get("visibility_warning")
        if isinstance(note, str) and note.strip() and note not in seen:
            seen.add(note)
            warnings.append(note.strip())
    return warnings


def append_visibility_notes_to_reply(text: str, tool_results: list[str]) -> str:
    """Append deterministic playlist-visibility notes from tool JSON (all LLM providers)."""
    base = (text or "").rstrip()
    for note in collect_visibility_warnings(tool_results):
        if note in base:
            continue
        suffix = f"\n\nNote: {note}"
        base = base + suffix
    return base


def scrub_internal_tool_references(text: str) -> str:
    """Remove internal spotify_* tool names and function-call syntax from user-visible replies."""
    out = _CODE_SPAN_SCRUB.sub("", text or "")
    out = _FC_SYNTAX_SCRUB.sub("", out)
    out = _TOOL_NAME_SCRUB.sub("Spotify", out)
    out = _TOOL_PARAMETER_SCRUB.sub("Spotify", out)
    out = _PARAMETER_WORD_SCRUB.sub("", out)
    out = re.sub(r"\bSpotify\s+Spotify\b", "Spotify", out)
    out = re.sub(r"\bparameter[s]?\b", "", out, flags=re.I)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()


def prepare_user_visible_reply(text: str, tool_results: list[str] | None = None) -> str:
    cleaned = scrub_internal_tool_references(text)
    if tool_results:
        cleaned = append_visibility_notes_to_reply(cleaned, tool_results)
    return cleaned


def tool_result_is_rejected_or_invalid_id(raw: str) -> bool:
    """True when a tool failed because an id/uri was wrong or rejected."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    if data.get("rejected_uris") or data.get("skipped_uris"):
        return True
    if data.get("sign_out_not_recommended") and data.get("error"):
        return True
    err = str(data.get("error") or "")
    low = err.lower()
    return any(
        token in low
        for token in (
            "no track",
            "no album",
            "no artist",
            "unknown playlist",
            "http 404",
            "not found",
            "malformed spotify",
        )
    )
