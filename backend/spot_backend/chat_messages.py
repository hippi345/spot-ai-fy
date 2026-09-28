"""User-visible chat fallback copy shared by the agent and frontend."""

from __future__ import annotations

import json
import re
from typing import Any

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
_CODE_SPAN_RE = re.compile(r"`([^`]*)`")
_TOOL_PARAMETER_SCRUB = re.compile(
    r"\b(?:the\s+)?[`']?(?:Spotify|spotify)[`']?\s+tool\b",
    re.I,
)
_PARAMETER_DOC_SCRUB = re.compile(
    r"\b(?:set|change)\s+the\s+(?:[`']?[a-z_][a-z0-9_]*[`']?\s+)?parameter\s+to\b[^.]*",
    re.I,
)
_PARAM_LIKE_SPAN = frozenset(
    {
        "public",
        "private",
        "device_id",
        "playlist_id",
        "query",
        "state",
        "volume_percent",
        "position_ms",
        "track_id",
        "album_id",
        "artist_id",
        "uri",
        "uris",
        "context_uri",
        "offset",
        "limit",
        "types",
        "market",
    }
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
        if data.get("verified_private") is False:
            pw = data.get("privacy_warning")
            if isinstance(pw, str) and pw.strip() and pw not in seen:
                seen.add(pw)
                warnings.append(pw.strip())
        if not data.get("visibility_change_requested"):
            continue
        note = data.get("visibility_warning")
        if isinstance(note, str) and note.strip() and note not in seen:
            seen.add(note)
            warnings.append(note.strip())
    return warnings


_VISIBILITY_NOTE_MARKERS = (
    "still shows it as public",
    "still reports this playlist as public",
    "still showing it as public",
    "might appear public",
    "shows this playlist as public",
    "make it private",
)

_MODEL_VISIBILITY_DISCUSSION_MARKERS = _VISIBILITY_NOTE_MARKERS + (
    "spotify is still showing",
    "may take a few",
    "may take some time",
    "sync delay",
    "known spotify api quirk",
    "visibility may",
)


def _sentence_mentions_visibility_discussion(sentence: str) -> bool:
    low = (sentence or "").lower()
    if any(marker in low for marker in _MODEL_VISIBILITY_DISCUSSION_MARKERS):
        return True
    if "public" in low and any(
        token in low for token in ("still", "delay", "sync", "lag", "showing", "reports")
    ):
        return True
    if "private" in low and "still" in low:
        return True
    return False


def _split_sentences(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if p.strip()]


def _join_sentences(parts: list[str]) -> str:
    return " ".join(p for p in parts if p).strip()


def _text_contains_visibility_note(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in _VISIBILITY_NOTE_MARKERS)


def _strip_duplicate_visibility_sentences(text: str) -> str:
    """Remove user-visible visibility mismatch sentences already present in model text."""
    if not text:
        return ""
    kept = [s for s in _split_sentences(text) if not _text_contains_visibility_note(s)]
    return _join_sentences(kept)


def strip_model_visibility_discussion(text: str) -> str:
    """Drop model-written visibility/sync sentences when the backend adds its own note."""
    if not text:
        return ""
    kept = [s for s in _split_sentences(text) if not _sentence_mentions_visibility_discussion(s)]
    return _join_sentences(kept)


_PRIVATE_VISIBILITY_CLAIM = re.compile(
    r"\b(?:currently\s+private|(?:it(?:'s|\s+is)|remains?|stays?)\s+(?:set\s+to\s+)?private)\b",
    re.I,
)
_PRIVATE_CHANGE_SUCCESS_CLAIM = re.compile(
    r"\b(?:"
    r"made\s+(?:it\s+|the\s+playlist\s+)?private|"
    r"(?:updated|changed)\s+(?:the\s+)?(?:playlist\s+)?(?:\"[^\"]+\"|'[^']+'|\S+\s+)?to\s+be\s+private|"
    r"(?:is|are)\s+now\s+private|"
    r"set\s+(?:it\s+|the\s+playlist\s+)?to\s+private|"
    r"updated\s+your\s+playlist\s+to\s+be\s+private"
    r")\b",
    re.I,
)
_PUBLIC_VISIBILITY_CLAIM = re.compile(
    r"\b(?:currently\s+public|(?:it(?:'s|\s+is)|remains?|stays?)\s+(?:set\s+to\s+)?public)\b",
    re.I,
)


def _tool_visibility_mismatch_results(tool_results: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in tool_results:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("visibility_mismatch"):
            rows.append(data)
            continue
        if (
            data.get("visibility_change_requested")
            and data.get("ok") is False
            and data.get("visibility_warning")
        ):
            rows.append(data)
    return rows


def _sentence_claims_private_change_success(sentence: str) -> bool:
    return bool(_PRIVATE_CHANGE_SUCCESS_CLAIM.search(sentence or ""))


def _strip_private_change_success_claims(text: str) -> str:
    if not text:
        return ""
    kept = [s for s in _split_sentences(text) if not _sentence_claims_private_change_success(s)]
    return _join_sentences(kept)


def _tool_results_with_public_field(tool_results: list[str]) -> list[tuple[bool | None, str | None]]:
    rows: list[tuple[bool | None, str | None]] = []
    for raw in tool_results:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(data, dict) or "public" not in data:
            continue
        pub = data.get("public")
        if pub is not True and pub is not False:
            continue
        name = data.get("name") if isinstance(data.get("name"), str) else None
        rows.append((bool(pub), name))
    return rows


def fix_playlist_visibility_contradictions(text: str, tool_results: list[str]) -> str:
    """Ensure user-visible text does not contradict spotify_create/update `public` field."""
    out = (text or "").strip()
    if not out:
        return out
    if _tool_visibility_mismatch_results(tool_results):
        out = _strip_private_change_success_claims(out)
    for actual_public, name in _tool_results_with_public_field(tool_results):
        contradicts = (actual_public and _PRIVATE_VISIBILITY_CLAIM.search(out)) or (
            not actual_public and _PUBLIC_VISIBILITY_CLAIM.search(out)
        )
        if not contradicts:
            continue
        cleaned_parts = []
        for sentence in _split_sentences(out):
            if actual_public and _PRIVATE_VISIBILITY_CLAIM.search(sentence):
                continue
            if not actual_public and _PUBLIC_VISIBILITY_CLAIM.search(sentence):
                continue
            cleaned_parts.append(sentence)
        out = _join_sentences(cleaned_parts)
        label = name.strip() if isinstance(name, str) and name.strip() else "your playlist"
        if actual_public:
            out = _join_sentences([out, f"{label} is public on Spotify."])
        else:
            out = _join_sentences([out, f"{label} is private on Spotify."])
    return out.strip()


def _visibility_note_fingerprint(note: str) -> str:
    low = re.sub(r"\s+", " ", (note or "").lower()).strip()
    for marker in _VISIBILITY_NOTE_MARKERS:
        if marker in low:
            return marker
    if "public" in low and "private" in low:
        return "visibility_public_private"
    if "public" in low and any(tok in low for tok in ("still", "spotify", "app", "show")):
        return "visibility_public"
    return low[:120]


def append_visibility_notes_to_reply(text: str, tool_results: list[str]) -> str:
    """Append deterministic playlist-visibility notes from tool JSON (all LLM providers)."""
    base = text or ""
    warnings = collect_visibility_warnings(tool_results)
    if warnings:
        base = strip_model_visibility_discussion(base)
    base = _strip_duplicate_visibility_sentences(base.rstrip())
    seen_fps: set[str] = set()
    for note in warnings:
        fp = _visibility_note_fingerprint(note)
        if fp in seen_fps:
            continue
        if note in base:
            seen_fps.add(fp)
            continue
        if _text_contains_visibility_note(note) and _text_contains_visibility_note(base):
            seen_fps.add(fp)
            continue
        if note.strip() and note.strip() in base:
            seen_fps.add(fp)
            continue
        if any(_visibility_note_fingerprint(existing) == fp for existing in seen_fps):
            continue
        suffix = f"\n\n{note.strip()}" if not base.endswith(note.strip()) else ""
        if suffix:
            base = base + suffix
            seen_fps.add(fp)
    return base


def _scrub_inline_code_span(match: re.Match[str]) -> str:
    """Drop backticks; remove span content only for tool/parameter/JSON-like internals."""
    inner = match.group(1)
    stripped = inner.strip()
    if not stripped:
        return ""
    if re.fullmatch(r"spotify_[a-z0-9_]+", stripped, re.I):
        return "Spotify"
    if stripped.lower() in ("spotify", "true", "false", "null"):
        return "" if stripped.lower() in ("true", "false", "null") else "Spotify"
    if re.fullmatch(r"[a-z][a-z0-9_]*", stripped) and stripped.lower() in _PARAM_LIKE_SPAN:
        return ""
    if stripped.startswith(("{", "[", '"')) or re.search(r'"\s*:\s*', stripped):
        return ""
    return inner


_HTTP_STATUS_SCRUB = re.compile(r"\bHTTP\s+\d{3}\b", re.I)
_SPOTIFY_ID_SCRUB = re.compile(r"\b[0-9A-Za-z]{22}\b")
_SEARCH_QUERY_ECHO_SCRUB = re.compile(
    r"No tracks found for\s+['\"].+?['\"]",
    re.I,
)


def _collapse_inline_whitespace_preserve_newlines(text: str) -> str:
    lines = (text or "").split("\n")
    cleaned = [re.sub(r"[ \t]{2,}", " ", line).strip() for line in lines]
    return "\n".join(cleaned).strip()


_INTERNAL_PAGINATION_HINT_RE = re.compile(
    r"\(?\s*(?:more .+ — pass offset=\d+|pass offset=\d+)[^)\n]*\)?",
    re.I,
)


def strip_internal_pagination_hints(text: str) -> str:
    out = _INTERNAL_PAGINATION_HINT_RE.sub("", text or "")
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def scrub_user_visible_spotify_errors(text: str) -> str:
    out = _HTTP_STATUS_SCRUB.sub("", text or "")
    out = _SEARCH_QUERY_ECHO_SCRUB.sub("I couldn't find that on Spotify", out)
    return _collapse_inline_whitespace_preserve_newlines(out)


def scrub_internal_tool_references(text: str) -> str:
    """Remove internal spotify_* tool names and function-call syntax from user-visible replies."""
    out = _CODE_SPAN_RE.sub(_scrub_inline_code_span, text or "")
    out = _FC_SYNTAX_SCRUB.sub("", out)
    out = _TOOL_NAME_SCRUB.sub("Spotify", out)
    out = _TOOL_PARAMETER_SCRUB.sub("Spotify", out)
    out = _PARAMETER_DOC_SCRUB.sub("", out)
    out = re.sub(r"\bSpotify\s+Spotify\b", "Spotify", out)
    return _collapse_inline_whitespace_preserve_newlines(out)


_CORRECTION_LEAK_RE = re.compile(
    r"(?:^|\n)\s*my apologies\b.*?(?=\n\n|\Z)",
    re.I | re.DOTALL,
)
_PREVIOUS_RESPONSE_LEAK_RE = re.compile(
    r"(?:^|\n)\s*(?:in my (?:previous|last) response\b|my last response stated\b).*?(?=\n\n|\Z)",
    re.I | re.DOTALL,
)
_RAW_JSON_BLOB_RE = re.compile(r"\{[^{}]*\"(?:ok|error|items)\"[^{}]*\}", re.DOTALL)


def strip_internal_correction_leaks(text: str) -> str:
    """Drop model self-correction / apology preambles from user-visible replies."""
    out = text or ""
    for pattern in (_CORRECTION_LEAK_RE, _PREVIOUS_RESPONSE_LEAK_RE):
        out = pattern.sub("", out)
    return out.strip()


def sanitize_raw_tool_json_in_reply(text: str) -> str:
    """Replace accidental raw Spotify tool JSON blobs with a short plain sentence."""
    stripped = (text or "").strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return text
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, TypeError, ValueError):
        return text
    if not isinstance(data, dict):
        return text
    if data.get("error"):
        return str(data.get("error"))
    if data.get("ok"):
        if "saved_single" in data:
            return (
                "Yes — that's saved in your Spotify library."
                if data.get("saved_single")
                else "No — that's not in your library."
            )
        if isinstance(data.get("user_message"), str) and data["user_message"].strip():
            return data["user_message"].strip()
        return "Done."
    return text


_NUMBERED_LIST_GLUE_RE = re.compile(r"(\S)\s+(\d{1,2}\.\s)")


def _split_glued_numbered_items(line: str) -> str:
    prev = None
    out = line
    while prev != out:
        prev = out
        out = _NUMBERED_LIST_GLUE_RE.sub(r"\1\n\2", out)
    return out


def fix_numbered_list_line_breaks(text: str) -> str:
    """Ensure glued builder preview lines like '…artist 9. Track 10. Other' break before each number."""
    if not text or not re.search(r"\d{1,2}\.\s", text):
        return text
    return "\n".join(_split_glued_numbered_items(ln) for ln in (text or "").split("\n"))


def collapse_duplicate_reply_text(text: str) -> str:
    """Drop exact duplicate sentences or paragraphs while preserving first occurrence."""
    raw = (text or "").strip()
    if not raw:
        return ""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if not paragraphs:
        return ""
    seen_paras: set[str] = set()
    kept_paras: list[str] = []
    for para in paragraphs:
        if para in seen_paras:
            continue
        seen_paras.add(para)
        seen_sent: set[str] = set()
        sent_parts: list[str] = []
        for sentence in _split_sentences(para):
            if sentence in seen_sent:
                continue
            seen_sent.add(sentence)
            sent_parts.append(sentence)
        if sent_parts:
            kept_paras.append(_join_sentences(sent_parts))
    return "\n\n".join(kept_paras).strip()


def prepare_user_visible_reply(
    text: str,
    tool_results: list[str] | None = None,
    *,
    tool_names: list[str] | None = None,
    user_text: str = "",
) -> str:
    from spot_backend.reply_grounding import ground_reply_artist_credits
    from spot_backend.reply_tool_fallback import ensure_substantive_user_reply

    cleaned = collapse_duplicate_reply_text(text)
    cleaned = scrub_internal_tool_references(cleaned)
    cleaned = scrub_user_visible_spotify_errors(cleaned)
    cleaned = strip_internal_pagination_hints(cleaned)
    cleaned = strip_internal_correction_leaks(cleaned)
    cleaned = sanitize_raw_tool_json_in_reply(cleaned)
    if tool_results:
        cleaned = ground_reply_artist_credits(cleaned, tool_results)
        cleaned = fix_playlist_visibility_contradictions(cleaned, tool_results)
        cleaned = append_visibility_notes_to_reply(cleaned, tool_results)
    cleaned = fix_numbered_list_line_breaks(cleaned)
    cleaned = ensure_substantive_user_reply(
        cleaned,
        tool_results,
        tool_names=tool_names,
        user_text=user_text,
    )
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
