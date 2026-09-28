"""Fallback user-visible replies from successful tool JSON (apologies, Done., hallucinations)."""

from __future__ import annotations

import json
import re
from typing import Any

_APOLOGY_OR_CORRECTION_RE = re.compile(
    r"\b(?:"
    r"i apologize|my apologies|you are right|sorry,?\s+i(?:\s+incorrectly)?|"
    r"i incorrectly claimed|in my (?:previous|last) response"
    r")\b",
    re.I,
)

_DONE_ONLY_RE = re.compile(r"^done\.?$", re.I)


def _parse_tool_dict(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def tool_result_has_actionable_summary(data: dict[str, Any]) -> bool:
    if data.get("user_message") and isinstance(data["user_message"], str):
        return True
    if data.get("preview_text") and isinstance(data["preview_text"], str):
        return True
    if data.get("summary_lines") and isinstance(data["summary_lines"], list):
        return True
    if "saved_single" in data:
        return True
    return False


def summary_text_from_tool_dict(data: dict[str, Any]) -> str | None:
    if isinstance(data.get("user_message"), str) and data["user_message"].strip():
        return data["user_message"].strip()
    if isinstance(data.get("preview_text"), str) and data["preview_text"].strip():
        return data["preview_text"].strip()
    lines = data.get("summary_lines")
    if isinstance(lines, list) and lines:
        joined = "\n".join(str(x) for x in lines if str(x).strip())
        if joined.strip():
            return joined.strip()
    if "saved_single" in data:
        saved = bool(data.get("saved_single"))
        return "Yes — that's saved in your Spotify library." if saved else "No — that's not in your library."
    return None


def best_tool_summary_fallback(tool_results: list[str] | None) -> str | None:
    if not tool_results:
        return None
    for raw in reversed(tool_results):
        data = _parse_tool_dict(raw)
        if not data or data.get("error") or data.get("ok") is False:
            continue
        text = summary_text_from_tool_dict(data)
        if text:
            return text
    return None


def reply_looks_like_stale_apology(text: str) -> bool:
    return bool(_APOLOGY_OR_CORRECTION_RE.search(text or ""))


def reply_ignores_tool_summary(text: str, tool_results: list[str] | None) -> bool:
    if not tool_results:
        return False
    fallback = best_tool_summary_fallback(tool_results)
    if not fallback:
        return False
    stripped = (text or "").strip()
    if not stripped:
        return True
    if _DONE_ONLY_RE.match(stripped):
        return True
    if reply_looks_like_stale_apology(stripped):
        return True
    low = stripped.lower()
    # If the model answer is very short and omits any token from the tool summary, prefer the tool.
    sample = fallback.lower()[:40]
    if len(stripped) < 24 and sample and sample not in low:
        first_line = fallback.split("\n", 1)[0].lower()
        if first_line and first_line[:20] not in low:
            return True
    return False


def last_tool_error_user_message(tool_results: list[str] | None) -> str | None:
    if not tool_results:
        return None
    for raw in reversed(tool_results):
        data = _parse_tool_dict(raw)
        if not data:
            continue
        if data.get("ok") is True and not data.get("error"):
            continue
        if data.get("error"):
            err = str(data["error"]).strip()
            reason = data.get("failure_reason")
            if isinstance(reason, str) and reason.strip():
                return f"Spotify lookup failed ({reason}): {err}"
            return err
    return None


def reply_hallucinates_after_tool_failure(text: str, tool_results: list[str] | None) -> bool:
    """True when the last tool failed but the reply reads like a successful result list."""
    if not tool_results:
        return False
    last = _parse_tool_dict(tool_results[-1])
    if not last or last.get("ok") is not False and not last.get("error"):
        return False
    stripped = (text or "").strip()
    if not stripped:
        return False
    low = stripped.lower()
    if any(w in low for w in ("failed", "couldn't", "could not", "error", "unavailable")):
        return False
    # Numbered or bullet lists of show/track names after a failed search.
    if re.search(r"(?:^|\n)\s*(?:\d+\.|[-*])\s+\S+", stripped):
        return True
    if re.search(r"\b(?:startalk|podcast|episode)\b", low) and "failed" not in low:
        return True
    return False


def apply_tool_grounded_reply(text: str, tool_results: list[str] | None) -> str:
    """Prefer tool summaries over apologies, bare Done., or post-error hallucinations."""
    if reply_hallucinates_after_tool_failure(text, tool_results):
        err = last_tool_error_user_message(tool_results)
        if err:
            return err
    if reply_ignores_tool_summary(text, tool_results):
        fallback = best_tool_summary_fallback(tool_results)
        if fallback:
            return fallback
    return text
