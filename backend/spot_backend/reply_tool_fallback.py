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


_BUILDER_TOOLS = frozenset(
    {
        "spotify_playlist_builder_preview",
        "spotify_playlist_builder_edit",
        "spotify_playlist_builder_commit",
    }
)
_LIST_REPLY_TOOLS = frozenset(
    {
        "spotify_saved_albums",
        "spotify_user_saved_shows",
        "spotify_user_playlists",
        "spotify_search",
        "spotify_search_playlists",
    }
)


def _tool_priority(name: str, user_text: str) -> int:
    low = (user_text or "").lower()
    if name in _BUILDER_TOOLS and any(w in low for w in ("playlist", "build", "preview", "track 3", "make it")):
        return 0
    if name in _LIST_REPLY_TOOLS:
        return 1
    if name in _BUILDER_TOOLS:
        return 2
    return 3


def best_tool_summary_fallback(
    tool_results: list[str] | None,
    *,
    user_text: str = "",
    tool_names: list[str] | None = None,
) -> str | None:
    if not tool_results:
        return None
    names = tool_names or []
    from spot_backend.turn_reply_intent import (
        classify_turn_primary_intent,
        pick_primary_tool_index,
        primary_tool_user_reply,
    )

    intent = classify_turn_primary_intent(user_text)
    idx = pick_primary_tool_index(names, list(tool_results), intent)
    if idx is not None:
        primary = primary_tool_user_reply(
            names[idx],
            tool_results[idx],
            user_text=user_text,
            intent=intent,
        )
        if primary:
            return primary
    indexed: list[tuple[int, int, str, dict[str, Any]]] = []
    for i, raw in enumerate(tool_results):
        data = _parse_tool_dict(raw)
        if not data or data.get("failure_reason") == "guard_refused":
            continue
        if data.get("error") or data.get("ok") is False:
            continue
        text = summary_text_from_tool_dict(data)
        if not text:
            continue
        tname = names[i] if i < len(names) else ""
        indexed.append((_tool_priority(tname, user_text), i, text, data))
    if not indexed:
        return None
    indexed.sort(key=lambda row: (row[0], -row[1]))
    return indexed[0][2]


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


_FAILURE_REASON_HUMAN: dict[str, str] = {
    "playback_not_verified": (
        "I called Spotify to start playback, but I couldn't confirm the player actually switched. "
        "Try tapping play on your device or ask me to transfer playback."
    ),
    "guard_refused": (
        "That was a question-only turn, so I didn't run a playback change. "
        "Ask me to play something if you want me to start it."
    ),
    "invalid_uri_type": "That Spotify URI type doesn't work for this action.",
    "unknown_id": "Spotify doesn't recognize that id — search or look it up in this chat first.",
    "no_episodes_for_show": "That podcast show has no episodes I could load.",
    "playlist_not_found": "I couldn't find a playlist with that exact name.",
    "no_exact_playlist_match": "I didn't find an exact playlist name match — pick one from the list.",
    "editorial_playlist_blocked": "Spotify editorial playlists can't be saved or modified from here.",
    "owned_playlist_protected": (
        "That's one of your own playlists. Say its exact name if you want me to remove or change it."
    ),
    "show_not_found": "I couldn't find that show's latest episode.",
    "save_not_verified": "I couldn't confirm that was saved to your Spotify library.",
    "no_episodes_for_show": "I couldn't find that show's latest episode.",
}


def humanize_failure_reason(reason: str | None, err: str | None = None) -> str:
    if isinstance(reason, str) and reason.strip():
        human = _FAILURE_REASON_HUMAN.get(reason.strip())
        if human:
            return human
    if err and err.strip():
        return err.strip()
    return "Something went wrong talking to Spotify just now. Please try again."


def last_tool_error_user_message(tool_results: list[str] | None) -> str | None:
    if not tool_results:
        return None
    for raw in reversed(tool_results):
        data = _parse_tool_dict(raw)
        if not data:
            continue
        if data.get("ok") is True and not data.get("error"):
            continue
        if data.get("error") or data.get("ok") is False:
            err = str(data.get("error") or "").strip()
            reason = data.get("failure_reason")
            if isinstance(reason, str):
                return humanize_failure_reason(reason, err)
            return humanize_failure_reason(None, err)
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
    if re.search(r"\b(?:startalk|podcast|episode|audiobook|mistborn|narrated by)\b", low) and "failed" not in low:
        return True
    last = _parse_tool_dict(tool_results[-1])
    if last and last.get("failure_reason") and str(last.get("failure_reason")).startswith("http_"):
        if re.search(r"\b(?:narrated by|audiobook)\b", low):
            return True
    if last and last.get("failure_reason") in (
        "audiobooks_unavailable_in_market",
        "invalid_market",
        "unknown_error",
    ):
        if re.search(r"(?:^|\n)\s*(?:\d+\.|[-*])\s+\S+", stripped):
            return True
        if re.search(r"\bnarrated by\b", low):
            return True
    return False


_BRAND_ONLY_REPLY_RE = re.compile(r"^(?:spotify|done\.?|ok\.?)$", re.I)


def reply_is_too_short_or_brand_only(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if _BRAND_ONLY_REPLY_RE.match(t):
        return True
    words = re.findall(r"[A-Za-z0-9']+", t)
    if len(words) <= 2 and t.lower() in ("spotify", "done", "ok"):
        return True
    if len(t) <= 16 and t.lower().replace(".", "") == "spotify":
        return True
    return False


def ensure_substantive_user_reply(
    text: str,
    tool_results: list[str] | None,
    *,
    tool_names: list[str] | None = None,
    user_text: str = "",
) -> str:
    if not reply_is_too_short_or_brand_only(text):
        return text
    fallback = best_tool_summary_fallback(
        tool_results,
        user_text=user_text,
        tool_names=tool_names or [],
    )
    if fallback:
        return fallback
    err = last_tool_error_user_message(tool_results)
    if err:
        return err
    return "Something went wrong talking to Spotify just now. Please try again."


def apply_tool_grounded_reply(
    text: str,
    tool_results: list[str] | None,
    *,
    user_text: str = "",
    tool_names: list[str] | None = None,
) -> str:
    """Prefer tool summaries over apologies, bare Done., or post-error hallucinations."""
    from spot_backend.action_claim_guard import is_failure_boilerplate
    from spot_backend.turn_reply_intent import intent_based_user_reply

    names = tool_names or []
    results = tool_results or []
    intent_reply = intent_based_user_reply(text, names, results, user_text=user_text)
    if intent_reply != text:
        return intent_reply

    if is_failure_boilerplate(text):
        fallback = best_tool_summary_fallback(results, user_text=user_text, tool_names=names)
        if fallback:
            return fallback
    if reply_hallucinates_after_tool_failure(text, results):
        err = last_tool_error_user_message(results)
        if err:
            return err
    if reply_ignores_tool_summary(text, results):
        fallback = best_tool_summary_fallback(results, user_text=user_text, tool_names=names)
        if fallback:
            return fallback
    return text
