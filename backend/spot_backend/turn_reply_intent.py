"""Classify user-turn intent and pick the primary Spotify tool result for replies."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from spot_backend.reply_tool_fallback import (
    humanize_failure_reason,
    summary_text_from_tool_dict,
    tool_result_has_actionable_summary,
)

_PARSE = json.JSONDecoder()


class TurnPrimaryIntent(str, Enum):
    QUESTION_SAVED = "question_saved"
    LIST_READ = "list_read"
    ACTION_PLAY = "action_play"
    ACTION_SAVE = "action_save"
    ACTION_REMOVE = "action_remove"
    ACTION_BUILD = "action_build"
    UNKNOWN = "unknown"


_QUESTION_SAVED_RE = re.compile(
    r"\b(?:"
    r"is\s+(?:this|that|it)\s+(?:saved|in my library|liked|followed)"
    r"|(?:do i|have i)\s+(?:already\s+)?(?:got|have)\s+(?:this|that|it|(?:this|that|the)\s+\w+)\s+saved"
    r"|(?:is|are)\s+(?:this|that)\s+(?:album|track|song|show|playlist|podcast)\s+saved"
    r"|is it saved"
    r")\b",
    re.I,
)
_LIST_READ_RE = re.compile(
    r"\b(?:"
    r"what\s+(?:albums|playlists|podcasts|shows)\s+do i"
    r"|(?:list|show)\s+(?:my|the)\s+(?:saved|followed)"
    r"|what\s+do i follow"
    r"|what\s+albums\s+do i have"
    r")\b",
    re.I,
)
_PLAY_RE = re.compile(
    r"\b(?:play|start|resume|listen to)\b",
    re.I,
)
_SAVE_RE = re.compile(
    r"\b(?:save|follow|add)\b.+\b(?:library|liked|show|album|playlist|podcast)\b"
    r"|\b(?:save|follow)\s+(?:this|that|it)\b",
    re.I,
)
_REMOVE_RE = re.compile(
    r"\b(?:remove|unfollow|unsave|delete)\b.+\b(?:library|liked|show|album|playlist)\b"
    r"|\b(?:remove|unfollow|unsave)\s+(?:this|that|it)\b",
    re.I,
)
_BUILD_RE = re.compile(
    r"\b(?:build|create|make)\b.+\bplaylist\b|\bplaylist builder\b|\bpreview\b",
    re.I,
)

_GUARD_BOILERPLATE_RE = re.compile(
    r"question-only turn|didn't run a playback change|didn't run a spotify action",
    re.I,
)

_INTENT_TOOLS: dict[TurnPrimaryIntent, tuple[str, ...]] = {
    TurnPrimaryIntent.QUESTION_SAVED: ("spotify_library_contains",),
    TurnPrimaryIntent.LIST_READ: (
        "spotify_saved_albums",
        "spotify_user_saved_shows",
        "spotify_user_playlists",
        "spotify_followed_artists",
        "spotify_recently_played",
    ),
    TurnPrimaryIntent.ACTION_PLAY: (
        "spotify_play_show_latest_episode",
        "spotify_play_track",
        "spotify_play_playlist",
        "spotify_start_resume_playback",
        "spotify_play_artist",
        "spotify_play_artist_popular_track",
        "spotify_play_artist_latest_release",
        "spotify_play_next",
        "spotify_add_to_queue",
    ),
    TurnPrimaryIntent.ACTION_SAVE: (
        "spotify_library_save",
        "spotify_save_tracks",
        "spotify_save_albums",
        "spotify_follow_playlist",
        "spotify_follow_artist",
        "spotify_add_tracks_to_playlist",
    ),
    TurnPrimaryIntent.ACTION_REMOVE: (
        "spotify_library_remove",
        "spotify_unsave_tracks",
        "spotify_unsave_albums",
        "spotify_unfollow_playlist",
        "spotify_unfollow_artist",
    ),
    TurnPrimaryIntent.ACTION_BUILD: (
        "spotify_playlist_builder_commit",
        "spotify_playlist_builder_edit",
        "spotify_playlist_builder_preview",
    ),
}

_PLAYBACK_FAILURE_TOOLS = frozenset(_INTENT_TOOLS[TurnPrimaryIntent.ACTION_PLAY])
_MUTATION_SUCCESS_TOOLS = frozenset(
    _INTENT_TOOLS[TurnPrimaryIntent.ACTION_SAVE] + _INTENT_TOOLS[TurnPrimaryIntent.ACTION_REMOVE]
)


def classify_turn_primary_intent(user_text: str) -> TurnPrimaryIntent:
    t = (user_text or "").strip()
    if not t:
        return TurnPrimaryIntent.UNKNOWN
    if _QUESTION_SAVED_RE.search(t):
        return TurnPrimaryIntent.QUESTION_SAVED
    if _LIST_READ_RE.search(t):
        return TurnPrimaryIntent.LIST_READ
    if _BUILD_RE.search(t):
        return TurnPrimaryIntent.ACTION_BUILD
    if _REMOVE_RE.search(t):
        return TurnPrimaryIntent.ACTION_REMOVE
    if _SAVE_RE.search(t):
        return TurnPrimaryIntent.ACTION_SAVE
    if _PLAY_RE.search(t):
        return TurnPrimaryIntent.ACTION_PLAY
    return TurnPrimaryIntent.UNKNOWN


def _parse_tool_dict(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _tool_succeeded(raw: str) -> bool:
    data = _parse_tool_dict(raw)
    if not data:
        return False
    if data.get("error") or data.get("ok") is False:
        return False
    if data.get("failure_reason") == "guard_refused":
        return False
    return True


def _tool_is_guard_refusal(raw: str) -> bool:
    data = _parse_tool_dict(raw)
    return bool(data and data.get("failure_reason") == "guard_refused")


def pick_primary_tool_index(
    tool_names: list[str],
    tool_results: list[str],
    intent: TurnPrimaryIntent,
) -> int | None:
    if not tool_names or not tool_results:
        return None
    preferred = _INTENT_TOOLS.get(intent, ())
    if not preferred:
        return None
    best: int | None = None
    for i, name in enumerate(tool_names):
        if name not in preferred:
            continue
        raw = tool_results[i] if i < len(tool_results) else ""
        if intent in (TurnPrimaryIntent.ACTION_PLAY, TurnPrimaryIntent.ACTION_SAVE, TurnPrimaryIntent.ACTION_REMOVE):
            best = i
            continue
        if intent == TurnPrimaryIntent.QUESTION_SAVED and name == "spotify_library_contains":
            if _tool_succeeded(raw) or _parse_tool_dict(raw):
                best = i
            continue
        if _tool_succeeded(raw):
            best = i
    return best


def _play_failure_user_message(data: dict[str, Any]) -> str:
    custom = data.get("user_message")
    if isinstance(custom, str) and custom.strip():
        return custom.strip()
    reason = data.get("failure_reason")
    if reason == "playback_not_verified":
        player = data.get("player_after")
        if isinstance(player, dict):
            item = player.get("item")
            if isinstance(item, dict) and item.get("type") in ("track", "episode"):
                name = item.get("name") if isinstance(item.get("name"), str) else None
                artists = item.get("artists") if isinstance(item.get("artists"), list) else []
                artist = ""
                if artists and isinstance(artists[0], dict):
                    artist = str(artists[0].get("name") or "").strip()
                if name and artist:
                    return f"Spotify didn't switch; it's still playing {name} by {artist}."
                if name:
                    return f"Spotify didn't switch; it's still playing {name}."
        ep_name = None
        for key in ("episode_name", "episode_title"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                ep_name = val.strip()
                break
        item = data.get("item") if isinstance(data.get("item"), dict) else None
        if not ep_name and item and isinstance(item.get("name"), str):
            ep_name = item["name"].strip()
        show = data.get("show_name") if isinstance(data.get("show_name"), str) else None
        if ep_name:
            return (
                f"I found the latest episode, {ep_name!r}, but Spotify didn't confirm it started playing. "
                "Try tapping play on your device or ask me to transfer playback."
            )
        if show:
            return (
                f"I found {show}'s latest episode, but Spotify didn't confirm it started playing. "
                "Try tapping play on your device."
            )
        return humanize_failure_reason("playback_not_verified", str(data.get("error") or ""))
    return humanize_failure_reason(
        reason if isinstance(reason, str) else None,
        str(data.get("error") or ""),
    )


def _library_mutation_confirmation(data: dict[str, Any], *, removed: bool) -> str | None:
    if removed and data.get("not_in_library"):
        custom = data.get("user_message")
        if isinstance(custom, str) and custom.strip():
            return custom.strip()
    uris = data.get("removed_uris") if removed else data.get("saved_uris")
    if not isinstance(uris, list) or not uris:
        return None
    uri = uris[0] if isinstance(uris[0], str) else ""
    if not uri.startswith("spotify:"):
        return None
    seg = uri.split(":", 2)[1]
    label = data.get("item_name") or data.get("name")
    if isinstance(label, str) and label.strip():
        name = label.strip()
    else:
        name = f"that {seg}"
    if removed:
        if data.get("ok") is True and data.get("verified_removed") is False:
            return f"Removed {name} from your library, but Spotify may still show it as saved."
        return f"Removed {name} from your library."
    return f"Saved {name} to your library."


def primary_tool_user_reply(
    tool_name: str,
    raw: str,
    *,
    user_text: str = "",
    intent: TurnPrimaryIntent | None = None,
) -> str | None:
    data = _parse_tool_dict(raw)
    if not data:
        return None
    intent = intent or classify_turn_primary_intent(user_text)
    if tool_name in _PLAYBACK_FAILURE_TOOLS and (data.get("ok") is False or data.get("failure_reason")):
        if data.get("failure_reason") == "guard_refused":
            return None
        return _play_failure_user_message(data)
    if tool_name in _MUTATION_SUCCESS_TOOLS and data.get("ok") is True:
        removed = tool_name in _INTENT_TOOLS[TurnPrimaryIntent.ACTION_REMOVE]
        msg = _library_mutation_confirmation(data, removed=removed)
        if msg:
            return msg
    if tool_name == "spotify_library_contains" and data.get("ok") is True:
        custom = data.get("user_message")
        if isinstance(custom, str) and custom.strip():
            if intent == TurnPrimaryIntent.QUESTION_SAVED:
                return custom.strip()
    if _tool_succeeded(raw) and tool_result_has_actionable_summary(data):
        text = summary_text_from_tool_dict(data)
        if text and not _looks_like_internal_hint(text):
            return text
    if data.get("ok") is False and tool_name in _PLAYBACK_FAILURE_TOOLS:
        return _play_failure_user_message(data)
    if data.get("ok") is False and tool_name in _MUTATION_SUCCESS_TOOLS:
        if data.get("failure_reason") == "owned_playlist_protected":
            err = data.get("error")
            if isinstance(err, str) and err.strip():
                return err.strip()
        err = humanize_failure_reason(
            data.get("failure_reason") if isinstance(data.get("failure_reason"), str) else None,
            str(data.get("error") or ""),
        )
        return err
    return None


def _looks_like_internal_hint(text: str) -> bool:
    low = text.lower()
    if "offset=" in low or "pass offset" in low:
        return True
    if "spotify_" in low:
        return True
    return False


def model_reply_is_guard_boilerplate(text: str) -> bool:
    return bool(_GUARD_BOILERPLATE_RE.search(text or ""))


def model_reply_consistent_with_primary(model_text: str, primary: str) -> bool:
    model = (model_text or "").strip()
    primary_s = (primary or "").strip()
    if not primary_s:
        return True
    if not model:
        return False
    if model_reply_is_guard_boilerplate(model):
        return False
    low = model.lower()
    if primary_s.lower() in low:
        return True
    first = primary_s.split("\n", 1)[0].lower()
    if len(first) > 12 and first[: min(24, len(first))] in low:
        return True
    if re.search(r"\b(?:yes|no)\b", primary_s.lower()) and re.search(r"\b(?:yes|no)\b", low):
        return primary_s.split()[0].lower() == model.split()[0].lower()
    return False


def intent_based_user_reply(
    model_text: str,
    tool_names: list[str],
    tool_results: list[str],
    *,
    user_text: str = "",
) -> str:
    intent = classify_turn_primary_intent(user_text)
    idx = pick_primary_tool_index(tool_names, tool_results, intent)
    if idx is None:
        return model_text
    name = tool_names[idx]
    raw = tool_results[idx]
    primary = primary_tool_user_reply(name, raw, user_text=user_text, intent=intent)
    if not primary:
        return model_text
    if model_reply_consistent_with_primary(model_text, primary):
        return model_text
    return primary


def intent_needs_library_contains_fallback(user_text: str, tool_names: list[str]) -> bool:
    if classify_turn_primary_intent(user_text) != TurnPrimaryIntent.QUESTION_SAVED:
        return False
    return not any(n == "spotify_library_contains" for n in tool_names)


def try_deterministic_reply_after_tools(
    user_text: str,
    tool_names: list[str],
    tool_results: list[str],
) -> str | None:
    """When primary intent is satisfied by tools, skip another LLM round."""
    intent = classify_turn_primary_intent(user_text)
    if intent == TurnPrimaryIntent.UNKNOWN:
        return None
    idx = pick_primary_tool_index(tool_names, tool_results, intent)
    if idx is None:
        return None
    name = tool_names[idx]
    raw = tool_results[idx]
    if intent == TurnPrimaryIntent.QUESTION_SAVED:
        data = _parse_tool_dict(raw)
        if not data or data.get("ok") is not True:
            return None
        if name != "spotify_library_contains":
            return None
        reply = primary_tool_user_reply(name, raw, user_text=user_text, intent=intent)
        return reply
    if intent == TurnPrimaryIntent.LIST_READ:
        data = _parse_tool_dict(raw)
        if not data or not _tool_succeeded(raw):
            return None
        reply = primary_tool_user_reply(name, raw, user_text=user_text, intent=intent)
        if reply:
            return reply
        return None
    if intent in (
        TurnPrimaryIntent.ACTION_SAVE,
        TurnPrimaryIntent.ACTION_REMOVE,
    ):
        data = _parse_tool_dict(raw)
        if not data or data.get("ok") is not True:
            return None
        if data.get("failure_reason") in ("guard_refused", "owned_playlist_protected"):
            return None
        reply = primary_tool_user_reply(name, raw, user_text=user_text, intent=intent)
        return reply
    return None


_SAVE_THIS_SHOW_RE = re.compile(
    r"\b(?:save|follow|add)\s+(?:this|that)\s+(?:show|podcast)\b",
    re.I,
)


def intent_needs_library_save_fallback(user_text: str, tool_names: list[str]) -> bool:
    if classify_turn_primary_intent(user_text) != TurnPrimaryIntent.ACTION_SAVE:
        return False
    if any(n == "spotify_library_save" for n in tool_names):
        return False
    t = user_text or ""
    if _SAVE_THIS_SHOW_RE.search(t):
        return True
    return bool(re.search(r"\bthis\s+(?:show|podcast)\b", t, re.I) and "save" in t.lower())


def library_save_fallback_args(user_text: str) -> dict[str, Any]:
    low = (user_text or "").lower()
    if "episode" in low:
        return {"uris": ["this episode"]}
    return {"uris": ["this show"]}


def library_contains_fallback_args(user_text: str) -> dict[str, Any]:
    low = (user_text or "").lower()
    if "album" in low and ("this" in low or "current" in low):
        return {"uris": ["this album"]}
    if re.search(r"\b(?:this|that)\s+(?:show|podcast)\b", low):
        return {"uris": ["this show"]}
    if re.search(r"\b(?:this|that)\s+(?:track|song)\b", low):
        return {"uris": ["this track"]}
    if re.search(r"\bit\b", low):
        return {"uris": ["it"]}
    return {"uris": ["this track"]}
