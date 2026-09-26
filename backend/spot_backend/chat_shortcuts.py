"""Deterministic chat shortcuts (like/save this, undo, play artist) without relying on the LLM."""

from __future__ import annotations

import json
import re

from spot_backend.deterministic_chat_types import DeterministicChatResult
from spot_backend.play_artist import format_play_artist_reply, play_artist_tool_step
from spot_backend.play_artist_intent import extract_play_artist_name
from spot_backend.prompt_intent import prompt_requests_recent_listening_history
from spot_backend.queue_track_intent import extract_queue_track_request
from spot_backend.spotify_tools import SpotifyToolRunner

_LIKE_THIS_RE = re.compile(
    r"^\s*(?:please\s+)?(?:(?:like|heart)\s+this|save\s+this(?:\s+(?:song|track))?)\s*[.!?]*\s*$",
    re.I,
)
_SAVE_SONG_RE = re.compile(
    r"^\s*(?:please\s+)?save\s+this\s+(?:song|track)\s*[.!?]*\s*$",
    re.I,
)
_UNDO_RE = re.compile(r"^\s*(?:please\s+)?undo\s+that\s*[.!?]*\s*$", re.I)

_NOTHING_TO_UNDO = "There's nothing to undo yet."


def _parse_tool_json(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        data = {}
    if isinstance(data, dict) and data.get("items"):
        return data
    # Truncated tool payloads: recover a partial items array if present.
    marker = '"items"'
    idx = (raw or "").find(marker)
    if idx >= 0:
        fragment = (raw or "")[idx:]
        if not fragment.strip().endswith("}"):
            fragment = fragment.rsplit("}", 1)[0] + "]}"
        try:
            wrapped = json.loads("{" + fragment)
        except (json.JSONDecodeError, TypeError, ValueError):
            wrapped = {}
        if isinstance(wrapped, dict) and isinstance(wrapped.get("items"), list):
            return wrapped
    return data if isinstance(data, dict) else {}


def _summarize_recent_plays(items: list) -> str:
    labels: list[str] = []
    for row in items[:12]:
        if not isinstance(row, dict):
            continue
        track = row.get("track") if isinstance(row.get("track"), dict) else row
        if not isinstance(track, dict):
            continue
        name = track.get("name") if isinstance(track.get("name"), str) else "Unknown track"
        artists = track.get("artists") if isinstance(track.get("artists"), list) else []
        artist_names = [
            a.get("name")
            for a in artists
            if isinstance(a, dict) and isinstance(a.get("name"), str)
        ]
        if artist_names:
            labels.append(f"{name} — {', '.join(artist_names)}")
        else:
            labels.append(name)
    if not labels:
        return ""
    if len(labels) == 1:
        return f"Your most recent play was {labels[0]}."
    head = ", ".join(labels[:8])
    extra = len(items) - len(labels[:8])
    if extra > 0:
        return f"Recently you played: {head}, and {extra} more."
    return f"Recently you played: {head}."


def _run_tool(
    runner: SpotifyToolRunner,
    name: str,
    args: dict,
    steps: list[tuple[str, dict, str]],
) -> str:
    raw = runner.run(name, args)
    steps.append((name, args, raw))
    return raw


def try_deterministic_recently_played_reply(
    user_text: str,
    runner: SpotifyToolRunner,
) -> DeterministicChatResult | None:
    if not prompt_requests_recent_listening_history(user_text):
        return None
    steps: list[tuple[str, dict, str]] = []
    args = {"limit": 20}
    raw = _run_tool(runner, "spotify_recently_played", args, steps)
    data = _parse_tool_json(raw)
    items = data.get("items") if isinstance(data.get("items"), list) else []
    if items:
        summary = _summarize_recent_plays(items)
        reply = summary or (
            f"I pulled your {len(items)} most recent plays from Spotify — "
            "check the tool results for track details."
        )
    else:
        reply = "I couldn't find any recent listening history in your Spotify account just now."
    return DeterministicChatResult(reply, steps)


def try_deterministic_chat_reply(
    user_text: str,
    runner: SpotifyToolRunner,
    *,
    conversation_id: str | None = None,
) -> DeterministicChatResult | None:
    """Run a fixed tool chain for obvious control phrases; return structured outcome or None."""
    t = (user_text or "").strip()
    if not t:
        return None

    steps: list[tuple[str, dict, str]] = []

    if _UNDO_RE.match(t):
        track_ids = runner.last_saved_track_ids_for_undo(
            conversation_id or runner.conversation_id
        )
        if not track_ids:
            return DeterministicChatResult(_NOTHING_TO_UNDO, [])
        tid = track_ids[-1]
        raw = _run_tool(runner, "spotify_unsave_tracks", {"track_id": tid}, steps)
        data = _parse_tool_json(raw)
        if data.get("ok"):
            removed = data.get("removed_track_ids") or data.get("track_ids") or []
            if isinstance(removed, list) and removed:
                return DeterministicChatResult("Removed that track from your liked songs.", steps)
            return DeterministicChatResult("Undid the last save to your library.", steps)
        return DeterministicChatResult(
            "I could not undo the last action — nothing was changed.",
            steps,
        )

    artist_name = extract_play_artist_name(t)
    if artist_name:
        name, args, raw = play_artist_tool_step(runner, artist_name)
        steps.append((name, args, raw))
        return DeterministicChatResult(format_play_artist_reply(artist_name, raw), steps)

    queue_req = extract_queue_track_request(t)
    if queue_req:
        track_title, artist = queue_req
        qargs: dict[str, str] = {"track_name": track_title}
        if artist:
            qargs["artist_name"] = artist
        raw = _run_tool(runner, "spotify_add_to_queue", qargs, steps)
        data = _parse_tool_json(raw)
        if data.get("ok"):
            label = f"{track_title} by {artist}" if artist else track_title
            return DeterministicChatResult(f"Queued {label} on Spotify.", steps)
        err = str(data.get("error") or "I could not queue that track.")
        return DeterministicChatResult(err, steps)

    if _LIKE_THIS_RE.match(t) or _SAVE_SONG_RE.match(t):
        state_raw = _run_tool(runner, "spotify_playback_state", {}, steps)
        state = _parse_tool_json(state_raw)
        item = state.get("item") if isinstance(state.get("item"), dict) else {}
        track_name = item.get("name") if isinstance(item.get("name"), str) else "that track"
        save_raw = _run_tool(runner, "spotify_save_tracks", {"track_id": "this"}, steps)
        save = _parse_tool_json(save_raw)
        if save.get("ok"):
            return DeterministicChatResult(f"Saved “{track_name}” to your liked songs.", steps)
        err = str(save.get("error") or save.get("hint") or "")
        if "playback" in err.lower() or "nothing" in err.lower():
            return DeterministicChatResult(
                "Nothing is playing right now, so I cannot save a track. "
                "Start playback first, then ask again.",
                steps,
            )
        return DeterministicChatResult(
            "I could not save that track to your library. Try again when something is playing.",
            steps,
        )

    return None
