"""Deterministic chat shortcuts (like/save this, undo) without relying on the LLM."""

from __future__ import annotations

import json
import re

from spot_backend.play_artist_intent import extract_play_artist_name
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


def _parse_tool_json(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def try_deterministic_chat_reply(user_text: str, runner: SpotifyToolRunner) -> str | None:
    """Run a fixed tool chain for obvious control phrases; return user text or None."""
    t = (user_text or "").strip()
    if not t:
        return None

    if _UNDO_RE.match(t):
        raw = runner.run("spotify_unsave_tracks", {"track_id": "that"})
        data = _parse_tool_json(raw)
        if data.get("ok"):
            removed = data.get("removed_track_ids") or data.get("track_ids") or []
            if isinstance(removed, list) and removed:
                return "Removed that track from your liked songs."
            return "Undid the last save to your library."
        err = str(data.get("error") or "")
        if "nothing" in err.lower() or "required" in err.lower() or not data:
            return "There is nothing from the last successful save that I can undo."
        return "I could not undo the last action — nothing was changed."

    artist_name = extract_play_artist_name(t)
    if artist_name:
        play_raw = runner.run("spotify_play_playlist", {"playlist_id": artist_name})
        play = _parse_tool_json(play_raw)
        if play.get("ok") is True or (
            isinstance(play.get("playback"), dict) and play["playback"].get("ok") is True
        ):
            return f"Playing {artist_name} on Spotify."
        err = str(play.get("error") or play.get("playback", {}).get("error") or "")
        if err:
            return f"I could not start playback for {artist_name}: {err}"
        return f"I could not start playback for {artist_name} just now."

    if _LIKE_THIS_RE.match(t) or _SAVE_SONG_RE.match(t):
        state_raw = runner.run("spotify_playback_state", {})
        state = _parse_tool_json(state_raw)
        item = state.get("item") if isinstance(state.get("item"), dict) else {}
        track_name = item.get("name") if isinstance(item.get("name"), str) else "that track"
        save_raw = runner.run("spotify_save_tracks", {"track_id": "this"})
        save = _parse_tool_json(save_raw)
        if save.get("ok"):
            return f"Saved “{track_name}” to your liked songs."
        err = str(save.get("error") or save.get("hint") or "")
        if "playback" in err.lower() or "nothing" in err.lower():
            return "Nothing is playing right now, so I cannot save a track. Start playback first, then ask again."
        return "I could not save that track to your library. Try again when something is playing."

    return None
