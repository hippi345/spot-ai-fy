"""Deterministic server-side overrides for model-emitted Spotify tool arguments."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from spot_backend.spotify_tools import SpotifyToolRunner, _normalize_spotify_id, _parse_spotify_context_ref
from spot_backend.tool_server_enforcement_overrides import (
    override_album_args,
    override_play_context,
    override_track_args,
    playback_id_for_segment,
)

_CURRENT_ALBUM_RE = re.compile(r"\b(?:this|current)\s+album\b", re.I)
_CURRENT_TRACK_RE = re.compile(
    r"\b(?:this|current)\s+(?:track|song)\b|\b(?:this|current)\s+album\b", re.I
)
_CURRENT_SHOW_RE = re.compile(r"\b(?:this|current)\s+(?:show|podcast)\b", re.I)
_CURRENT_EPISODE_RE = re.compile(r"\b(?:this|current)\s+episode\b", re.I)
_IT_REFERS_PLAYBACK_RE = re.compile(
    r"\b(?:is\s+)?(?:this|it)\s+(?:saved|in my library|liked)\b", re.I
)

_PLAYLIST_SAVE_RE = re.compile(
    r"(?:save|follow|add)\s+(?:the\s+)?playlist\s+['\"]?(.+?)['\"]?\s*(?:\?|$)",
    re.I,
)
_PLAYLIST_NAMED_RE = re.compile(
    r"playlist\s+['\"]([^'\"]+)['\"]|playlist\s+([A-Za-z0-9][^?.!]{2,60})",
    re.I,
)

_FOLLOW_TOOLS = frozenset({"spotify_follow_playlist"})
_SEARCH_PLAYLIST_TOOLS = frozenset({"spotify_search_playlists"})
_LIBRARY_ALBUM_TOOLS = frozenset(
    {
        "spotify_library_contains",
        "spotify_save_albums",
        "spotify_unsave_albums",
    }
)
_PLAY_CONTEXT_TOOLS = frozenset(
    {
        "spotify_start_resume_playback",
        "spotify_play_playlist",
    }
)


def _user_wants_current_album(user_text: str) -> bool:
    t = user_text or ""
    return bool(_CURRENT_ALBUM_RE.search(t) or (_IT_REFERS_PLAYBACK_RE.search(t) and "album" in t.lower()))


def _user_wants_current_track(user_text: str) -> bool:
    t = user_text or ""
    if _CURRENT_TRACK_RE.search(t):
        return True
    return bool(_IT_REFERS_PLAYBACK_RE.search(t) and "album" not in t.lower())


def extract_requested_playlist_name(user_text: str) -> str | None:
    t = (user_text or "").strip()
    if not t:
        return None
    m = _PLAYLIST_SAVE_RE.search(t)
    if m:
        return m.group(1).strip()
    m2 = _PLAYLIST_NAMED_RE.search(t)
    if m2:
        return (m2.group(1) or m2.group(2) or "").strip()
    return None


def _clamp_playlist_search_limit(args: dict[str, Any]) -> None:
    raw_limit = args.get("limit")
    try:
        lim = int(raw_limit) if raw_limit is not None else 5
    except (TypeError, ValueError):
        lim = 5
    args["limit"] = max(5, min(lim, 10))


def _apply_playback_overrides(
    tool_name: str,
    args: dict[str, Any],
    *,
    user_text: str,
    runner: SpotifyToolRunner,
) -> dict[str, Any]:
    out = args
    if tool_name in _LIBRARY_ALBUM_TOOLS and _user_wants_current_album(user_text):
        live = playback_id_for_segment(runner, "album")
        if live:
            out = override_album_args(out, live)
    if tool_name in ("spotify_save_tracks", "spotify_unsave_tracks", "spotify_library_contains"):
        if _user_wants_current_track(user_text):
            live = playback_id_for_segment(runner, "track")
            if live:
                out = override_track_args(out, live)
        elif _user_wants_current_album(user_text):
            live = playback_id_for_segment(runner, "album")
            if live:
                out = override_album_args(out, live)
    if tool_name in ("spotify_library_save", "spotify_library_remove", "spotify_library_contains"):
        if _CURRENT_SHOW_RE.search(user_text or ""):
            show = playback_id_for_segment(runner, "show") or getattr(runner, "_last_show_search_id", None)
            if isinstance(show, str) and show.strip():
                out = deepcopy(out)
                out["uris"] = [f"spotify:show:{show.strip()}"]
        elif _CURRENT_EPISODE_RE.search(user_text or ""):
            ep = playback_id_for_segment(runner, "episode")
            if ep:
                out = deepcopy(out)
                out["uris"] = [f"spotify:episode:{ep}"]
    if tool_name in _PLAY_CONTEXT_TOOLS:
        if _user_wants_current_album(user_text):
            live = playback_id_for_segment(runner, "album")
            if live:
                out = override_play_context(out, "album", live)
        elif _user_wants_current_track(user_text):
            live = playback_id_for_segment(runner, "track")
            if live:
                out = override_play_context(out, "track", live)
    return out


def enforce_tool_arguments_for_turn(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    user_text: str,
    runner: SpotifyToolRunner,
) -> dict[str, Any]:
    """Return arguments after deterministic server overrides (never trust stale model ids)."""
    args = deepcopy(arguments) if isinstance(arguments, dict) else {}
    if tool_name in _SEARCH_PLAYLIST_TOOLS:
        _clamp_playlist_search_limit(args)
        q = args.get("query") or args.get("q")
        if isinstance(q, str) and q.strip():
            runner.note_playlist_search_query(q.strip())
    args = _apply_playback_overrides(tool_name, args, user_text=user_text, runner=runner)
    if tool_name in _FOLLOW_TOOLS:
        requested = extract_requested_playlist_name(user_text) or runner.last_playlist_search_query()
        pid = args.get("playlist_id") or args.get("id")
        if requested and isinstance(pid, str) and _normalize_spotify_id(pid, "playlist"):
            args = deepcopy(args)
            args["_follow_requested_name"] = requested.strip()
    return args


def follow_playlist_needs_name_confirmation(
    arguments: dict[str, Any],
    *,
    playlist_name_from_api: str | None,
) -> bool:
    requested = arguments.get("_follow_requested_name")
    if not isinstance(requested, str) or not requested.strip():
        return False
    if not playlist_name_from_api:
        return True
    return requested.strip().lower() != playlist_name_from_api.strip().lower()


def parse_context_album_id(arguments: dict[str, Any]) -> str | None:
    raw = arguments.get("context_uri") or arguments.get("album_id") or ""
    if isinstance(raw, str):
        parsed = _parse_spotify_context_ref(raw)
        if parsed and parsed[0] == "album":
            return parsed[1]
        bare = _normalize_spotify_id(raw, "album")
        if bare:
            return bare
    return None
