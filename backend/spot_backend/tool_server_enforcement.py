"""Deterministic server-side overrides for model-emitted Spotify tool arguments."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from spot_backend.spotify_tools import SpotifyToolRunner, _normalize_spotify_id, _parse_spotify_context_ref

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
    if _IT_REFERS_PLAYBACK_RE.search(t) and "album" not in t.lower():
        return True
    return False


def _user_wants_current_show(user_text: str) -> bool:
    return bool(_CURRENT_SHOW_RE.search(user_text or ""))


def _user_wants_current_episode(user_text: str) -> bool:
    return bool(_CURRENT_EPISODE_RE.search(user_text or ""))


def _playback_id(runner: SpotifyToolRunner, segment: str) -> str | None:
    if hasattr(runner, "_playback_catalog_id"):
        return runner._playback_catalog_id(segment)
    return None


def _override_album_args(arguments: dict[str, Any], live_id: str) -> dict[str, Any]:
    out = deepcopy(arguments)
    for key in ("album_id", "album_ids", "ids", "uri", "uris"):
        if key not in out:
            continue
        val = out[key]
        if key in ("uris", "album_ids", "ids") and isinstance(val, list):
            out[key] = [f"spotify:album:{live_id}"]
        elif isinstance(val, str):
            low = val.strip().lower()
            if low in ("this album", "this", "current album", "current") or val.strip():
                out[key] = f"spotify:album:{live_id}" if key in ("uri", "uris") else live_id
    if "uris" not in out and "album_id" not in out and "album_ids" not in out:
        out["uris"] = [f"spotify:album:{live_id}"]
    return out


def _override_track_args(arguments: dict[str, Any], live_id: str) -> dict[str, Any]:
    out = deepcopy(arguments)
    for key in ("track_id", "track_ids", "ids", "uri", "uris"):
        if key not in out:
            continue
        val = out[key]
        if key in ("uris", "track_ids", "ids") and isinstance(val, list):
            out[key] = [f"spotify:track:{live_id}"]
        elif isinstance(val, str):
            out[key] = live_id
    return out


def _override_play_context(arguments: dict[str, Any], segment: str, live_id: str) -> dict[str, Any]:
    out = deepcopy(arguments)
    uri = f"spotify:{segment}:{live_id}"
    if "context_uri" in out or segment != "track":
        out["context_uri"] = uri
    if segment == "track":
        out["uris"] = [uri]
    return out


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
        raw_limit = args.get("limit")
        try:
            lim = int(raw_limit) if raw_limit is not None else 5
        except (TypeError, ValueError):
            lim = 5
        args["limit"] = max(5, min(lim, 10))
        q = args.get("query") or args.get("q")
        if isinstance(q, str) and q.strip():
            runner._last_playlist_search_query = q.strip()  # type: ignore[attr-defined]

    if tool_name in _LIBRARY_ALBUM_TOOLS and _user_wants_current_album(user_text):
        live = _playback_id(runner, "album")
        if live:
            args = _override_album_args(args, live)

    if tool_name in ("spotify_save_tracks", "spotify_unsave_tracks", "spotify_library_contains"):
        if _user_wants_current_track(user_text):
            live = _playback_id(runner, "track")
            if live:
                args = _override_track_args(args, live)
        elif _user_wants_current_album(user_text):
            live = _playback_id(runner, "album")
            if live:
                args = _override_album_args(args, live)

    if tool_name in ("spotify_library_save", "spotify_library_remove", "spotify_library_contains"):
        if _user_wants_current_show(user_text):
            show = _playback_id(runner, "show")
            if not show:
                last = getattr(runner, "_last_show_search_id", None)
                if isinstance(last, str):
                    show = last
            if show:
                args = deepcopy(args)
                args["uris"] = [f"spotify:show:{show}"]
        elif _user_wants_current_episode(user_text):
            ep = _playback_id(runner, "episode")
            if ep:
                args = deepcopy(args)
                args["uris"] = [f"spotify:episode:{ep}"]

    if tool_name in _PLAY_CONTEXT_TOOLS:
        if _user_wants_current_album(user_text):
            live = _playback_id(runner, "album")
            if live:
                args = _override_play_context(args, "album", live)
        elif _user_wants_current_track(user_text):
            live = _playback_id(runner, "track")
            if live:
                args = _override_play_context(args, "track", live)

    if tool_name in _FOLLOW_TOOLS:
        requested = extract_requested_playlist_name(user_text)
        if not requested:
            requested = getattr(runner, "_last_playlist_search_query", None)
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
