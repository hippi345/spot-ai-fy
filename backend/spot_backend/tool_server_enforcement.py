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
_IT_SAVED_QUESTION_RE = re.compile(
    r"\b(?:is\s+)?(?:this|it|that)\s+(?:saved|in my library|liked|followed)\b|\bis it saved\b",
    re.I,
)
_IT_REFERS_PLAYBACK_RE = re.compile(
    r"\b(?:is\s+)?(?:this|it)\s+(?:saved|in my library|liked)\b", re.I
)
_DEICTIC_REMOVE_RE = re.compile(
    r"\b(?:remove|unfollow|unsave|delete)\s+(?:it|that|this)(?:\s+from\b|\s*$|\?)",
    re.I,
)
_DEICTIC_SAVE_RE = re.compile(
    r"\b(?:save|follow|add)\s+(?:it|that|this)(?:\s+to\b|\s*$|\?)",
    re.I,
)


def _user_asks_it_or_last_mutation_saved(user_text: str) -> bool:
    t = user_text or ""
    if not _IT_SAVED_QUESTION_RE.search(t):
        return False
    if _CURRENT_ALBUM_RE.search(t) or "album" in t.lower():
        return False
    return bool(re.search(r"\bit\b", t, re.I) or "that" in t.lower())


def _apply_last_mutation_library_override(
    tool_name: str,
    args: dict[str, Any],
    runner: SpotifyToolRunner,
) -> dict[str, Any]:
    mut = getattr(runner, "_last_library_mutation", None)
    if not isinstance(mut, dict):
        return args
    segment = mut.get("segment")
    ids = mut.get("ids")
    if not isinstance(segment, str) or not isinstance(ids, list) or not ids:
        return args
    bare = str(ids[0]).strip()
    if not bare:
        return args
    out = deepcopy(args)
    uri = f"spotify:{segment}:{bare}"
    out["uris"] = [uri]
    if segment == "album":
        out = override_album_args(out, bare)
    elif segment == "track":
        out = override_track_args(out, bare)
    elif segment == "playlist":
        out["uris"] = [uri]
    return out

_PLAYLIST_SAVE_RE = re.compile(
    r"(?:save|follow|add)\s+(?:the\s+)?playlist\s+['\"]?(.+?)['\"]?\s*(?:\?|$)",
    re.I,
)
_PLAYLIST_NAMED_RE = re.compile(
    r"playlist\s+['\"]([^'\"]+)['\"]|playlist\s+([A-Za-z0-9][^?.!]{2,60})",
    re.I,
)

_FOLLOW_TOOLS = frozenset({"spotify_follow_playlist"})
_SHOW_PLAY_TOOLS = frozenset({"spotify_play_show_latest_episode", "spotify_get_show", "spotify_get_show_episodes"})
_PLAYLIST_ID_TOOLS = frozenset(
    {
        "spotify_unfollow_playlist",
        "spotify_remove_playlist_tracks",
        "spotify_update_playlist",
        "spotify_replace_playlist_tracks",
        "spotify_reorder_playlist_tracks",
        "spotify_get_playlist",
        "spotify_playlist_tracks",
        "spotify_add_tracks_to_playlist",
    }
)
_SHOW_SAVE_INTENT_RE = re.compile(
    r"\bsave\b.+\b(?:this|that)\s+(?:show|podcast)\b|\bsave\s+(?:this|that)\s+(?:show|podcast)\b",
    re.I,
)
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


def _user_names_specific_playlist(user_text: str) -> bool:
    return bool(extract_requested_playlist_name(user_text))


def _user_refers_deictic_library_item(user_text: str) -> bool:
    t = user_text or ""
    if _user_names_specific_playlist(t):
        return False
    if _CURRENT_ALBUM_RE.search(t) or _CURRENT_TRACK_RE.search(t):
        return False
    if _CURRENT_SHOW_RE.search(t) or _CURRENT_EPISODE_RE.search(t):
        return False
    if _DEICTIC_REMOVE_RE.search(t) or _DEICTIC_SAVE_RE.search(t):
        return True
    if _user_asks_it_or_last_mutation_saved(t):
        return True
    if re.search(r"\b(?:this|that)\s+playlist\b", t, re.I):
        return True
    return bool(re.search(r"\b(?:remove|unfollow|unsave)\s+(?:it|that)\b", t, re.I))


def _preferred_playlist_id_from_context(runner: SpotifyToolRunner) -> str | None:
    mut = getattr(runner, "_last_library_mutation", None)
    if isinstance(mut, dict) and mut.get("segment") == "playlist":
        ids = mut.get("ids")
        if isinstance(ids, list) and ids:
            bare = str(ids[-1]).strip()
            if bare:
                return bare
    sess = getattr(runner, "_last_session_playlist_id", None)
    if isinstance(sess, str) and sess.strip():
        return sess.strip()
    return None


def _override_playlist_id_from_context(
    arguments: dict[str, Any],
    *,
    user_text: str,
    runner: SpotifyToolRunner,
) -> dict[str, Any]:
    if not _user_refers_deictic_library_item(user_text):
        return arguments
    preferred = _preferred_playlist_id_from_context(runner)
    if not preferred:
        return arguments
    raw = arguments.get("playlist_id") or arguments.get("id")
    if isinstance(raw, str):
        norm = _normalize_spotify_id(raw, "playlist")
        if norm == preferred:
            return arguments
    out = deepcopy(arguments)
    out["playlist_id"] = preferred
    return out


def _user_wants_current_track(user_text: str) -> bool:
    t = user_text or ""
    if _CURRENT_ALBUM_RE.search(t):
        return False
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
        if _user_asks_it_or_last_mutation_saved(user_text):
            out = _apply_last_mutation_library_override(tool_name, out, runner)
        elif _user_wants_current_track(user_text):
            live = playback_id_for_segment(runner, "track")
            if live:
                out = override_track_args(out, live)
        elif _user_wants_current_album(user_text):
            live = playback_id_for_segment(runner, "album")
            if live:
                out = override_album_args(out, live)
    if tool_name in ("spotify_library_save", "spotify_library_remove", "spotify_library_contains"):
        if _CURRENT_SHOW_RE.search(user_text or "") or re.search(
            r"\b(?:this|that)\s+(?:show|podcast)\b", user_text or "", re.I
        ):
            from spot_backend.show_session_resolve import session_show_id

            show = session_show_id(runner)
            if not show:
                show = playback_id_for_segment(runner, "show")
            if not show:
                show = getattr(runner, "_last_show_search_id", None)
            if not show:
                mut = getattr(runner, "_last_library_mutation", None)
                if isinstance(mut, dict) and mut.get("segment") == "show":
                    ids = mut.get("ids")
                    if isinstance(ids, list) and ids:
                        show = str(ids[-1]).strip()
            if isinstance(show, str) and show.strip():
                out = deepcopy(out)
                out["uris"] = [f"spotify:show:{show.strip()}"]
        elif _user_refers_deictic_library_item(user_text or ""):
            out = _apply_last_mutation_library_override(tool_name, out, runner)
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


def _resolved_show_id_for_save(runner: SpotifyToolRunner) -> str | None:
    from spot_backend.show_session_resolve import resolve_show_id_for_turn

    return resolve_show_id_for_turn(runner, "", None)


def _uris_are_show_segment(uris: list[Any]) -> bool:
    for raw in uris:
        if not isinstance(raw, str):
            continue
        parsed = _parse_spotify_context_ref(raw.strip())
        if parsed and parsed[0] == "show":
            return True
    return False


def _show_save_rewrite_to_library(
    arguments: dict[str, Any],
    *,
    user_text: str,
    runner: SpotifyToolRunner,
) -> tuple[str, dict[str, Any]] | None:
    from spot_backend.show_session_resolve import session_show_id

    show_id = session_show_id(runner) or _resolved_show_id_for_save(runner)
    if not show_id:
        return None
    out = deepcopy(arguments) if isinstance(arguments, dict) else {}
    out["uris"] = [f"spotify:show:{show_id}"]
    out["_turn_user_text"] = user_text
    return "spotify_library_save", out


def _maybe_rewrite_show_save_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    user_text: str,
    runner: SpotifyToolRunner,
) -> tuple[str, dict[str, Any]] | None:
    if tool_name == "spotify_save_tracks":
        ids = arguments.get("track_ids") or arguments.get("ids") or arguments.get("track_id")
        blob = ids if isinstance(ids, list) else [ids] if isinstance(ids, str) else []
        if any(isinstance(x, str) and "show:" in x.lower() for x in blob):
            return _show_save_rewrite_to_library(arguments, user_text=user_text, runner=runner)
    if tool_name not in ("spotify_save_tracks", "spotify_save_albums", "spotify_library_save"):
        return None
    uris = arguments.get("uris")
    if not isinstance(uris, list) or not uris:
        return _show_save_rewrite_to_library(arguments, user_text=user_text, runner=runner)
    from spot_backend.show_session_resolve import session_show_id

    session = session_show_id(runner)
    if not session:
        if not _uris_are_show_segment(uris):
            return _show_save_rewrite_to_library(arguments, user_text=user_text, runner=runner)
        return None
    pronoun_only = all(
        isinstance(x, str)
        and x.strip().lower()
        in (
            "it",
            "this",
            "that",
            "this show",
            "that show",
            "this podcast",
            "that podcast",
        )
        for x in uris
    )
    if pronoun_only or not _uris_are_show_segment(uris):
        return _show_save_rewrite_to_library(arguments, user_text=user_text, runner=runner)
    return None


def rewrite_tool_call_for_turn(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    user_text: str,
    runner: SpotifyToolRunner,
) -> tuple[str, dict[str, Any]]:
    """Rewrite wrong tool/args before the first Spotify call on show-save and similar turns."""
    from spot_backend.show_session_resolve import is_show_library_intent

    if is_show_library_intent(user_text, runner):
        rewritten = _maybe_rewrite_show_save_tool_call(
            tool_name, arguments, user_text=user_text, runner=runner
        )
        if rewritten:
            return rewritten
    elif _SHOW_SAVE_INTENT_RE.search(user_text or ""):
        rewritten = _maybe_rewrite_show_save_tool_call(
            tool_name, arguments, user_text=user_text, runner=runner
        )
        if rewritten:
            return rewritten
    return tool_name, arguments if isinstance(arguments, dict) else {}


def enforce_tool_arguments_for_turn(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    user_text: str,
    runner: SpotifyToolRunner,
) -> dict[str, Any]:
    """Return arguments after deterministic server overrides (never trust stale model ids)."""
    args = deepcopy(arguments) if isinstance(arguments, dict) else {}
    args["_turn_user_text"] = user_text
    if tool_name in _SEARCH_PLAYLIST_TOOLS:
        _clamp_playlist_search_limit(args)
        q = args.get("query") or args.get("q")
        if isinstance(q, str) and q.strip():
            runner.note_playlist_search_query(q.strip())
    args = _apply_playback_overrides(tool_name, args, user_text=user_text, runner=runner)
    if tool_name in (
        "spotify_library_save",
        "spotify_library_remove",
        "spotify_library_contains",
        "spotify_save_tracks",
        "spotify_save_albums",
    ):
        from spot_backend.show_session_resolve import is_show_library_intent, rewrite_library_uris_for_show_intent

        if is_show_library_intent(user_text, runner):
            raw_uris = args.get("uris")
            if isinstance(raw_uris, list):
                args = deepcopy(args)
                args["uris"] = rewrite_library_uris_for_show_intent(runner, user_text, raw_uris)
            for key in ("track_ids", "ids", "track_id", "album_ids", "album_id"):
                val = args.get(key)
                if isinstance(val, list):
                    args = deepcopy(args)
                    args[key] = rewrite_library_uris_for_show_intent(runner, user_text, val)
                elif isinstance(val, str) and val.strip():
                    args = deepcopy(args)
                    args[key] = rewrite_library_uris_for_show_intent(runner, user_text, [val])[0]
    if tool_name in _PLAYLIST_ID_TOOLS:
        args = _override_playlist_id_from_context(args, user_text=user_text, runner=runner)
        from spot_backend.playlist_session_resolve import resolve_playlist_id_for_turn

        raw_pid = args.get("playlist_id") or args.get("id")
        resolved = resolve_playlist_id_for_turn(
            runner,
            user_text,
            str(raw_pid) if raw_pid is not None else None,
        )
        if resolved:
            args = deepcopy(args)
            args["playlist_id"] = resolved
    if tool_name in _SHOW_PLAY_TOOLS:
        from spot_backend.show_session_resolve import resolve_show_id_for_turn

        raw_sid = args.get("show_id") or args.get("id")
        resolved = resolve_show_id_for_turn(
            runner,
            user_text,
            str(raw_sid) if raw_sid is not None else None,
        )
        if resolved:
            args = deepcopy(args)
            args["show_id"] = resolved
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
