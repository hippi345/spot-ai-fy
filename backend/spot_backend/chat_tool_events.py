"""Chat stream tool events (tool_done payloads safe for live eval traces)."""

from __future__ import annotations

from typing import Any

from spot_backend.llm_secret_safety import redact_known_api_keys

_PLAY_TOOL_NAMES = frozenset(
    {
        "spotify_play_track",
        "spotify_add_to_queue",
        "spotify_play_next",
    }
)

_TITLE_KEYS = ("track_name", "title", "name")
_ARTIST_KEYS = ("artist_name", "artist")


def _uri_from_play_args(args: dict[str, Any]) -> str:
    uri = args.get("uri")
    if isinstance(uri, str) and uri.strip():
        return uri.strip()
    uris = args.get("uris")
    if isinstance(uris, list):
        for entry in uris:
            if isinstance(entry, str) and entry.strip().startswith("spotify:"):
                return entry.strip()
    return ""


def play_tool_arguments_for_event_trace(
    tool_name: str,
    arguments: dict[str, Any] | None,
) -> dict[str, str]:
    """Title/artist/uri only — for live eval BlankTrackTitleRejected (no tokens or API keys)."""
    if tool_name not in _PLAY_TOOL_NAMES:
        return {}
    args = arguments if isinstance(arguments, dict) else {}
    has_title_key = any(k in args for k in _TITLE_KEYS)
    has_artist_key = any(k in args for k in _ARTIST_KEYS)
    if not has_title_key and not has_artist_key:
        uri_only = _uri_from_play_args(args)
        if uri_only:
            return {"uri": redact_known_api_keys(uri_only)}
        return {}
    out: dict[str, str] = {}
    if has_title_key:
        raw_title = args.get("track_name") or args.get("title") or args.get("name") or ""
        out["track_name"] = redact_known_api_keys(str(raw_title))
    if has_artist_key:
        raw_artist = args.get("artist_name") or args.get("artist") or ""
        out["artist_name"] = redact_known_api_keys(str(raw_artist))
    uri = _uri_from_play_args(args)
    if uri:
        out["uri"] = redact_known_api_keys(uri)
    return out


def tool_done_event(
    name: str,
    preview: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ev: dict[str, Any] = {"type": "tool_done", "name": name, "preview": preview}
    trace_args = play_tool_arguments_for_event_trace(name, arguments)
    if trace_args:
        ev["arguments"] = trace_args
    return ev
