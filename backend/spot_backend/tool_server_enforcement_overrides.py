"""Argument override helpers for tool_server_enforcement."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from spot_backend.spotify_tools import SpotifyToolRunner


def playback_id_for_segment(runner: SpotifyToolRunner, segment: str) -> str | None:
    return runner.playback_catalog_id(segment)


def override_album_args(arguments: dict[str, Any], live_id: str) -> dict[str, Any]:
    out = deepcopy(arguments)
    for key in ("album_id", "album_ids", "ids", "uri", "uris"):
        if key not in out:
            continue
        val = out[key]
        if key in ("uris", "album_ids", "ids") and isinstance(val, list):
            out[key] = [f"spotify:album:{live_id}"]
        elif isinstance(val, str) and val.strip():
            out[key] = f"spotify:album:{live_id}" if key in ("uri", "uris") else live_id
    if "uris" not in out and "album_id" not in out and "album_ids" not in out:
        out["uris"] = [f"spotify:album:{live_id}"]
    return out


def override_track_args(arguments: dict[str, Any], live_id: str) -> dict[str, Any]:
    out = deepcopy(arguments)
    for key in ("track_id", "track_ids", "ids", "uri", "uris"):
        if key not in out:
            continue
        val = out[key]
        if key in ("uris", "track_ids", "ids") and isinstance(val, list):
            out[key] = [f"spotify:track:{live_id}"]
        elif isinstance(val, str) and val.strip():
            out[key] = live_id
    return out


def override_play_context(arguments: dict[str, Any], segment: str, live_id: str) -> dict[str, Any]:
    out = deepcopy(arguments)
    uri = f"spotify:{segment}:{live_id}"
    if "context_uri" in out or segment != "track":
        out["context_uri"] = uri
    if segment == "track":
        out["uris"] = [uri]
    return out
