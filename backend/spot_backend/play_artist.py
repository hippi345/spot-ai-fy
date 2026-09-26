"""User-facing artist play helper — one code path for shortcuts and tools."""

from __future__ import annotations

import json

from spot_backend.spotify_tools import SpotifyToolRunner


def play_artist_tool_step(runner: SpotifyToolRunner, artist_name: str) -> tuple[str, dict[str, str], str]:
    args = {"artist_name": artist_name}
    raw = runner.run("spotify_play_artist", args)
    return "spotify_play_artist", args, raw


def format_play_artist_reply(artist_name: str, raw: str) -> str:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return f"I could not start playback for {artist_name} just now."
    if not isinstance(data, dict):
        return f"I could not start playback for {artist_name} just now."
    if data.get("ok") is True:
        return f"Playing {artist_name} on Spotify."
    err = str(data.get("error") or data.get("user_message") or "")
    if err and "playback.error" not in err and "playback.detail" not in err:
        return f"I could not start playback for {artist_name}: {err}"
    return (
        f"I could not start playback for {artist_name} just now. "
        "Try opening Spotify on your device and pressing play, then ask again."
    )
