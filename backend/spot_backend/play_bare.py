"""Bare 'play <query>' helper — track vs artist disambiguation."""

from __future__ import annotations

import json

from spot_backend.play_artist import format_play_artist_reply, play_artist_tool_step
from spot_backend.play_track import format_play_track_chat_reply, play_track_tool_step
from spot_backend.spotify_tools import SpotifyToolRunner


def play_bare_tool_step(
    runner: SpotifyToolRunner,
    query: str,
) -> tuple[str, dict[str, str], str]:
    args = {"query": query}
    raw = runner.run("spotify_play_bare", args)
    return "spotify_play_bare", args, raw


def format_play_bare_chat_reply(query: str, raw: str) -> str:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return f"I could not start playback for {query} just now."
    if not isinstance(data, dict):
        return f"I could not start playback for {query} just now."
    mode = str(data.get("mode") or "").strip().lower()
    if mode == "artist":
        artist = str(data.get("artist_name") or query).strip()
        inner = data.get("playback_result")
        if isinstance(inner, str):
            return format_play_artist_reply(artist, inner)
    if mode == "track":
        title = str(data.get("track_name") or query).strip()
        artist = str(data.get("artist_name") or "").strip()
        inner = data.get("playback_result")
        if isinstance(inner, str):
            return format_play_track_chat_reply(title, artist, inner)
    err = str(data.get("user_message") or data.get("error") or "").strip()
    if err:
        return err
    return f"I could not start playback for {query} just now."
