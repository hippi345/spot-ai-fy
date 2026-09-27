"""User-facing track play helper — shortcuts and formatted replies."""

from __future__ import annotations

import json

from spot_backend.playback_reply import format_play_track_reply
from spot_backend.spotify_tools import SpotifyToolRunner


def play_track_tool_step(
    runner: SpotifyToolRunner,
    track_title: str,
    artist_name: str,
) -> tuple[str, dict[str, str], str]:
    args = {"track_name": track_title, "artist_name": artist_name}
    raw = runner.run("spotify_play_track", args)
    return "spotify_play_track", args, raw


def format_play_track_chat_reply(track_title: str, artist_name: str, raw: str) -> str:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return f"I could not start playback for {track_title} by {artist_name} just now."
    if not isinstance(data, dict):
        return f"I could not start playback for {track_title} by {artist_name} just now."
    player = data.get("player_after")
    verified = bool(data.get("playback_verified"))
    if data.get("ok") is True or verified:
        return format_play_track_reply(
            track_title,
            artist_name,
            player if isinstance(player, dict) else None,
            playback_verified=verified,
        )
    err = str(data.get("error") or data.get("user_message") or "").strip()
    if err:
        return err
    return format_play_track_reply(
        track_title,
        artist_name,
        player if isinstance(player, dict) else None,
        playback_verified=False,
    )
