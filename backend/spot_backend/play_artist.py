"""User-facing artist play helper — one code path for shortcuts and tools."""

from __future__ import annotations

import json

from spot_backend.playback_reply import describe_playing_item
from spot_backend.spotify_tools import SpotifyToolRunner


def play_artist_tool_step(runner: SpotifyToolRunner, artist_name: str) -> tuple[str, dict[str, str], str]:
    args = {"artist_name": artist_name}
    raw = runner.run("spotify_play_artist", args)
    return "spotify_play_artist", args, raw


def _artist_credit_matches_requested(player: dict | None, requested: str) -> bool:
    if not player or not isinstance(player, dict):
        return False
    item = player.get("item") if isinstance(player.get("item"), dict) else None
    if not item:
        return False
    _, credit = describe_playing_item(item)
    req = requested.strip().lower()
    if not req:
        return False
    return req in credit.lower()


def format_play_artist_reply(artist_name: str, raw: str) -> str:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return f"I could not start playback for {artist_name} just now."
    if not isinstance(data, dict):
        return f"I could not start playback for {artist_name} just now."
    resolved_name = str(data.get("artist_name") or artist_name).strip()
    playback = data.get("playback") if isinstance(data.get("playback"), dict) else {}
    player = data.get("player_after")
    if not isinstance(player, dict):
        player = playback.get("player_after") if isinstance(playback.get("player_after"), dict) else None
    verified = bool(
        data.get("playback_verified") is True
        or playback.get("playback_verified") is True
    )
    if data.get("ok") is True and verified and _artist_credit_matches_requested(player, resolved_name):
        title, credit = describe_playing_item(
            player.get("item") if isinstance(player, dict) else None
        )
        if credit:
            return f"Playing {title} by {credit}."
        return f"Playing {title}."
    if data.get("ok") is True and not verified:
        return (
            f"I tried to start {resolved_name}, but I could not confirm playback on your device."
        )
    err = str(
        data.get("user_message")
        or data.get("error")
        or (data.get("playback") or {}).get("user_message")
        or (data.get("playback") or {}).get("error")
        or ""
    )
    if err.startswith("Spotify wouldn't play"):
        return err
    if err and "playback.error" not in err and "playback.detail" not in err:
        return f"I could not start playback for {artist_name}: {err}"
    return (
        f"I could not start playback for {artist_name} just now. "
        "Try opening Spotify on your device and pressing play, then ask again."
    )
