"""Format user-visible replies from Spotify /me/player payloads."""

from __future__ import annotations

import re
from typing import Any

_BY_SUFFIX_RE = re.compile(r"\s+by\s+.+$", re.I)


def _artist_names(artists: Any) -> list[str]:
    if not isinstance(artists, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for a in artists:
        if not isinstance(a, dict):
            continue
        name = a.get("name")
        if not isinstance(name, str):
            continue
        cleaned = name.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _strip_redundant_by_suffix(track_name: str, artist_names: list[str]) -> str:
    """Drop a trailing ' by …' on the title when it repeats credited artists."""
    name = (track_name or "").strip()
    if not name or not artist_names:
        return name
    m = _BY_SUFFIX_RE.search(name)
    if not m:
        return name
    suffix = m.group(0).strip().lower()
    credited = ", ".join(artist_names).lower()
    if credited and credited in suffix:
        return name[: m.start()].strip()
    return name


def format_now_playing_chat_reply(player: dict[str, Any] | None) -> str:
    """Plain-language answer for 'what's playing?' style questions."""
    if not player or not isinstance(player, dict):
        return "Nothing is playing on Spotify right now."
    item = player.get("item") if isinstance(player.get("item"), dict) else None
    if not item:
        return "Nothing is playing on Spotify right now."
    title = item.get("name") if isinstance(item.get("name"), str) else "Unknown track"
    artists = _artist_names(item.get("artists"))
    title = _strip_redundant_by_suffix(title, artists)
    if artists:
        credit = ", ".join(artists)
        return f"You're listening to {title} by {credit}."
    return f"You're listening to {title}."


def describe_playing_item(item: dict[str, Any] | None) -> tuple[str, str]:
    """Return (title, artist_credit) for a Spotify track object."""
    if not item or not isinstance(item, dict):
        return "Unknown track", ""
    title = item.get("name") if isinstance(item.get("name"), str) else "Unknown track"
    artists = _artist_names(item.get("artists"))
    title = _strip_redundant_by_suffix(title, artists)
    credit = ", ".join(artists) if artists else ""
    return title, credit


def format_play_track_reply(
    requested_title: str,
    requested_artist: str,
    player: dict[str, Any] | None,
    *,
    playback_verified: bool,
) -> str:
    """User-visible play reply based on read-back, not the search guess."""
    if not player or not isinstance(player, dict):
        if playback_verified:
            label = f"{requested_title} by {requested_artist}".strip()
            return f"Playing {label} on Spotify."
        return (
            f"I tried to play {requested_title} by {requested_artist}, "
            "but I could not confirm playback on your device."
        )
    item = player.get("item") if isinstance(player.get("item"), dict) else None
    if not item:
        return (
            f"I tried to play {requested_title} by {requested_artist}, "
            "but nothing is playing on Spotify right now."
        )
    actual_title, actual_credit = describe_playing_item(item)
    req_title = requested_title.strip()
    req_artist = requested_artist.strip()
    req_norm = f"{req_title} by {req_artist}".lower()
    actual_norm = f"{actual_title} by {actual_credit}".lower() if actual_credit else actual_title.lower()
    if req_norm in actual_norm or actual_norm in req_norm:
        if actual_credit:
            return f"Playing {actual_title} by {actual_credit}."
        return f"Playing {actual_title}."
    if actual_credit:
        return (
            f"Now playing {actual_title} by {actual_credit} "
            f"(you asked for {req_title} by {req_artist})."
        )
    return (
        f"Now playing {actual_title} (you asked for {req_title} by {req_artist})."
    )


def format_skip_reply(
    player: dict[str, Any] | None,
    *,
    skipped: bool,
    direction: str = "next",
) -> str:
    if not skipped:
        return (
            "I sent skip to Spotify, but the track did not change after a few seconds. "
            "Try skipping once in the Spotify app, then ask again."
        )
    if not player or not isinstance(player, dict):
        return f"Skipped to the {direction} track."
    item = player.get("item") if isinstance(player.get("item"), dict) else None
    if not item:
        return f"Skipped to the {direction} track."
    title, credit = describe_playing_item(item)
    if credit:
        return f"Skipped — now playing {title} by {credit}."
    return f"Skipped — now playing {title}."


def prompt_asks_skip(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    if not t:
        return False
    if "playlist" in t:
        return False
    return bool(re.fullmatch(r"(?:please\s+)?(?:skip(?:\s+(?:this|the))?(?:\s+song|\s+track)?|next(?:\s+song|\s+track)?)", t))


def prompt_asks_whats_playing(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    if not t:
        return False
    return bool(
        re.search(
            r"\bwhat(?:'s|s| is)\s+playing\b|\bnow playing\b|\bcurrent(?:ly)?\s+playing\b|\bwhat song\b",
            t,
        )
    )
