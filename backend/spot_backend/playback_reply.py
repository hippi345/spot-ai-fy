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
