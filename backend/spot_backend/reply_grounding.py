"""Ground assistant replies in artist names returned by Spotify tools."""

from __future__ import annotations

import json
import re
from typing import Any

_QUOTED_TITLE_ARTIST_RE = re.compile(
    r"[\"'“”‘’]([^\"'“”‘’]+)[\"'“”‘’]\s+by\s+"
    r"([A-Za-z0-9'’.&\-]+(?:\s+[A-Za-z0-9'’.&\-]+){0,2})"
    r"(?=\s+is\b|[\.\?!,]|\s*$)",
    re.I,
)

# Unquoted credits like: Playing Unchained by Drake.
_UNQUOTED_TITLE_BY_ARTIST_RE = re.compile(
    r"(?:playing|queued|added|try|suggest(?:ing)?|here(?:'s| is)|found)\s+"
    r"([A-Za-z0-9][A-Za-z0-9'’.&\- ]{0,60}?)\s+by\s+"
    r"([A-Za-z0-9'’.&\-]+(?:\s+[A-Za-z0-9'’.&\-]+){0,2})"
    r"(?=[\.\?!,]|\s|$)",
    re.I,
)


def _walk_collect_tracks(node: Any, out: set[str]) -> None:
    if isinstance(node, dict):
        if node.get("type") == "track" and isinstance(node.get("name"), str):
            out.add(node["name"].strip())
        if isinstance(node.get("name"), str) and isinstance(node.get("artists"), list):
            out.add(str(node["name"]).strip())
        if isinstance(node.get("name"), str) and isinstance(node.get("uri"), str):
            if str(node["uri"]).startswith("spotify:track:"):
                out.add(str(node["name"]).strip())
        for val in node.values():
            _walk_collect_tracks(val, out)
    elif isinstance(node, list):
        for item in node:
            _walk_collect_tracks(item, out)


def track_and_artist_names_from_tool_results(tool_results: list[str] | None) -> tuple[set[str], set[str]]:
    artists = artist_names_from_tool_results(tool_results)
    tracks: set[str] = set()
    for raw in tool_results or []:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        _walk_collect_tracks(data, tracks)
        for row in data.get("results") or [] if isinstance(data, dict) else []:
            if isinstance(row, dict) and isinstance(row.get("name"), str):
                tracks.add(row["name"].strip())
    return tracks, artists


def reply_mentions_catalog_outside_tool_data(text: str, tool_results: list[str] | None) -> list[str]:
    """Return human-readable ungrounded title/artist mentions in the reply."""
    tracks, artists = track_and_artist_names_from_tool_results(tool_results)
    allowed_artists = {a.lower() for a in artists}
    allowed_tracks = {t.lower() for t in tracks}
    issues: list[str] = []

    for m in _QUOTED_TITLE_ARTIST_RE.finditer(text or ""):
        title, artist = m.group(1).strip(), m.group(2).strip()
        if title and title.lower() not in allowed_tracks:
            issues.append(f"{title} by {artist}")
        elif artist and artist.lower() not in allowed_artists:
            issues.append(f"{title} by {artist}")

    for m in _UNQUOTED_TITLE_BY_ARTIST_RE.finditer(text or ""):
        title, artist = m.group(1).strip(), m.group(2).strip()
        if title.lower() not in allowed_tracks or artist.lower() not in allowed_artists:
            issues.append(f"{title} by {artist}")

    return issues


def _walk_collect_artists(node: Any, out: set[str]) -> None:
    if isinstance(node, dict):
        if isinstance(node.get("name"), str) and (
            node.get("type") == "artist"
            or "artists" not in node
            and isinstance(node.get("id"), str)
            and len(str(node.get("id"))) == 22
        ):
            # Heuristic: artist objects in search results
            if isinstance(node.get("uri"), str) and node["uri"].startswith("spotify:artist:"):
                out.add(node["name"].strip())
        artists = node.get("artists")
        if isinstance(artists, list):
            for a in artists:
                if isinstance(a, dict) and isinstance(a.get("name"), str) and a["name"].strip():
                    out.add(a["name"].strip())
        for val in node.values():
            _walk_collect_artists(val, out)
    elif isinstance(node, list):
        for item in node:
            _walk_collect_artists(item, out)


def artist_names_from_tool_results(tool_results: list[str] | None) -> set[str]:
    names: set[str] = set()
    for raw in tool_results or []:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        _walk_collect_artists(data, names)
    return {n for n in names if n}


def reply_mentions_artists_outside_tool_data(text: str, tool_results: list[str] | None) -> list[str]:
    """Return artist names mentioned in the reply that are absent from tool JSON."""
    allowed = {a.lower() for a in artist_names_from_tool_results(tool_results)}
    if not allowed:
        return []
    mentioned: set[str] = set()
    for m in _QUOTED_TITLE_ARTIST_RE.finditer(text or ""):
        artist = m.group(2).strip()
        if artist:
            mentioned.add(artist)
    bad = [a for a in mentioned if a.lower() not in allowed]
    return bad


def ground_reply_artist_credits(text: str, tool_results: list[str] | None) -> str:
    """Replace mis-attributed artist credits with names from tool results when possible."""
    bad = reply_mentions_artists_outside_tool_data(text, tool_results)
    if not bad or not tool_results:
        return text
    allowed = sorted(artist_names_from_tool_results(tool_results), key=len, reverse=True)
    if not allowed:
        return text
    replacement = allowed[0]
    out = text
    repl_l = replacement.lower()
    for wrong in bad:
        wrong_l = wrong.lower()
        if wrong_l and wrong_l in repl_l and wrong_l != repl_l:
            # Avoid turning "John Mayer" into "John Mayer Mayer" when only a first-name
            # token was mis-parsed as the credited artist.
            continue
        out = re.sub(
            rf"\b{re.escape(wrong)}\b",
            replacement,
            out,
            count=1,
            flags=re.I,
        )
    return out
