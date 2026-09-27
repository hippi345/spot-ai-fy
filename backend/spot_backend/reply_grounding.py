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
