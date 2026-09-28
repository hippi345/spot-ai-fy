"""Block Spotify Web API routes removed in dev-mode (Feb 2026 / Nov 2024 migrations)."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

API_PREFIX = "https://api.spotify.com/v1"

# Paths that must never be called (method-sensitive where noted).
_REMOVED_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("GET", re.compile(r"^/browse/new-releases/?$")),
    ("GET", re.compile(r"^/browse/categories/?$")),
    ("GET", re.compile(r"^/browse/categories/[^/]+/?$")),
    ("GET", re.compile(r"^/browse/categories/[^/]+/playlists/?$")),
    ("GET", re.compile(r"^/browse/featured-playlists/?$")),
    ("GET", re.compile(r"^/recommendations/?$")),
    ("GET", re.compile(r"^/recommendations/available-genre-seeds/?$")),
    ("GET", re.compile(r"^/audio-features/?$")),
    ("GET", re.compile(r"^/audio-analysis/[^/]+/?$")),
    ("GET", re.compile(r"^/markets/?$")),
    ("GET", re.compile(r"^/users/[^/]+/?$")),
    ("GET", re.compile(r"^/users/[^/]+/playlists/?$")),
    ("POST", re.compile(r"^/users/[^/]+/playlists/?$")),
    ("GET", re.compile(r"^/artists/[^/]+/top-tracks/?$")),
    ("GET", re.compile(r"^/artists/[^/]+/related-artists/?$")),
    ("PUT", re.compile(r"^/me/tracks/?$")),
    ("DELETE", re.compile(r"^/me/tracks/?$")),
    ("PUT", re.compile(r"^/me/albums/?$")),
    ("DELETE", re.compile(r"^/me/albums/?$")),
    ("PUT", re.compile(r"^/me/episodes/?$")),
    ("DELETE", re.compile(r"^/me/episodes/?$")),
    ("PUT", re.compile(r"^/me/shows/?$")),
    ("DELETE", re.compile(r"^/me/shows/?$")),
    ("PUT", re.compile(r"^/me/audiobooks/?$")),
    ("DELETE", re.compile(r"^/me/audiobooks/?$")),
    ("GET", re.compile(r"^/me/tracks/contains/?$")),
    ("GET", re.compile(r"^/me/albums/contains/?$")),
    ("GET", re.compile(r"^/me/episodes/contains/?$")),
    ("GET", re.compile(r"^/me/shows/contains/?$")),
    ("GET", re.compile(r"^/me/audiobooks/contains/?$")),
    ("GET", re.compile(r"^/me/following/contains/?$")),
    ("PUT", re.compile(r"^/me/following/?$")),
    ("DELETE", re.compile(r"^/me/following/?$")),
    ("PUT", re.compile(r"^/playlists/[^/]+/followers/?$")),
    ("DELETE", re.compile(r"^/playlists/[^/]+/followers/?$")),
    ("GET", re.compile(r"^/playlists/[^/]+/followers/contains/?$")),
    ("GET", re.compile(r"^/playlists/[^/]+/tracks/?$")),
    ("PUT", re.compile(r"^/playlists/[^/]+/tracks/?$")),
    ("DELETE", re.compile(r"^/playlists/[^/]+/tracks/?$")),
]

_BATCH_COLLECTIONS = frozenset(
    {"albums", "artists", "audiobooks", "chapters", "episodes", "shows", "tracks"}
)


class SpotifyRemovedRouteError(RuntimeError):
    """Attempted call to a removed Spotify Web API route."""

    def __init__(self, method: str, path: str, reason: str = "removed_route") -> None:
        self.method = method.upper()
        self.path = path
        self.failure_reason = reason
        super().__init__(f"Removed Spotify route blocked: {self.method} {path}")


def _normalize_path(url: str) -> str:
    if url.startswith("http"):
        parsed = urlparse(url)
        path = parsed.path or ""
        if path.startswith("/v1"):
            return path[len("/v1") :] or "/"
        return path or "/"
    path = url if url.startswith("/") else f"/{url}"
    if path.startswith("/v1"):
        return path[len("/v1") :] or "/"
    return path


def _query_params(url: str, params: dict[str, Any] | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if params:
        for key, val in params.items():
            if val is None:
                continue
            if isinstance(val, (list, tuple)):
                out[str(key)] = [str(v) for v in val]
            else:
                out[str(key)] = [str(val)]
    if url.startswith("http"):
        parsed = urlparse(url)
        for key, vals in parse_qs(parsed.query, keep_blank_values=True).items():
            out.setdefault(key, vals)
    return out


def spotify_route_is_removed(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
) -> str | None:
    """Return a failure_reason string when the call must be blocked, else None."""
    m = method.upper()
    path = _normalize_path(url)
    qp = _query_params(url, params)

    if m == "GET":
        parts = [p for p in path.split("/") if p]
        if len(parts) == 1 and parts[0] in _BATCH_COLLECTIONS and qp.get("ids"):
            return "removed_batch_get"
        if path.rstrip("/") == "/audio-features" and qp.get("ids"):
            return "removed_batch_get"

    for rule_method, pattern in _REMOVED_PATTERNS:
        if rule_method != m:
            continue
        if pattern.match(path.rstrip("/") or "/"):
            return "removed_route"
    return None


def assert_spotify_route_allowed(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
) -> None:
    reason = spotify_route_is_removed(method, url, params=params)
    if reason:
        raise SpotifyRemovedRouteError(method, _normalize_path(url), reason)
