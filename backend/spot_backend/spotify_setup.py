"""Spotify developer-app fields validated during in-app setup."""

from __future__ import annotations

import re

_SPOTIFY_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9]{32}$")


def validate_spotify_client_id(raw: str) -> str:
    cid = (raw or "").strip()
    if not cid:
        raise ValueError("client_id is required")
    if not _SPOTIFY_CLIENT_ID_RE.fullmatch(cid):
        raise ValueError("Spotify Client ID must be exactly 32 alphanumeric characters")
    return cid
