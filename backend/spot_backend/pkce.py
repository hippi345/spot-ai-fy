"""Spot-AI-fy: PKCE helpers for Spotify OAuth without a client secret."""

from __future__ import annotations

import base64
import hashlib
import secrets


def new_pkce_params() -> tuple[str, str, str]:
    """Return (code_verifier, code_challenge, state) for Spotify PKCE (S256)."""
    ver = secrets.token_urlsafe(64)
    while len(ver) < 43:
        ver += secrets.token_urlsafe(16)
    if len(ver) > 128:
        ver = ver[:128]
    digest = hashlib.sha256(ver.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    state = secrets.token_urlsafe(24)
    return ver, challenge, state
