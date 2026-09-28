"""Short deterministic answers for app capability questions."""

from __future__ import annotations

import re

_WHAT_CAN_YOU_DO_RE = re.compile(
    r"^\s*what\s+can\s+you\s+do\b",
    re.I,
)

_MAKE_PLAYLIST_CAPABILITY_RE = re.compile(
    r"^\s*(?:can|could|would)\s+you\s+(?:please\s+)?(?:make|create)\s+(?:me\s+)?(?:a\s+)?playlists?\s*\??\s*$",
    re.I,
)


def try_capability_question_reply(user_text: str) -> str | None:
    t = (user_text or "").strip()
    if not t:
        return None
    if _WHAT_CAN_YOU_DO_RE.match(t):
        return (
            "I can search Spotify, play tracks, artists, albums, and your playlists, "
            "and help with queue, shuffle, repeat, and your library. "
            "Tell me what you want to hear or ask what's playing."
        )
    if _MAKE_PLAYLIST_CAPABILITY_RE.match(t):
        return (
            "Yes — I can put together a playlist for you. "
            "What vibe or artists should I start with?"
        )
    return None
