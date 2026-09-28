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

_UNSUPPORTED_CAPABILITY_RE = re.compile(
    r"^\s*(?:can|could|would)\s+(?:you|this)\b",
    re.I,
)

_UNSUPPORTED_PODCAST_RE = re.compile(r"\b(?:podcasts?|episodes?)\b", re.I)
_UNSUPPORTED_AUDIOBOOK_RE = re.compile(r"\baudiobooks?\b", re.I)
_UNSUPPORTED_LYRICS_RE = re.compile(r"\blyrics?\b", re.I)
_UNSUPPORTED_DOWNLOAD_RE = re.compile(r"\bdownload(?:s|ing)?\b", re.I)

_UNSUPPORTED_FEATURE_REPLIES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        _UNSUPPORTED_PODCAST_RE,
        "No — this chat plays music on Spotify only; podcasts and episodes aren't supported here.",
    ),
    (
        _UNSUPPORTED_AUDIOBOOK_RE,
        "No — audiobooks aren't available through this Spotify assistant.",
    ),
    (
        _UNSUPPORTED_LYRICS_RE,
        "No — Spotify's API doesn't expose lyrics here, so I can't show them in this chat.",
    ),
    (
        _UNSUPPORTED_DOWNLOAD_RE,
        "No — downloads and offline files aren't something I can manage via this Spotify interface.",
    ),
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
    if _UNSUPPORTED_CAPABILITY_RE.match(t):
        for pattern, reply in _UNSUPPORTED_FEATURE_REPLIES:
            if pattern.search(t):
                if _UNSUPPORTED_PODCAST_RE.search(t) and re.search(
                    r"\bplay\s+(?:the\s+)?(?:episode|podcast)\s+[\"']?\w",
                    t,
                    re.I,
                ):
                    continue
                return reply
    return None
