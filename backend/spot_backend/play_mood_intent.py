"""Bare 'play something <mood>' requests — act immediately without LLM confirmation."""

from __future__ import annotations

import re

_PLAY_SOMETHING_MOOD_RE = re.compile(
    r"^\s*play\s+something\s+([a-z][a-z0-9'-]{1,24})\s*[.!?]*\s*$",
    re.I,
)


def extract_play_something_mood(user_text: str) -> str | None:
    t = (user_text or "").strip()
    m = _PLAY_SOMETHING_MOOD_RE.match(t)
    if not m:
        return None
    mood = (m.group(1) or "").strip()
    if not mood:
        return None
    low = mood.lower()
    if low in {"by", "from", "on", "in", "at", "for", "with"}:
        return None
    return mood
