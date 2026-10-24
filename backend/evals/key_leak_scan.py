"""Scan text for API key patterns (used in eval guards and tests)."""

from __future__ import annotations

import re
from typing import Iterable

_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    re.compile(r"key=[^&\s'\"]+", re.I),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(r"xai-[A-Za-z0-9_-]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}", re.I),
)


def find_key_leaks(text: str, extra_secrets: Iterable[str] = ()) -> list[str]:
    issues: list[str] = []
    for pat in _KEY_PATTERNS:
        for m in pat.finditer(text or ""):
            issues.append(f"pattern {pat.pattern}: {m.group(0)[:12]}…")
    for secret in extra_secrets:
        s = (secret or "").strip()
        if len(s) >= 12 and s in (text or ""):
            issues.append("literal secret substring")
    return issues


def assert_no_key_leaks(text: str, extra_secrets: Iterable[str] = ()) -> None:
    hits = find_key_leaks(text, extra_secrets)
    if hits:
        raise AssertionError("API key leak detected: " + "; ".join(hits[:5]))
