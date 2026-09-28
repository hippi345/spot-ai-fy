"""Redact API keys from log lines and user-facing error strings."""

from __future__ import annotations

import re
from typing import Iterable

from spot_backend.llm_catalog import SECRET_KEY_BY_PROVIDER

# Typical OpenAI / Anthropic / Google / xAI key shapes (never log full values).
_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{8,}"),
    re.compile(r"xai-[A-Za-z0-9_-]{8,}"),
)


def redact_known_api_keys(text: str, extra_secrets: Iterable[str] = ()) -> str:
    if not text:
        return text
    out = text
    for pat in _KEY_PATTERNS:
        out = pat.sub("••••••••", out)
    for secret in extra_secrets:
        s = (secret or "").strip()
        if len(s) >= 8 and s in out:
            out = out.replace(s, "••••••••")
    return out


def all_secret_env_field_names() -> frozenset[str]:
    return frozenset(SECRET_KEY_BY_PROVIDER.values())
