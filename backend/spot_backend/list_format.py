"""Shared formatting for numbered list lines in user-visible summaries."""

from __future__ import annotations


def format_indexed_name_detail(
    index: int,
    name: str | None,
    detail: str | None,
    *,
    default_name: str = "Item",
) -> str:
    primary = (name or default_name).strip() or default_name
    extra = (detail or "").strip()
    if extra:
        return f"{index}. {primary} — {extra}"
    return f"{index}. {primary}"
