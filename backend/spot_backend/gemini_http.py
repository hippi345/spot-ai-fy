"""Gemini REST auth — API key in header only (never query string)."""

from __future__ import annotations


def gemini_rest_headers(api_key: str) -> dict[str, str]:
    key = (api_key or "").strip()
    if not key:
        return {}
    return {"x-goog-api-key": key}
