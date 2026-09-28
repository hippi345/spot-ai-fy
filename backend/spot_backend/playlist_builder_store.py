"""Pending playlist-builder preview state (survives across chat turns)."""

from __future__ import annotations

from typing import Any

_store: dict[str, dict[str, Any]] = {}


def _key(conversation_id: str | None) -> str | None:
    cid = (conversation_id or "").strip()
    return cid if cid else None


def save_playlist_preview(conversation_id: str | None, preview: dict[str, Any]) -> None:
    key = _key(conversation_id)
    if key is None:
        return
    _store[key] = dict(preview)


def load_playlist_preview(conversation_id: str | None) -> dict[str, Any] | None:
    key = _key(conversation_id)
    if key is None:
        return None
    row = _store.get(key)
    return dict(row) if isinstance(row, dict) else None


def clear_playlist_preview(conversation_id: str | None) -> None:
    key = _key(conversation_id)
    if key is not None:
        _store.pop(key, None)


def clear_session(conversation_id: str | None) -> None:
    clear_playlist_preview(conversation_id)
