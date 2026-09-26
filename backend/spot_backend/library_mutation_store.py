"""Per-conversation last library mutation (survives across chat turns / runner instances)."""

from __future__ import annotations

from typing import Any

_store: dict[str, dict[str, Any]] = {}


def _key(conversation_id: str | None) -> str | None:
    cid = (conversation_id or "").strip()
    return cid if cid else None


def load_last_library_mutation(conversation_id: str | None) -> dict[str, Any] | None:
    key = _key(conversation_id)
    if key is None:
        return None
    mut = _store.get(key)
    return mut if isinstance(mut, dict) else None


def record_library_mutation(
    conversation_id: str | None,
    segment: str,
    ids: list[str],
) -> None:
    key = _key(conversation_id)
    if key is None:
        return
    clean = [i for i in ids if isinstance(i, str) and i.strip()]
    if not clean:
        return
    _store[key] = {"segment": segment, "ids": clean}


def last_saved_track_ids(conversation_id: str | None) -> list[str]:
    mut = load_last_library_mutation(conversation_id)
    if not isinstance(mut, dict) or mut.get("segment") != "track":
        return []
    ids = mut.get("ids")
    if not isinstance(ids, list):
        return []
    return [str(i) for i in ids if str(i).strip()]


def clear_session(conversation_id: str | None) -> None:
    """Test helper — drop stored mutation for a conversation key."""
    key = _key(conversation_id)
    if key is not None:
        _store.pop(key, None)
