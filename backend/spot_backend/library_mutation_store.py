"""Per-conversation last library mutation (survives across chat turns / runner instances)."""

from __future__ import annotations

from typing import Any

_store: dict[str, dict[str, Any]] = {}
_playlist_id_store: dict[str, str] = {}
_played_playlist_ids_store: dict[str, list[str]] = {}
_focus_show_store: dict[str, dict[str, str]] = {}


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


def record_last_playlist_id(conversation_id: str | None, playlist_id: str) -> None:
    key = _key(conversation_id)
    pid = (playlist_id or "").strip()
    if key is None or not pid:
        return
    _playlist_id_store[key] = pid


def load_last_playlist_id(conversation_id: str | None) -> str | None:
    key = _key(conversation_id)
    if key is None:
        return None
    pid = _playlist_id_store.get(key)
    return pid if isinstance(pid, str) and pid.strip() else None


def record_played_playlist_id(conversation_id: str | None, playlist_id: str) -> None:
    key = _key(conversation_id)
    pid = (playlist_id or "").strip()
    if key is None or not pid:
        return
    seen = _played_playlist_ids_store.setdefault(key, [])
    if pid not in seen:
        seen.append(pid)


def load_played_playlist_ids(conversation_id: str | None) -> list[str]:
    key = _key(conversation_id)
    if key is None:
        return []
    rows = _played_playlist_ids_store.get(key)
    if not isinstance(rows, list):
        return []
    return [str(r).strip() for r in rows if str(r).strip()]


def record_session_focus_show(
    conversation_id: str | None,
    show_id: str,
    name: str = "",
) -> None:
    key = _key(conversation_id)
    sid = (show_id or "").strip()
    label = (name or "").strip()
    if key is None or not sid:
        return
    _focus_show_store[key] = {"id": sid, "name": label or sid}


def load_session_focus_show(conversation_id: str | None) -> tuple[str, str] | None:
    key = _key(conversation_id)
    if key is None:
        return None
    row = _focus_show_store.get(key)
    if not isinstance(row, dict):
        return None
    sid = row.get("id")
    name = row.get("name")
    if not isinstance(sid, str) or not sid.strip():
        return None
    label = name.strip() if isinstance(name, str) and name.strip() else sid.strip()
    return sid.strip(), label


def clear_session(conversation_id: str | None) -> None:
    """Test helper — drop stored mutation for a conversation key."""
    key = _key(conversation_id)
    if key is not None:
        _store.pop(key, None)
        _playlist_id_store.pop(key, None)
        _played_playlist_ids_store.pop(key, None)
        _focus_show_store.pop(key, None)
