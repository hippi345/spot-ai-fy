"""Resolve podcast show ids from session context and user text (never trust invented ids)."""

from __future__ import annotations

import re
from typing import Any

from spot_backend.spotify_tools import SpotifyToolRunner, _looks_like_spotify_catalog_id, _normalize_spotify_id

_SHOW_NAME_FROM_PLAY_RE = re.compile(
    r"\b(?:latest|newest)\s+episode\s+of\s+(.+?)(?:[?.!]|$)",
    re.I,
)
_SHOW_NAME_TOKEN_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9'’\-\s]{2,40})\b",
)


def extract_show_name_hint(user_text: str) -> str | None:
    t = (user_text or "").strip()
    if not t:
        return None
    m = _SHOW_NAME_FROM_PLAY_RE.search(t)
    if m:
        return m.group(1).strip()
    if re.search(r"\bstartalk\b", t, re.I):
        return "StarTalk"
    return None


def _name_matches_hint(show_name: str, hint: str) -> bool:
    a = (show_name or "").strip().lower()
    b = (hint or "").strip().lower()
    if not a or not b:
        return False
    if a == b or b in a or a in b:
        return True
    a_tokens = {w for w in re.findall(r"[a-z0-9']+", a) if len(w) > 2}
    b_tokens = {w for w in re.findall(r"[a-z0-9']+", b) if len(w) > 2}
    return bool(a_tokens and b_tokens and (a_tokens <= b_tokens or b_tokens <= a_tokens))


def _session_show_catalog(runner: SpotifyToolRunner) -> list[tuple[str, str]]:
    catalog = getattr(runner, "_session_show_catalog", None)
    if not isinstance(catalog, list):
        return []
    out: list[tuple[str, str]] = []
    for row in catalog:
        if isinstance(row, dict):
            sid = row.get("id")
            name = row.get("name")
            if isinstance(sid, str) and isinstance(name, str) and sid.strip() and name.strip():
                out.append((sid.strip(), name.strip()))
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            out.append((str(row[0]).strip(), str(row[1]).strip()))
    return out


def _show_id_known_to_session(runner: SpotifyToolRunner, show_id: str) -> bool:
    sid = (show_id or "").strip()
    if not _looks_like_spotify_catalog_id(sid):
        return False
    known = getattr(runner, "_session_known_ids", None)
    if isinstance(known, set) and sid in known:
        return True
    for cid, _ in _session_show_catalog(runner):
        if cid == sid:
            return True
    last = getattr(runner, "_last_show_search_id", None)
    return isinstance(last, str) and last.strip() == sid


def _match_show_from_session(runner: SpotifyToolRunner, hint: str) -> str | None:
    for sid, name in reversed(_session_show_catalog(runner)):
        if _name_matches_hint(name, hint):
            return sid
    last = getattr(runner, "_last_show_search_id", None)
    if isinstance(last, str) and last.strip():
        if not hint:
            return last.strip()
        for sid, name in _session_show_catalog(runner):
            if sid == last.strip() and _name_matches_hint(name, hint):
                return sid
        if len(_session_show_catalog(runner)) == 0:
            return last.strip()
    mut = getattr(runner, "_last_library_mutation", None)
    if isinstance(mut, dict) and mut.get("segment") == "show":
        ids = mut.get("ids")
        if isinstance(ids, list) and ids:
            return str(ids[-1]).strip()
    return None


def _search_show_id_by_name(runner: SpotifyToolRunner, hint: str) -> str | None:
    q = (hint or "").strip()
    if not q:
        return None
    try:
        data = runner.client.api_get(
            "/search",
            params={"q": q, "type": "show", "limit": 5},
        )
    except Exception:
        return None
    shows = data.get("shows") if isinstance(data, dict) else None
    items = shows.get("items") if isinstance(shows, dict) else None
    if not isinstance(items, list):
        return None
    best_id: str | None = None
    best_name: str | None = None
    for row in items:
        if not isinstance(row, dict):
            continue
        sid = row.get("id")
        name = row.get("name")
        if not isinstance(sid, str) or not isinstance(name, str):
            continue
        if _name_matches_hint(name, q):
            return sid.strip()
        if best_id is None:
            best_id, best_name = sid.strip(), name.strip()
    if best_id and best_name and _name_matches_hint(best_name, q):
        return best_id
    if best_id and len(items) == 1:
        return best_id
    return None


def resolve_show_id_for_turn(
    runner: SpotifyToolRunner,
    user_text: str,
    candidate: str | None,
) -> str | None:
    """Pick a show id: keep verified session ids; else match name from user text."""
    raw = (candidate or "").strip()
    norm = _normalize_spotify_id(raw, "show") if raw else None
    if norm and _show_id_known_to_session(runner, norm):
        return norm
    hint = extract_show_name_hint(user_text) or ""
    if not hint:
        tokens = _SHOW_NAME_TOKEN_RE.findall(user_text or "")
        for tok in reversed(tokens):
            if tok.lower() not in ("play", "the", "latest", "episode", "podcast", "show"):
                hint = tok
                break
    from_session = _match_show_from_session(runner, hint) if hint else _match_show_from_session(runner, "")
    if from_session:
        return from_session
    searched = _search_show_id_by_name(runner, hint) if hint else None
    if searched:
        if hasattr(runner, "note_session_show"):
            runner.note_session_show(searched, hint or searched)
        return searched
    if norm and _show_id_known_to_session(runner, norm):
        return norm
    return None
