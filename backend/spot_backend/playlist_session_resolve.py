"""Resolve playlist ids from session context and user text (never trust invented ids)."""

from __future__ import annotations

import re
from typing import Any

from spot_backend.chat_tool_state import is_playlist_pronoun_reference
from spot_backend.spotify_tools import SpotifyToolRunner, _looks_like_spotify_catalog_id, _normalize_spotify_id
from spot_backend.tool_server_enforcement import extract_requested_playlist_name

_PLAYLIST_NAME_IN_TEXT_RE = re.compile(
    r"\b(?:my\s+)?(?:new\s+)?(?:test\s+)?playlist\s+['\"]?([^'\"?.!]+?)['\"]?(?:\s*\?|$)",
    re.I,
)


def _session_playlist_ids(runner: SpotifyToolRunner) -> set[str]:
    catalog = getattr(runner, "_session_playlist_ids", None)
    if isinstance(catalog, set):
        return catalog
    out: set[str] = set()
    last = getattr(runner, "_last_session_playlist_id", None)
    if isinstance(last, str) and last.strip():
        out.add(last.strip())
    mut = getattr(runner, "_last_library_mutation", None)
    if isinstance(mut, dict) and mut.get("segment") == "playlist":
        ids = mut.get("ids")
        if isinstance(ids, list):
            for i in ids:
                if isinstance(i, str) and i.strip():
                    out.add(i.strip())
    return out


def _playlist_id_known_to_session(runner: SpotifyToolRunner, playlist_id: str) -> bool:
    pid = (playlist_id or "").strip()
    if not _looks_like_spotify_catalog_id(pid):
        return False
    if pid in _session_playlist_ids(runner):
        return True
    known = getattr(runner, "_session_known_ids", None)
    if isinstance(known, set) and pid in known:
        last = getattr(runner, "_last_session_playlist_id", None)
        if isinstance(last, str) and last == pid:
            return True
    return False


def _name_matches_hint(playlist_name: str, hint: str) -> bool:
    a = (playlist_name or "").strip().lower()
    b = (hint or "").strip().lower()
    if not a or not b:
        return False
    if a == b or b in a or a in b:
        return True
    a_tokens = {w for w in re.findall(r"[a-z0-9']+", a) if len(w) > 2}
    b_tokens = {w for w in re.findall(r"[a-z0-9']+", b) if len(w) > 2}
    return bool(a_tokens and b_tokens and (a_tokens <= b_tokens or b_tokens <= a_tokens))


def _match_playlist_from_user_library(runner: SpotifyToolRunner, hint: str) -> str | None:
    q = (hint or "").strip()
    if not q:
        return None
    try:
        page = runner.client.api_get("/me/playlists", params={"limit": 50})
    except Exception:
        return None
    items = page.get("items") if isinstance(page, dict) else None
    if not isinstance(items, list):
        return None
    exact: list[str] = []
    fuzzy: list[tuple[str, str]] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        pid = row.get("id")
        name = row.get("name")
        if not isinstance(pid, str) or not isinstance(name, str):
            continue
        if name.strip().lower() == q.lower():
            exact.append(pid.strip())
        elif _name_matches_hint(name, q):
            fuzzy.append((pid.strip(), name.strip()))
    if len(exact) == 1:
        return exact[0]
    if len(fuzzy) == 1:
        return fuzzy[0][0]
    return None


def resolve_playlist_id_for_turn(
    runner: SpotifyToolRunner,
    user_text: str,
    candidate: str | None,
) -> str | None:
    """Pick a playlist id: keep verified session ids; else session context or name match."""
    raw = (candidate or "").strip()
    if is_playlist_pronoun_reference(raw):
        raw = ""
    norm = _normalize_spotify_id(raw, "playlist") if raw else None
    if norm and _playlist_id_known_to_session(runner, norm):
        return norm
    preferred = getattr(runner, "_last_session_playlist_id", None)
    if isinstance(preferred, str) and preferred.strip():
        preferred = preferred.strip()
    else:
        preferred = None
    if not preferred:
        mut = getattr(runner, "_last_library_mutation", None)
        if isinstance(mut, dict) and mut.get("segment") == "playlist":
            ids = mut.get("ids")
            if isinstance(ids, list) and ids:
                last = str(ids[-1]).strip()
                if _looks_like_spotify_catalog_id(last):
                    preferred = last
    if norm and not _playlist_id_known_to_session(runner, norm) and preferred:
        return preferred
    hint = extract_requested_playlist_name(user_text) or ""
    if not hint:
        m = _PLAYLIST_NAME_IN_TEXT_RE.search(user_text or "")
        if m:
            hint = m.group(1).strip()
    if hint:
        from_lib = _match_playlist_from_user_library(runner, hint)
        if from_lib:
            if hasattr(runner, "note_session_playlist_id"):
                runner.note_session_playlist_id(from_lib)
            return from_lib
    if preferred and (not norm or not _playlist_id_known_to_session(runner, norm)):
        return preferred
    return norm
