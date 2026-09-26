"""Carry Spotify catalog context across chat turns (playlist pronouns, session seeding)."""

from __future__ import annotations

import re

from spot_backend.spotify_tools import SpotifyToolRunner, _looks_like_spotify_catalog_id

_PLAYLIST_ID_IN_TEXT = re.compile(
    r"(?:spotify:playlist:|playlist[_\s]?id[:\s]+[`'\"]?)([A-Za-z0-9]{22})",
    re.I,
)
_BARE_PLAYLIST_ID = re.compile(r"\b([A-Za-z0-9]{22})\b")

_PLAYLIST_PRONOUNS = frozenset(
    {
        "it",
        "that",
        "this",
        "the playlist",
        "that playlist",
        "this playlist",
    }
)


def is_playlist_pronoun_reference(raw: str) -> bool:
    return (raw or "").strip().lower() in _PLAYLIST_PRONOUNS


def last_playlist_id_from_chat_history(history: list[dict[str, str]] | None) -> str | None:
    """Best-effort: last playlist id mentioned in prior assistant turns."""
    if not history:
        return None
    for turn in reversed(history):
        if turn.get("role") != "assistant":
            continue
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        for pattern in (_PLAYLIST_ID_IN_TEXT,):
            matches = pattern.findall(content)
            if matches:
                return matches[-1]
        if "playlist" in content.lower():
            for bare in _BARE_PLAYLIST_ID.findall(content):
                if _looks_like_spotify_catalog_id(bare):
                    return bare
    return None


def seed_runner_from_chat_history(
    runner: SpotifyToolRunner,
    history: list[dict[str, str]] | None,
    *,
    conversation_id: str | None = None,
) -> None:
    pid = last_playlist_id_from_chat_history(history)
    if not pid:
        from spot_backend.library_mutation_store import load_last_playlist_id

        cid = (conversation_id or runner.conversation_id or "").strip() or None
        pid = load_last_playlist_id(cid)
    if pid:
        runner.note_session_playlist_id(pid)
