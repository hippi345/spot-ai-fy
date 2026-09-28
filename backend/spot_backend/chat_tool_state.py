"""Carry Spotify catalog context across chat turns (playlist pronouns, session seeding)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from spot_backend.spotify_tools import SpotifyToolRunner, _looks_like_spotify_catalog_id

if TYPE_CHECKING:
    pass

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

_LIBRARY_PRONOUNS = frozenset(
    {
        "it",
        "that",
        "this",
        "this show",
        "that show",
        "this episode",
        "that episode",
        "this track",
        "this song",
        "this album",
        "this playlist",
        "that playlist",
    }
)


def is_playlist_pronoun_reference(raw: str) -> bool:
    return (raw or "").strip().lower() in _PLAYLIST_PRONOUNS


def is_library_pronoun_reference(raw: str) -> bool:
    return (raw or "").strip().lower() in _LIBRARY_PRONOUNS


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


def format_runner_session_context(runner: SpotifyToolRunner) -> str:
    """Short system suffix: ids the model should use for it/that/rename/reorder/remove."""
    lines: list[str] = []
    pid = getattr(runner, "_last_session_playlist_id", None)
    if isinstance(pid, str) and pid.strip():
        lines.append(f"Session playlist id (it/that playlist for edits): {pid.strip()}")
    mut = getattr(runner, "_last_library_mutation", None)
    if isinstance(mut, dict):
        seg = mut.get("segment")
        ids = mut.get("ids")
        if isinstance(seg, str) and isinstance(ids, list) and ids:
            clean = [str(i) for i in ids if str(i).strip()]
            if clean:
                lines.append(
                    f"Last library action: {seg} id(s) {clean} — 'it/this/that' remove or "
                    f"save/check should use these ids (spotify:{seg}:… URIs)."
                )
    if not lines:
        return ""
    return "\n\nConversation session context:\n- " + "\n- ".join(lines) + "\n"


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
