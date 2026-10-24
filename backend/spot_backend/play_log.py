"""Local play history log with vector + FTS retrieval (agentic RAG)."""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from spot_backend.config import Settings, get_settings
from spot_backend.data_dir_lock import data_dir_lock
from spot_backend.library_index import format_fts_match_query
from spot_backend.ollama_embed import Embedder, HashingEmbedder, OllamaEmbedder
from spot_backend.play_log_windows import TimeWindow, resolve_play_history_window

logger = logging.getLogger(__name__)

_embed_queue: queue.Queue[int] = queue.Queue()
_embed_worker_started = False
_prune_guard = threading.Lock()
_last_prune_day: str | None = None


@dataclass
class PlayHistoryHit:
    track_uri: str
    title: str
    artist: str
    album: str
    played_at: str
    request_snippet: str
    score: float
    mode: str


def play_log_path(settings: Settings) -> Path:
    return settings.data_dir / "play_log.sqlite"


def play_log_enabled(settings: Settings | None = None) -> bool:
    s = settings or get_settings()
    return bool(s.play_log_enabled)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS play_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_uri TEXT NOT NULL,
            title TEXT NOT NULL,
            artist TEXT NOT NULL,
            album TEXT NOT NULL,
            played_at TEXT NOT NULL,
            request_text TEXT,
            conversation_id TEXT,
            skipped INTEGER NOT NULL DEFAULT 0,
            listened_ms INTEGER NOT NULL DEFAULT 0,
            vec BLOB,
            index_text TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            play_count INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_play_log_played_at ON play_log(played_at);
        CREATE INDEX IF NOT EXISTS idx_play_log_uri ON play_log(track_uri);
        CREATE VIRTUAL TABLE IF NOT EXISTS play_log_fts USING fts5(
            id UNINDEXED, index_text, tokenize='porter'
        );
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    conn.commit()


def play_index_text(
    title: str,
    artist: str,
    album: str,
    request_text: str | None,
    *,
    mood_tags: str = "",
) -> str:
    parts = [title or "", artist or "", album or "", (request_text or "").strip()]
    if mood_tags:
        parts.append(mood_tags)
    return " — ".join(p for p in parts if p)


class PlayLogStore:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        path: Path | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.path = path or play_log_path(self.settings)
        self.embedder: Embedder = embedder or OllamaEmbedder(self.settings)
        self._keyword_only = False

    def _ensure(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = _connect(self.path)
        _init_schema(conn)
        return conn

    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def insert_play(
        self,
        *,
        track_uri: str,
        title: str,
        artist: str,
        album: str,
        request_text: str | None,
        conversation_id: str | None,
        played_at: datetime | None = None,
        mood_tags: str = "",
        skipped: bool = False,
        listened_ms: int = 0,
        vec: list[float] | None = None,
    ) -> int:
        when = played_at or datetime.now(timezone.utc)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        played_iso = when.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        itext = play_index_text(title, artist, album, request_text, mood_tags=mood_tags)
        th = self._text_hash(itext)
        blob = None
        if vec is not None:
            blob = np.array(vec, dtype=np.float32).tobytes()
        with data_dir_lock(self.settings.data_dir):
            with self._ensure() as conn:
                conn.execute(
                    """
                    INSERT INTO play_log(
                        track_uri, title, artist, album, played_at, request_text, conversation_id,
                        skipped, listened_ms, vec, index_text, text_hash, play_count
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1)
                    """,
                    (
                        track_uri,
                        title,
                        artist,
                        album,
                        played_iso,
                        request_text,
                        conversation_id,
                        1 if skipped else 0,
                        int(listened_ms),
                        blob,
                        itext,
                        th,
                    ),
                )
                row_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
                conn.execute(
                    "INSERT INTO play_log_fts(id, index_text) VALUES(?,?)",
                    (row_id, itext),
                )
                conn.commit()
        if vec is None:
            _enqueue_embed(row_id, self.settings)
        return row_id

    def mark_skipped(self, row_id: int, *, listened_ms: int | None = None) -> None:
        with data_dir_lock(self.settings.data_dir):
            with self._ensure() as conn:
                if listened_ms is not None:
                    conn.execute(
                        "UPDATE play_log SET skipped=1, listened_ms=? WHERE id=?",
                        (int(listened_ms), row_id),
                    )
                else:
                    conn.execute("UPDATE play_log SET skipped=1 WHERE id=?", (row_id,))
                conn.commit()

    def mark_latest_open_skipped(self, track_uri: str, *, listened_ms: int = 0) -> None:
        with self._ensure() as conn:
            row = conn.execute(
                """
                SELECT id FROM play_log
                WHERE track_uri=? AND skipped=0
                ORDER BY played_at DESC LIMIT 1
                """,
                (track_uri,),
            ).fetchone()
            if row:
                self.mark_skipped(int(row["id"]), listened_ms=listened_ms)

    def clear_all(self) -> int:
        with data_dir_lock(self.settings.data_dir):
            with self._ensure() as conn:
                n = conn.execute("SELECT COUNT(*) AS c FROM play_log").fetchone()["c"]
                conn.execute("DELETE FROM play_log")
                conn.execute("DELETE FROM play_log_fts")
                conn.commit()
                conn.execute("VACUUM")
                conn.commit()
        return int(n)

    def prune(self, *, now: datetime | None = None) -> dict[str, int]:
        now = now or datetime.now(timezone.utc)
        cutoff_90 = (now - timedelta(days=90)).isoformat().replace("+00:00", "Z")
        cutoff_skip = (now - timedelta(days=7)).isoformat().replace("+00:00", "Z")
        with data_dir_lock(self.settings.data_dir):
            with self._ensure() as conn:
                old = conn.execute(
                    "SELECT id FROM play_log WHERE played_at < ?", (cutoff_90,)
                ).fetchall()
                skip_old = conn.execute(
                    "SELECT id FROM play_log WHERE skipped=1 AND played_at < ?", (cutoff_skip,)
                ).fetchall()
                ids = {int(r["id"]) for r in old} | {int(r["id"]) for r in skip_old}
                for rid in ids:
                    conn.execute("DELETE FROM play_log WHERE id=?", (rid,))
                    conn.execute("DELETE FROM play_log_fts WHERE id=?", (rid,))
                conn.commit()
                if ids:
                    conn.execute("VACUUM")
                    conn.commit()
        return {"deleted": len(ids), "older_than_90d": len(old), "skipped_older_than_7d": len(skip_old)}

    def _cosine_search(
        self,
        conn: sqlite3.Connection,
        query_vec: np.ndarray,
        *,
        start_utc: datetime,
        end_utc: datetime,
        include_skipped: bool,
        limit: int,
    ) -> list[PlayHistoryHit]:
        start_s = start_utc.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        end_s = end_utc.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        skip_clause = "" if include_skipped else "AND skipped=0"
        rows = conn.execute(
            f"""
            SELECT id, track_uri, title, artist, album, played_at, request_text, vec, listened_ms, play_count
            FROM play_log
            WHERE played_at >= ? AND played_at <= ? {skip_clause} AND vec IS NOT NULL
            """,
            (start_s, end_s),
        ).fetchall()
        if not rows:
            return []
        mat: list[np.ndarray] = []
        meta: list[sqlite3.Row] = []
        for row in rows:
            blob = row["vec"]
            if not blob:
                continue
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.size == 0:
                continue
            mat.append(vec)
            meta.append(row)
        if not mat:
            return []
        matrix = np.vstack(mat)
        q = query_vec.astype(np.float32)
        qn = np.linalg.norm(q) or 1.0
        mn = np.linalg.norm(matrix, axis=1)
        mn = np.where(mn == 0, 1.0, mn)
        scores = (matrix @ q) / (mn * qn)
        order = np.argsort(-scores)
        hits: list[PlayHistoryHit] = []
        seen_uri: set[str] = set()
        for idx in order:
            row = meta[int(idx)]
            uri = str(row["track_uri"])
            if uri in seen_uri:
                continue
            seen_uri.add(uri)
            base = float(scores[int(idx)])
            boost = min(0.15, 0.03 * int(row["play_count"] or 1))
            if int(row["listened_ms"] or 0) > 120_000:
                boost += 0.05
            score = base + boost
            req = (row["request_text"] or "")[:120]
            hits.append(
                PlayHistoryHit(
                    track_uri=uri,
                    title=str(row["title"]),
                    artist=str(row["artist"]),
                    album=str(row["album"]),
                    played_at=str(row["played_at"]),
                    request_snippet=req,
                    score=score,
                    mode="semantic",
                )
            )
            if len(hits) >= limit:
                break
        return hits

    def _fts_search(
        self,
        conn: sqlite3.Connection,
        query: str,
        *,
        start_utc: datetime,
        end_utc: datetime,
        include_skipped: bool,
        limit: int,
    ) -> list[PlayHistoryHit]:
        match = format_fts_match_query(query)
        start_s = start_utc.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        end_s = end_utc.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        skip_clause = "" if include_skipped else "AND p.skipped=0"
        rows = conn.execute(
            f"""
            SELECT p.id, p.track_uri, p.title, p.artist, p.album, p.played_at, p.request_text,
                   bm25(play_log_fts) AS rank
            FROM play_log_fts
            JOIN play_log p ON p.id = play_log_fts.id
            WHERE play_log_fts MATCH ? AND p.played_at >= ? AND p.played_at <= ? {skip_clause}
            ORDER BY rank
            LIMIT ?
            """,
            (match, start_s, end_s, max(limit * 3, limit)),
        ).fetchall()
        hits: list[PlayHistoryHit] = []
        seen: set[str] = set()
        for row in rows:
            uri = str(row["track_uri"])
            if uri in seen:
                continue
            seen.add(uri)
            hits.append(
                PlayHistoryHit(
                    track_uri=uri,
                    title=str(row["title"]),
                    artist=str(row["artist"]),
                    album=str(row["album"]),
                    played_at=str(row["played_at"]),
                    request_snippet=(row["request_text"] or "")[:120],
                    score=float(-row["rank"]),
                    mode="keyword",
                )
            )
            if len(hits) >= limit:
                break
        return hits

    def search(
        self,
        query: str,
        *,
        window: str = "this_week",
        limit: int = 20,
        include_skipped: bool = False,
        tz_name: str | None = None,
        explicit_start: str | None = None,
        explicit_end: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        limit = max(1, min(50, int(limit)))
        tw = resolve_play_history_window(
            window,
            tz_name=tz_name or self.settings.user_timezone,
            explicit_start=explicit_start,
            explicit_end=explicit_end,
            now=now,
        )
        with self._ensure() as conn:
            hits: list[PlayHistoryHit] = []
            try:
                qvec = np.array(self.embedder.embed_query(query), dtype=np.float32)
                hits = self._cosine_search(
                    conn,
                    qvec,
                    start_utc=tw.start_utc,
                    end_utc=tw.end_utc,
                    include_skipped=include_skipped,
                    limit=limit,
                )
            except Exception:
                hits = []
            if not hits:
                hits = self._fts_search(
                    conn,
                    query,
                    start_utc=tw.start_utc,
                    end_utc=tw.end_utc,
                    include_skipped=include_skipped,
                    limit=limit,
                )
        return {
            "available": True,
            "window": tw.label,
            "start_utc": tw.start_utc.isoformat().replace("+00:00", "Z"),
            "end_utc": tw.end_utc.isoformat().replace("+00:00", "Z"),
            "query": query,
            "results": [_hit_dict(h) for h in hits],
            "count": len(hits),
        }

    def embed_pending_rows(self, max_rows: int = 32) -> int:
        with self._ensure() as conn:
            rows = conn.execute(
                "SELECT id, index_text FROM play_log WHERE vec IS NULL LIMIT ?", (max_rows,)
            ).fetchall()
        if not rows:
            return 0
        texts = [str(r["index_text"]) for r in rows]
        try:
            vectors = self.embedder.embed_documents(texts)
            self._keyword_only = False
        except Exception:
            return 0
        with data_dir_lock(self.settings.data_dir):
            with self._ensure() as conn:
                for row, vec in zip(rows, vectors):
                    blob = np.array(vec, dtype=np.float32).tobytes()
                    conn.execute("UPDATE play_log SET vec=? WHERE id=?", (blob, int(row["id"])))
                conn.commit()
        return len(rows)


def _hit_dict(hit: PlayHistoryHit) -> dict[str, Any]:
    return {
        "track_uri": hit.track_uri,
        "title": hit.title,
        "artist": hit.artist,
        "album": hit.album,
        "played_at": hit.played_at,
        "request_snippet": hit.request_snippet,
        "score": round(hit.score, 4),
        "mode": hit.mode,
    }


def _enqueue_embed(row_id: int, settings: Settings) -> None:
    global _embed_worker_started
    _embed_queue.put(row_id)
    if not _embed_worker_started:
        _embed_worker_started = True

        def _worker() -> None:
            store = PlayLogStore(settings=settings)
            while True:
                try:
                    _embed_queue.get(timeout=2.0)
                except queue.Empty:
                    store.embed_pending_rows()
                    continue
                store.embed_pending_rows()

        threading.Thread(target=_worker, name="play-log-embed", daemon=True).start()


def schedule_daily_prune(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    if not play_log_enabled(settings):
        return
    global _last_prune_day
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _prune_guard:
        if _last_prune_day == today:
            return
        _last_prune_day = today
    try:
        PlayLogStore(settings=settings).prune()
    except Exception:
        logger.exception("play log prune failed")


def search_play_history_tool(
    query: str,
    window: str = "this_week",
    limit: int = 20,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    if not play_log_enabled(settings):
        return {"available": False, "reason": "disabled"}
    schedule_daily_prune(settings)
    return PlayLogStore(settings=settings).search(query, window=window, limit=limit)


def clear_play_history_tool(*, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    if not play_log_enabled(settings):
        return {"available": False, "reason": "disabled"}
    deleted = PlayLogStore(settings=settings).clear_all()
    return {"ok": True, "deleted": deleted}
