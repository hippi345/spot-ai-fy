"""Local semantic index of the user's Spotify library."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from spot_backend.config import Settings, get_settings
from spot_backend.data_dir_lock import data_dir_lock
from spot_backend.ollama_embed import Embedder, HashingEmbedder, OllamaEmbedder
from spot_backend.spotify_client import SpotifyClient, SpotifyRateLimitError

logger = logging.getLogger(__name__)

Kind = Literal["track", "playlist", "artist", "any"]

_refresh_guard = threading.Lock()
_last_refresh_at: float = 0.0
_indexing = False


@dataclass
class LibraryHit:
    kind: str
    name: str
    artists: str
    uri: str
    score: float
    why: str
    mode: str


def index_path(settings: Settings) -> Path:
    return settings.data_dir / "library_index.sqlite"


def library_index_enabled(settings: Settings | None = None) -> bool:
    s = settings or get_settings()
    return bool(s.library_index_enabled)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            uri TEXT NOT NULL,
            text TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            vec BLOB,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
            id UNINDEXED, kind, uri, text, tokenize='porter'
        );
        """
    )
    conn.commit()


def _meta_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row else None


def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


class LibraryIndex:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: SpotifyClient | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or SpotifyClient(settings=self.settings)
        self.path = index_path(self.settings)
        self.embedder: Embedder = embedder or OllamaEmbedder(self.settings)
        self._keyword_only = False

    def status(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"ready": False, "reason": "indexing"}
        with _connect(self.path) as conn:
            n = conn.execute("SELECT COUNT(*) AS c FROM items").fetchone()["c"]
            return {
                "ready": n > 0,
                "items": int(n),
                "mode": "keyword" if self._keyword_only else "semantic",
                "embed_model": _meta_get(conn, "embed_model"),
            }

    def delete_index_file(self) -> None:
        if self.path.is_file():
            self.path.unlink()

    def _ensure_db(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = _connect(self.path)
        _init_schema(conn)
        return conn

    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cosine_search(self, conn: sqlite3.Connection, query_vec: np.ndarray, kind: Kind, limit: int) -> list[LibraryHit]:
        rows = conn.execute("SELECT id, kind, uri, text, vec FROM items WHERE vec IS NOT NULL").fetchall()
        if not rows:
            return []
        ids: list[str] = []
        kinds: list[str] = []
        uris: list[str] = []
        texts: list[str] = []
        mat: list[np.ndarray] = []
        for row in rows:
            if kind != "any" and row["kind"] != kind:
                continue
            blob = row["vec"]
            if not blob:
                continue
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.size == 0:
                continue
            ids.append(row["id"])
            kinds.append(row["kind"])
            uris.append(row["uri"])
            texts.append(row["text"])
            mat.append(vec)
        if not mat:
            return []
        matrix = np.vstack(mat)
        q = query_vec.astype(np.float32)
        qn = np.linalg.norm(q) or 1.0
        mn = np.linalg.norm(matrix, axis=1)
        mn = np.where(mn == 0, 1.0, mn)
        scores = (matrix @ q) / (mn * qn)
        order = np.argsort(-scores)[:limit]
        hits: list[LibraryHit] = []
        for idx in order:
            i = int(idx)
            snippet = texts[i][:160]
            hits.append(
                LibraryHit(
                    kind=kinds[i],
                    name=_display_name(texts[i]),
                    artists=_artist_hint(texts[i]),
                    uri=uris[i],
                    score=float(scores[i]),
                    why=snippet,
                    mode="semantic",
                )
            )
        return hits

    def _fts_search(self, conn: sqlite3.Connection, query: str, kind: Kind, limit: int) -> list[LibraryHit]:
        sql = (
            "SELECT i.kind, i.uri, i.text, bm25(items_fts) AS rank "
            "FROM items_fts JOIN items i ON items_fts.id = i.id "
            "WHERE items_fts MATCH ? "
        )
        params: list[Any] = [query.replace('"', " ")]
        if kind != "any":
            sql += "AND i.kind = ? "
            params.append(kind)
        sql += "ORDER BY rank LIMIT ?"
        params.append(limit)
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            rows = []
        hits: list[LibraryHit] = []
        for row in rows:
            text = row["text"]
            hits.append(
                LibraryHit(
                    kind=row["kind"],
                    name=_display_name(text),
                    artists=_artist_hint(text),
                    uri=row["uri"],
                    score=float(-row["rank"]),
                    why=text[:160],
                    mode="keyword",
                )
            )
        return hits

    def search(self, query: str, kind: Kind = "any", limit: int = 10) -> dict[str, Any]:
        limit = max(1, min(25, int(limit)))
        if not self.path.is_file():
            return {"available": False, "reason": "indexing"}
        with _connect(self.path) as conn:
            count = conn.execute("SELECT COUNT(*) AS c FROM items").fetchone()["c"]
            if count == 0:
                return {"available": False, "reason": "empty"}
            use_keyword = self._keyword_only
            if not use_keyword:
                try:
                    qv = np.array(self.embedder.embed_query(query), dtype=np.float32)
                    hits = self._cosine_search(conn, qv, kind, limit)
                    if hits:
                        return {"available": True, "mode": "semantic", "results": [_hit_dict(h) for h in hits]}
                except Exception:
                    use_keyword = True
            hits = self._fts_search(conn, query, kind, limit)
            if not hits:
                return {"available": True, "mode": "keyword", "results": []}
            return {"available": True, "mode": "keyword", "results": [_hit_dict(h) for h in hits]}

    def sync_full(self, *, embedder: Embedder | None = None) -> None:
        global _indexing
        with _refresh_guard:
            _indexing = True
        try:
            if embedder is not None:
                self.embedder = embedder
            self._sync_impl(full=True)
        finally:
            with _refresh_guard:
                _indexing = False

    def sync_incremental(self) -> None:
        global _last_refresh_at
        now = time.time()
        hours = float(self.settings.library_index_refresh_hours)
        if now - _last_refresh_at < hours * 3600:
            return
        with _refresh_guard:
            _last_refresh_at = now
        self._sync_impl(full=False)

    def _sync_impl(self, *, full: bool) -> None:
        lock = data_dir_lock(self.settings.data_dir)
        with lock:
            conn = self._ensure_db()
            try:
                stored_model = _meta_get(conn, "embed_model")
                if stored_model and stored_model != self.embedder.model_name:
                    conn.execute("DELETE FROM items")
                    conn.execute("DELETE FROM items_fts")
                    conn.commit()
                docs = self._collect_documents()
                self._upsert_documents(conn, docs)
                _meta_set(conn, "embed_model", self.embedder.model_name)
                _meta_set(conn, "embed_dim", str(self.embedder.dimensions))
                _meta_set(conn, "last_sync", str(time.time()))
                conn.commit()
            finally:
                conn.close()

    def _collect_documents(self) -> list[dict[str, str]]:
        docs: list[dict[str, str]] = []
        docs.extend(self._liked_tracks_docs())
        docs.extend(self._playlist_docs())
        docs.extend(self._artist_docs())
        return docs

    def _liked_tracks_docs(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        offset = 0
        while True:
            try:
                data = self.client.api_get("/me/tracks", params={"limit": 50, "offset": offset})
            except SpotifyRateLimitError:
                break
            items = data.get("items") if isinstance(data, dict) else None
            if not isinstance(items, list) or not items:
                break
            for row in items:
                track = row.get("track") if isinstance(row, dict) else None
                if not isinstance(track, dict):
                    continue
                tid = track.get("id")
                uri = track.get("uri") or (f"spotify:track:{tid}" if tid else "")
                if not uri:
                    continue
                artists = ", ".join(
                    a.get("name", "") for a in (track.get("artists") or []) if isinstance(a, dict)
                )
                album = track.get("album") if isinstance(track.get("album"), dict) else {}
                year = (album.get("release_date") or "")[:4]
                text = f"{track.get('name', '')} — {artists} · album {album.get('name', '')} · {year}"
                out.append({"id": f"track:{tid}", "kind": "track", "uri": uri, "text": text})
            if len(items) < 50:
                break
            offset += 50
        return out

    def _playlist_docs(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        offset = 0
        while True:
            try:
                data = self.client.api_get("/me/playlists", params={"limit": 50, "offset": offset})
            except SpotifyRateLimitError:
                break
            items = data.get("items") if isinstance(data, dict) else None
            if not isinstance(items, list) or not items:
                break
            for pl in items:
                if not isinstance(pl, dict):
                    continue
                pid = pl.get("id")
                uri = pl.get("uri") or f"spotify:playlist:{pid}"
                desc = (pl.get("description") or "").strip()
                artists = self._top_playlist_artists(pid)
                text = f"{pl.get('name', '')}: {desc} · top artists: {artists}"
                out.append({"id": f"playlist:{pid}", "kind": "playlist", "uri": uri, "text": text})
            if len(items) < 50:
                break
            offset += 50
        return out

    def _top_playlist_artists(self, playlist_id: str | None) -> str:
        if not playlist_id:
            return ""
        try:
            data = self.client.api_get(
                f"/playlists/{playlist_id}/items", params={"limit": 50, "fields": "items(track(artists(name)))"}
            )
        except Exception:
            return ""
        names: list[str] = []
        for row in data.get("items") or []:
            track = row.get("track") if isinstance(row, dict) else None
            if not isinstance(track, dict):
                continue
            for a in track.get("artists") or []:
                if isinstance(a, dict) and a.get("name"):
                    names.append(str(a["name"]))
            if len(names) >= 10:
                break
        return ", ".join(names[:10])

    def _artist_docs(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        after: str | None = None
        while True:
            params: dict[str, Any] = {"limit": 50}
            if after:
                params["after"] = after
            try:
                data = self.client.api_get("/me/following", params={**params, "type": "artist"})
            except SpotifyRateLimitError:
                break
            items = data.get("artists", {}).get("items") if isinstance(data, dict) else None
            if not isinstance(items, list) or not items:
                break
            ids = [a.get("id") for a in items if isinstance(a, dict) and a.get("id")]
            genres_map = self._artist_genres_batch([i for i in ids if i])
            for artist in items:
                if not isinstance(artist, dict):
                    continue
                aid = artist.get("id")
                uri = artist.get("uri") or f"spotify:artist:{aid}"
                genres = genres_map.get(aid or "", [])
                text = f"{artist.get('name', '')} · genres: {', '.join(genres)}"
                out.append({"id": f"artist:{aid}", "kind": "artist", "uri": uri, "text": text})
            cursors = data.get("artists", {}).get("cursors") if isinstance(data, dict) else {}
            after = cursors.get("after") if isinstance(cursors, dict) else None
            if not after:
                break
        return out

    def _artist_genres_batch(self, ids: list[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for i in range(0, len(ids), 50):
            chunk = ids[i : i + 50]
            try:
                data = self.client.api_get("/artists", params={"ids": ",".join(chunk)})
            except Exception:
                continue
            for artist in data.get("artists") or []:
                if isinstance(artist, dict) and artist.get("id"):
                    out[str(artist["id"])] = list(artist.get("genres") or [])
        return out

    def _upsert_documents(self, conn: sqlite3.Connection, docs: list[dict[str, str]]) -> None:
        existing = {row["id"]: row for row in conn.execute("SELECT id, text_hash, uri FROM items").fetchall()}
        seen: set[str] = set()
        to_embed: list[tuple[str, dict[str, str]]] = []
        for doc in docs:
            did = doc["id"]
            seen.add(did)
            th = self._text_hash(doc["text"])
            prev = existing.get(did)
            if prev and prev["text_hash"] == th:
                continue
            to_embed.append((did, doc))
        if to_embed:
            try:
                vectors = self.embedder.embed_documents([d[1]["text"] for d in to_embed])
                self._keyword_only = False
            except Exception:
                vectors = [None] * len(to_embed)
                self._keyword_only = True
        else:
            vectors = []
        for (did, doc), vec in zip(to_embed, vectors):
            blob = None
            if vec is not None:
                blob = np.array(vec, dtype=np.float32).tobytes()
            conn.execute(
                "INSERT INTO items(id, kind, uri, text, text_hash, vec, updated_at) "
                "VALUES(?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, uri=excluded.uri, text=excluded.text, "
                "text_hash=excluded.text_hash, vec=excluded.vec, updated_at=excluded.updated_at",
                (did, doc["kind"], doc["uri"], doc["text"], self._text_hash(doc["text"]), blob, time.time()),
            )
            conn.execute("DELETE FROM items_fts WHERE id = ?", (did,))
            conn.execute("INSERT INTO items_fts(id, kind, uri, text) VALUES(?,?,?,?)", (did, doc["kind"], doc["uri"], doc["text"]))
        stale = set(existing.keys()) - seen
        for sid in stale:
            conn.execute("DELETE FROM items WHERE id = ?", (sid,))
            conn.execute("DELETE FROM items_fts WHERE id = ?", (sid,))


def _display_name(text: str) -> str:
    return (text.split("—")[0].split(":")[0].strip() or text)[:80]


def _artist_hint(text: str) -> str:
    if "—" in text:
        return text.split("—", 1)[1].split("·")[0].strip()[:80]
    if "genres:" in text:
        return text.split("genres:", 1)[0].strip()[:80]
    return ""


def _hit_dict(hit: LibraryHit) -> dict[str, Any]:
    return {
        "kind": hit.kind,
        "name": hit.name,
        "artists": hit.artists,
        "uri": hit.uri,
        "score": round(hit.score, 4),
        "why": hit.why,
        "mode": hit.mode,
    }


def search_my_library_tool(
    query: str,
    kind: str = "any",
    limit: int = 10,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    if not library_index_enabled(settings):
        return {"available": False, "reason": "disabled"}
    idx = LibraryIndex(settings=settings)
    idx.sync_incremental()
    k: Kind = kind if kind in ("track", "playlist", "artist", "any") else "any"
    payload = idx.search(query, kind=k, limit=limit)
    raw = json.dumps(payload, ensure_ascii=False)
    if len(raw) > 1500:
        payload["results"] = (payload.get("results") or [])[: max(1, limit // 2)]
    return payload


def start_background_build(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    if not library_index_enabled(settings):
        return

    def _worker() -> None:
        try:
            idx = LibraryIndex(settings=settings)
            idx.sync_full()
        except Exception:
            logger.exception("library index background build failed")

    threading.Thread(target=_worker, name="library-index-build", daemon=True).start()
