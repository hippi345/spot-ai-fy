"""PR #9 Spotify tools: podcasts, library contains, playlist builder preview."""

from __future__ import annotations

import json
from typing import Any

import httpx

from spot_backend.playlist_builder_store import (
    clear_playlist_preview,
    load_playlist_preview,
    save_playlist_preview,
)
from spot_backend.playlist_pick import playlist_id_is_spotify_curated
from spot_backend.spotify_dev_limits import (
    SPOTIFY_DEV_MAX_PAGE,
    SPOTIFY_SEARCH_DEFAULT_LIMIT,
    clamp_spotify_page_limit,
)

# Imported from spotify_tools at runtime via mixin — duplicate small helpers inline.
_LIBRARY_URI_CHUNK = 40
_TEST_PLAYLIST_NAME = "spot-ai-fy test"


def _compact_pr9(data: Any, limit: int = 6000) -> str:
    s = json.dumps(data, ensure_ascii=False)
    if len(s) > limit:
        return s[:limit] + "\n... (truncated)"
    return s


class SpotifyToolRunnerPr9Mixin:
    """Extra tool handlers mixed into SpotifyToolRunner."""

    def _normalize_library_uris(self, arguments: dict[str, Any]) -> list[str]:
        from spot_backend.spotify_tools import _normalize_spotify_id

        raw = arguments.get("uris")
        if raw is None:
            raw = arguments.get("uri")
        out: list[str] = []
        if isinstance(raw, str) and raw.strip():
            raw = [raw]
        if not isinstance(raw, list):
            return out
        for item in raw:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s:
                continue
            if s.lower().startswith("spotify:"):
                out.append(s)
                continue
            for seg in ("track", "album", "episode", "show", "audiobook", "playlist", "artist", "user"):
                bare = _normalize_spotify_id(s, seg)
                if bare:
                    out.append(f"spotify:{seg}:{bare}")
                    break
        seen: set[str] = set()
        uniq: list[str] = []
        for u in out:
            if u not in seen:
                seen.add(u)
                uniq.append(u)
        return uniq[: _LIBRARY_URI_CHUNK]

    def _library_contains(self, arguments: dict[str, Any]) -> str:
        uris = self._normalize_library_uris(arguments)
        if not uris:
            return json.dumps({"error": "uris is required (max 40 spotify: URIs)"})
        data = self.client.api_get(
            "/me/library/contains",
            params={"uris": ",".join(uris)},
        )
        if isinstance(data, list):
            return _compact_pr9({"uris": uris, "saved": data})
        return _compact_pr9(data)

    def _library_save_uris(self, arguments: dict[str, Any]) -> str:
        uris = self._normalize_library_uris(arguments)
        if not uris:
            return json.dumps({"error": "uris is required"})
        for offset in range(0, len(uris), _LIBRARY_URI_CHUNK):
            chunk = uris[offset : offset + _LIBRARY_URI_CHUNK]
            self.client.api_put("/me/library", params={"uris": ",".join(chunk)})
        return json.dumps({"ok": True, "saved_uris": uris})

    def _library_remove_uris(self, arguments: dict[str, Any]) -> str:
        uris = self._normalize_library_uris(arguments)
        if not uris:
            return json.dumps({"error": "uris is required"})
        for offset in range(0, len(uris), _LIBRARY_URI_CHUNK):
            chunk = uris[offset : offset + _LIBRARY_URI_CHUNK]
            self.client.api_delete("/me/library", params={"uris": ",".join(chunk)})
        return json.dumps({"ok": True, "removed_uris": uris})

    def _get_show(self, arguments: dict[str, Any]) -> str:
        from spot_backend.spotify_tools import _normalize_spotify_id, _pick_arg

        sid = _normalize_spotify_id(_pick_arg(arguments, "show_id", "id"), "show")
        if not sid:
            return json.dumps({"error": "show_id is required"})
        data = self.client.api_get_cached(f"/shows/{sid}")
        return _compact_pr9(data)

    def _get_show_episodes(self, arguments: dict[str, Any]) -> str:
        from spot_backend.spotify_tools import _normalize_spotify_id, _pick_arg, _safe_int

        sid = _normalize_spotify_id(_pick_arg(arguments, "show_id", "id"), "show")
        if not sid:
            return json.dumps({"error": "show_id is required"})
        limit = clamp_spotify_page_limit(arguments.get("limit"), default=SPOTIFY_DEV_MAX_PAGE)
        offset = _safe_int(arguments.get("offset"), 0, lo=0, hi=900_000)
        data = self.client.api_get(
            f"/shows/{sid}/episodes",
            params={"limit": limit, "offset": offset},
        )
        return _compact_pr9(data)

    def _get_episode(self, arguments: dict[str, Any]) -> str:
        from spot_backend.spotify_tools import _normalize_spotify_id, _pick_arg

        eid = _normalize_spotify_id(_pick_arg(arguments, "episode_id", "id"), "episode")
        if not eid:
            return json.dumps({"error": "episode_id is required"})
        data = self.client.api_get_cached(f"/episodes/{eid}")
        return _compact_pr9(data)

    def _get_audiobook(self, arguments: dict[str, Any]) -> str:
        from spot_backend.spotify_tools import _normalize_spotify_id, _pick_arg

        aid = _normalize_spotify_id(_pick_arg(arguments, "audiobook_id", "id"), "audiobook")
        if not aid:
            return json.dumps({"error": "audiobook_id is required"})
        data = self.client.api_get_cached(f"/audiobooks/{aid}")
        return _compact_pr9(data)

    def _get_audiobook_chapters(self, arguments: dict[str, Any]) -> str:
        from spot_backend.spotify_tools import _normalize_spotify_id, _pick_arg, _safe_int

        aid = _normalize_spotify_id(_pick_arg(arguments, "audiobook_id", "id"), "audiobook")
        if not aid:
            return json.dumps({"error": "audiobook_id is required"})
        limit = clamp_spotify_page_limit(arguments.get("limit"), default=SPOTIFY_DEV_MAX_PAGE)
        offset = _safe_int(arguments.get("offset"), 0, lo=0, hi=900_000)
        data = self.client.api_get(
            f"/audiobooks/{aid}/chapters",
            params={"limit": limit, "offset": offset},
        )
        return _compact_pr9(data)

    def _get_chapter(self, arguments: dict[str, Any]) -> str:
        from spot_backend.spotify_tools import _normalize_spotify_id, _pick_arg

        cid = _normalize_spotify_id(_pick_arg(arguments, "chapter_id", "id"), "chapter")
        if not cid:
            return json.dumps({"error": "chapter_id is required"})
        data = self.client.api_get_cached(f"/chapters/{cid}")
        return _compact_pr9(data)

    def _me_shows(self, arguments: dict[str, Any]) -> str:
        from spot_backend.spotify_tools import _safe_int

        limit = clamp_spotify_page_limit(arguments.get("limit"), default=SPOTIFY_DEV_MAX_PAGE)
        offset = _safe_int(arguments.get("offset"), 0, lo=0, hi=900_000)
        data = self.client.api_get("/me/shows", params={"limit": limit, "offset": offset})
        return _compact_pr9(data)

    def _me_episodes(self, arguments: dict[str, Any]) -> str:
        from spot_backend.spotify_tools import _safe_int

        limit = clamp_spotify_page_limit(arguments.get("limit"), default=SPOTIFY_DEV_MAX_PAGE)
        offset = _safe_int(arguments.get("offset"), 0, lo=0, hi=900_000)
        data = self.client.api_get("/me/episodes", params={"limit": limit, "offset": offset})
        return _compact_pr9(data)

    def _me_audiobooks(self, arguments: dict[str, Any]) -> str:
        from spot_backend.spotify_tools import _safe_int

        limit = clamp_spotify_page_limit(arguments.get("limit"), default=SPOTIFY_DEV_MAX_PAGE)
        offset = _safe_int(arguments.get("offset"), 0, lo=0, hi=900_000)
        data = self.client.api_get("/me/audiobooks", params={"limit": limit, "offset": offset})
        return _compact_pr9(data)

    def _resolve_track_query(self, query: str, market: str) -> dict[str, Any] | None:
        from spot_backend.spotify_tools import _normalize_market, _pick_arg

        q = query.strip()
        if not q:
            return None
        data = self.client.api_get(
            "/search",
            params={
                "q": q,
                "type": "track",
                "limit": SPOTIFY_SEARCH_DEFAULT_LIMIT,
                "market": _normalize_market(market),
            },
        )
        if not isinstance(data, dict):
            return None
        tracks = data.get("tracks")
        items = tracks.get("items") if isinstance(tracks, dict) else None
        if not isinstance(items, list):
            return None
        for row in items:
            if isinstance(row, dict) and row.get("uri"):
                return row
        return None

    def _playlist_builder_preview(self, arguments: dict[str, Any]) -> str:
        from spot_backend.spotify_tools import _coerce_str, _normalize_market, _pick_arg

        name = _coerce_str(_pick_arg(arguments, "name", "playlist_name"), "New mix").strip()
        queries = arguments.get("track_queries") or arguments.get("tracks") or []
        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, list) or not queries:
            return json.dumps({"error": "track_queries is required (10-25 search strings)"})
        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        resolved: list[dict[str, Any]] = []
        seen_uris: set[str] = set()
        for raw_q in queries:
            if not isinstance(raw_q, str):
                continue
            q = raw_q.strip()
            if not q:
                continue
            if len(resolved) >= 25:
                break
            track = self._resolve_track_query(q, market)
            if not track:
                continue
            uri = track.get("uri")
            if not isinstance(uri, str) or uri in seen_uris:
                continue
            seen_uris.add(uri)
            artists = track.get("artists") if isinstance(track.get("artists"), list) else []
            artist_name = ""
            if artists and isinstance(artists[0], dict):
                artist_name = str(artists[0].get("name") or "")
            resolved.append(
                {
                    "n": len(resolved) + 1,
                    "uri": uri,
                    "name": track.get("name"),
                    "artist": artist_name,
                    "query": q,
                }
            )
        if len(resolved) < 1:
            return json.dumps(
                {
                    "ok": False,
                    "failure_reason": "playlist_preview_empty",
                    "error": "Could not resolve any tracks for that list.",
                }
            )
        preview = {
            "proposed_name": name,
            "tracks": resolved,
            "awaiting_approval": True,
        }
        save_playlist_preview(self.conversation_id, preview)
        lines = [f'Proposed playlist "{name}" ({len(resolved)} tracks):']
        for row in resolved:
            lines.append(f"{row['n']}. {row.get('name')} — {row.get('artist')}")
        lines.append("Reply yes / make it to create this private playlist, or ask to edit the list.")
        preview_text = "\n".join(lines)
        return json.dumps(
            {
                "ok": True,
                "preview": preview,
                "preview_text": preview_text,
                "awaiting_approval": True,
                "message": preview_text,
            },
            ensure_ascii=False,
        )

    def _playlist_builder_commit(self, arguments: dict[str, Any]) -> str:
        from spot_backend.spotify_tools import _coerce_str, _pick_arg

        preview = load_playlist_preview(self.conversation_id)
        if not preview:
            return json.dumps(
                {
                    "error": "No pending playlist preview in this conversation.",
                    "hint": "Call spotify_playlist_builder_preview first.",
                }
            )
        approve = arguments.get("approve", arguments.get("confirmed", True))
        if isinstance(approve, str):
            approve = approve.strip().lower() in ("yes", "y", "true", "1", "make it", "create", "ok")
        if not approve:
            clear_playlist_preview(self.conversation_id)
            return json.dumps({"ok": False, "cancelled": True})
        name = _coerce_str(arguments.get("name") or preview.get("proposed_name"), _TEST_PLAYLIST_NAME)
        if "test" in name.lower() or name == preview.get("proposed_name"):
            pass
        tracks = preview.get("tracks")
        if not isinstance(tracks, list) or not tracks:
            return json.dumps({"error": "Preview has no tracks"})
        uris = [t.get("uri") for t in tracks if isinstance(t, dict) and t.get("uri")]
        uris = [u for u in uris if isinstance(u, str)]
        if not uris:
            return json.dumps({"error": "Preview has no track URIs"})
        body = {"name": name, "public": False, "description": "Created by Spot-AI-fy"}
        created = self.client.api_post("/me/playlists", json_body=body)
        if not isinstance(created, dict) or not created.get("id"):
            return json.dumps({"ok": False, "error": "Playlist creation failed", "detail": created})
        pid = str(created["id"])
        add_error: str | None = None
        for attempt in range(2):
            try:
                self.client.api_post(
                    f"/playlists/{pid}/items",
                    json_body={"uris": uris[:100]},
                )
                add_error = None
                break
            except httpx.HTTPStatusError as e:
                add_error = f"HTTP {e.response.status_code}"
            except (OSError, RuntimeError, ValueError) as e:
                add_error = str(e)
        clear_playlist_preview(self.conversation_id)
        if add_error:
            return json.dumps(
                {
                    "ok": False,
                    "playlist_id": pid,
                    "failure_reason": "playlist_add_failed",
                    "error": (
                        "Playlist was created but adding tracks failed after a retry. "
                        "The playlist was left as-is — try spotify_add_tracks_to_playlist."
                    ),
                    "add_error": add_error[:400],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "ok": True,
                "playlist_id": pid,
                "name": name,
                "public": False,
                "track_count": len(uris),
            },
            ensure_ascii=False,
        )

    def _block_editorial_playlist_id(self, playlist_id: str) -> str | None:
        pid = (playlist_id or "").strip()
        if playlist_id_is_spotify_curated(pid):
            return json.dumps(
                {
                    "ok": False,
                    "failure_reason": "editorial_playlist_blocked",
                    "error": (
                        "That playlist is a Spotify editorial list and cannot be played from this app."
                    ),
                    "reconnect_spotify_unnecessary": True,
                },
                ensure_ascii=False,
            )
        return None
