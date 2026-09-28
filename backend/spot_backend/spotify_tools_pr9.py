"""PR #9 Spotify tools: podcasts, library contains, playlist builder preview."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

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

_LIBRARY_URI_CHUNK = 40
_TEST_PLAYLIST_NAME = "spot-ai-fy test"
_PLAYLIST_PRIVACY_USER_NOTE = (
    "Spotify still shows this playlist as public. To make it private, open it in the Spotify app, "
    "tap ⋯, and choose Make private."
)
_MIN_BUILDER_TRACKS = 10
_MAX_BUILDER_TRACKS = 25

_LIBRARY_PRONOUNS = frozenset(
    {
        "it",
        "this",
        "that",
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

_PLAYBACK_FIRST_PRONOUN_SEGMENT: dict[str, str] = {
    "this album": "album",
    "this song": "track",
    "this track": "track",
    "this show": "show",
    "that show": "show",
    "this episode": "episode",
    "that episode": "episode",
    "this podcast": "show",
    "that podcast": "show",
}

_DEICTIC_LIBRARY_PRONOUNS = frozenset(
    {
        "it",
        "that",
        "this",
        "this playlist",
        "that playlist",
    }
)

_SOUNDTRACK_MARKERS = (
    "soundtrack",
    "theme",
    "original score",
    "video game",
    "game ost",
    "nintendo",
    "zelda",
    "mario",
    "pokemon",
    "halo",
    "skyrim",
)


def _compact_pr9(data: Any, limit: int = 6000) -> str:
    s = json.dumps(data, ensure_ascii=False)
    if len(s) > limit:
        return s[:limit] + "\n... (truncated)"
    return s


def _empty_library_args_error(action: str) -> str:
    return json.dumps(
        {
            "ok": False,
            "failure_reason": "empty_args",
            "error": f"{action} requires at least one Spotify URI (or a resolved 'this/it' reference).",
            "hint": (
                "Call spotify_playback_state, spotify_search, or spotify_get_show first, "
                "then pass the id/uri from that tool output."
            ),
            "reconnect_spotify_unnecessary": True,
        },
        ensure_ascii=False,
    )


_GENERIC_QUERY_ECHO_WORDS = frozenset(
    {"chill", "vibes", "vibe", "relax", "relaxing", "slow", "jam", "jams", "mellow", "baby", "rnb", "r&b"}
)


def _theme_requests_nineties(theme_blob: str) -> bool:
    b = (theme_blob or "").lower()
    return bool(re.search(r"\b(?:90s|1990s|nineties)\b", b) or "1990" in b)


class SpotifyToolRunnerPr9Mixin:
    """Extra tool handlers mixed into SpotifyToolRunner."""

    def _library_argument_is_pronoun(self, arguments: dict[str, Any]) -> bool:
        raw = arguments.get("uris")
        if raw is None:
            raw = arguments.get("uri")
        if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], str):
            return raw[0].strip().lower() in _LIBRARY_PRONOUNS
        if isinstance(raw, str):
            return raw.strip().lower() in _LIBRARY_PRONOUNS
        return False

    def _uris_from_last_library_mutation(self, *, segment: str | None = None) -> list[str]:
        mut = getattr(self, "_last_library_mutation", None)
        if not isinstance(mut, dict):
            return []
        seg_mut = mut.get("segment")
        ids = mut.get("ids")
        if not isinstance(seg_mut, str) or not isinstance(ids, list):
            return []
        seg = seg_mut.strip().lower()
        if seg not in ("track", "album", "artist", "playlist", "show", "episode"):
            return []
        if segment and seg != segment.strip().lower():
            return []
        out: list[str] = []
        for bare in ids:
            if isinstance(bare, str) and bare.strip():
                out.append(f"spotify:{seg}:{bare.strip()}")
        return out[: _LIBRARY_URI_CHUNK]

    def _uris_from_playback_segment(self, segment: str) -> list[str]:
        if not hasattr(self, "_playback_catalog_id"):
            return []
        cid = self._playback_catalog_id(segment)
        if cid:
            return [f"spotify:{segment}:{cid}"]
        return []

    def _playback_not_verified_user_message(self, episode_title: str) -> str:
        base = (
            f"I found the latest episode, {episode_title!r}, but Spotify didn't confirm it started playing. "
            "Try tapping play on your device or ask me to transfer playback."
        )
        state = self._player_state_snapshot() if hasattr(self, "_player_state_snapshot") else None
        if not isinstance(state, dict):
            return base
        item = state.get("item")
        if not isinstance(item, dict):
            return base
        if item.get("type") not in ("track", "episode"):
            return base
        name = item.get("name") if isinstance(item.get("name"), str) else None
        artists = item.get("artists") if isinstance(item.get("artists"), list) else []
        artist = ""
        if artists and isinstance(artists[0], dict):
            artist = str(artists[0].get("name") or "").strip()
        if name and artist:
            return f"{base} You're currently listening to {name} by {artist}."
        if name:
            return f"{base} You're currently listening to {name}."
        return base

    def _uris_from_show_session_context(self) -> list[str]:
        last_show = getattr(self, "_last_show_search_id", None)
        if isinstance(last_show, str) and last_show.strip():
            return [f"spotify:show:{last_show.strip()}"]
        from_mut = self._uris_from_last_library_mutation(segment="show")
        if from_mut:
            return from_mut
        return []

    def _uris_from_session_playlist_context(self) -> list[str]:
        sess = getattr(self, "_last_session_playlist_id", None)
        if isinstance(sess, str) and sess.strip():
            return [f"spotify:playlist:{sess.strip()}"]
        return []

    def _resolve_pronoun_to_uris(self, pronoun: str) -> list[str]:
        low = (pronoun or "").strip().lower()
        seg = _PLAYBACK_FIRST_PRONOUN_SEGMENT.get(low)
        if seg:
            if seg == "show":
                from_playback = self._uris_from_playback_segment("show")
                if from_playback:
                    return from_playback
                from_search = self._uris_from_show_session_context()
                if from_search:
                    return from_search
                from_mut = self._uris_from_last_library_mutation(segment="show")
                if from_mut:
                    return from_mut
                return []
            from_playback = self._uris_from_playback_segment(seg)
            if from_playback:
                return from_playback
            from_mut = self._uris_from_last_library_mutation(segment=seg)
            if from_mut:
                return from_mut
            return []
        if low in ("this show", "that show", "this podcast", "that podcast"):
            from_playback = self._uris_from_playback_segment("show")
            if from_playback:
                return from_playback
            from_search = self._uris_from_show_session_context()
            if from_search:
                return from_search
            from_mut = self._uris_from_last_library_mutation(segment="show")
            if from_mut:
                return from_mut
            return []
        if low in ("this playlist", "that playlist") or low in _DEICTIC_LIBRARY_PRONOUNS:
            from_mut = self._uris_from_last_library_mutation()
            if from_mut:
                return from_mut
            from_sess = self._uris_from_session_playlist_context()
            if from_sess:
                return from_sess
            for try_seg in ("track", "album", "show", "episode", "playlist", "artist"):
                from_playback = self._uris_from_playback_segment(try_seg)
                if from_playback:
                    return from_playback
            return []
        if low in _LIBRARY_PRONOUNS:
            from_mut = self._uris_from_last_library_mutation()
            if from_mut:
                return from_mut
            for try_seg in ("track", "album", "show", "episode", "playlist", "artist"):
                from_playback = self._uris_from_playback_segment(try_seg)
                if from_playback:
                    return from_playback
            from_sess = self._uris_from_session_playlist_context()
            if from_sess:
                return from_sess
            return []
        return []

    def _normalize_library_uris(self, arguments: dict[str, Any]) -> list[str]:
        from spot_backend.spotify_tools import _normalize_spotify_id

        if self._library_argument_is_pronoun(arguments):
            raw = arguments.get("uris") or arguments.get("uri")
            if isinstance(raw, list) and len(raw) == 1:
                resolved = self._resolve_pronoun_to_uris(str(raw[0]))
            elif isinstance(raw, str):
                resolved = self._resolve_pronoun_to_uris(raw)
            else:
                resolved = []
            if resolved:
                return resolved
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
            low = s.lower()
            if low in _LIBRARY_PRONOUNS:
                out.extend(self._resolve_pronoun_to_uris(low))
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
            return _empty_library_args_error("spotify_library_contains")
        data = self.client.api_get(
            "/me/library/contains",
            params={"uris": ",".join(uris)},
        )
        if isinstance(data, list):
            payload: dict[str, Any] = {"ok": True, "uris": uris, "saved": data}
            if len(uris) == 1:
                payload["saved_single"] = bool(data[0]) if data else False
                label = self._library_label_for_uri(uris[0])
                if label:
                    payload["item_name"] = label
                if payload["saved_single"]:
                    name = label or "that item"
                    payload["user_message"] = f"Yes — {name} is saved in your Spotify library."
                else:
                    name = label or "That item"
                    payload["user_message"] = f"No — {name} is not in your library."
            self._record_library_mutation_from_uris(uris)
            return _compact_pr9(payload)
        return _compact_pr9(data)

    def _library_label_for_uri(self, uri: str) -> str | None:
        if not isinstance(uri, str) or not uri.lower().startswith("spotify:"):
            return None
        parts = uri.split(":")
        if len(parts) < 3:
            return None
        seg, bare = parts[1], parts[2]
        path_map = {
            "album": f"/albums/{bare}",
            "track": f"/tracks/{bare}",
            "show": f"/shows/{bare}",
            "playlist": f"/playlists/{bare}",
            "episode": f"/episodes/{bare}",
        }
        path = path_map.get(seg)
        if not path:
            return None
        try:
            meta = self.client.api_get_cached(path)
        except Exception:
            return None
        if isinstance(meta, dict) and isinstance(meta.get("name"), str):
            return meta["name"].strip()
        return None

    def _reject_invalid_library_uri_segments(self, uris: list[str], *, allowed: frozenset[str]) -> str | None:
        for uri in uris:
            if not isinstance(uri, str) or not uri.lower().startswith("spotify:"):
                continue
            seg = uri.split(":", 2)[1].strip().lower()
            if seg not in allowed:
                return json.dumps(
                    {
                        "ok": False,
                        "failure_reason": "invalid_uri_type",
                        "error": f"URI type {seg!r} is not allowed here (expected {sorted(allowed)}).",
                        "uri": uri,
                        "hint": "Use spotify_library_save for shows/episodes; save_tracks only accepts tracks.",
                    },
                    ensure_ascii=False,
                )
        return None

    def _library_save_uris(self, arguments: dict[str, Any]) -> str:
        uris = self._normalize_library_uris(arguments)
        if not uris:
            return _empty_library_args_error("spotify_library_save")
        blocked = self._reject_invalid_library_uri_segments(
            uris,
            allowed=frozenset({"track", "album", "artist", "playlist", "show", "episode", "audiobook"}),
        )
        if blocked:
            return blocked
        for offset in range(0, len(uris), _LIBRARY_URI_CHUNK):
            chunk = uris[offset : offset + _LIBRARY_URI_CHUNK]
            self.client.api_put("/me/library", params={"uris": ",".join(chunk)})
        self._record_library_mutation_from_uris(uris)
        label = self._library_label_for_uri(uris[0]) if uris else None
        payload: dict[str, Any] = {"ok": True, "saved_uris": uris}
        if label:
            payload["item_name"] = label
            payload["user_message"] = f"Saved {label} to your library."
        return json.dumps(payload, ensure_ascii=False)

    def _library_remove_uris(self, arguments: dict[str, Any]) -> str:
        uris = self._normalize_library_uris(arguments)
        if not uris:
            return _empty_library_args_error("spotify_library_remove")
        try:
            pre_check = self.client.api_get(
                "/me/library/contains",
                params={"uris": ",".join(uris[: _LIBRARY_URI_CHUNK])},
            )
        except httpx.HTTPStatusError:
            pre_check = None
        if isinstance(pre_check, list) and pre_check and not any(bool(x) for x in pre_check):
            label = self._library_label_for_uri(uris[0]) if uris else None
            name = label or "That item"
            return json.dumps(
                {
                    "ok": True,
                    "not_in_library": True,
                    "removed_uris": uris,
                    "item_name": label,
                    "user_message": f"{name} wasn't in your library.",
                },
                ensure_ascii=False,
            )
        for offset in range(0, len(uris), _LIBRARY_URI_CHUNK):
            chunk = uris[offset : offset + _LIBRARY_URI_CHUNK]
            self.client.api_delete("/me/library", params={"uris": ",".join(chunk)})
        try:
            check = self.client.api_get(
                "/me/library/contains",
                params={"uris": ",".join(uris[: _LIBRARY_URI_CHUNK])},
            )
            still = isinstance(check, list) and any(bool(x) for x in check)
        except httpx.HTTPStatusError:
            still = None
        if still is True:
            return json.dumps(
                {
                    "ok": False,
                    "failure_reason": "remove_verify_failed",
                    "error": "Spotify still reports at least one URI as saved after removal.",
                    "removed_uris": uris,
                },
                ensure_ascii=False,
            )
        label = self._library_label_for_uri(uris[0]) if uris else None
        payload_out: dict[str, Any] = {
            "ok": True,
            "removed_uris": uris,
            "verified_removed": still is False,
        }
        if label:
            payload_out["item_name"] = label
            payload_out["user_message"] = f"Removed {label} from your library."
        return json.dumps(payload_out, ensure_ascii=False)

    def _record_library_mutation_from_uris(self, uris: list[str]) -> None:
        from spot_backend.spotify_tools import _looks_like_spotify_catalog_id

        by_seg: dict[str, list[str]] = {}
        for uri in uris:
            if not isinstance(uri, str) or not uri.lower().startswith("spotify:"):
                continue
            parts = uri.split(":")
            if len(parts) < 3:
                continue
            seg, bare = parts[1], parts[2]
            if _looks_like_spotify_catalog_id(bare):
                by_seg.setdefault(seg, []).append(bare)
        if len(by_seg) == 1:
            seg, ids = next(iter(by_seg.items()))
            if hasattr(self, "_record_library_mutation"):
                self._record_library_mutation(seg, ids)

    def _get_show(self, arguments: dict[str, Any]) -> str:
        from spot_backend.spotify_tools import _normalize_spotify_id, _pick_arg

        sid = _normalize_spotify_id(_pick_arg(arguments, "show_id", "id"), "show")
        if not sid:
            return json.dumps({"error": "show_id is required", "failure_reason": "validation_error"})
        data = self.client.api_get_cached(f"/shows/{sid}")
        if isinstance(data, dict) and data.get("id"):
            self._session_known_ids.add(str(data["id"]))
        return _compact_pr9(data)

    def _get_show_episodes(self, arguments: dict[str, Any]) -> str:
        from spot_backend.spotify_tools import (
            _normalize_market,
            _normalize_spotify_id,
            _pick_arg,
            _safe_int,
        )

        sid = _normalize_spotify_id(_pick_arg(arguments, "show_id", "id"), "show")
        if not sid:
            return json.dumps({"error": "show_id is required", "failure_reason": "validation_error"})
        limit = clamp_spotify_page_limit(arguments.get("limit"), default=SPOTIFY_DEV_MAX_PAGE)
        offset = _safe_int(arguments.get("offset"), 0, lo=0, hi=900_000)
        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        if not market or len(market) != 2:
            market = "from_token"
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "market": market,
        }
        data = self.client.api_get(f"/shows/{sid}/episodes", params=params)
        if isinstance(data, dict):
            items = data.get("items")
            if isinstance(items, list):
                data["items"] = [it for it in items if isinstance(it, dict)]
                data["returned_count"] = len(data["items"])
        return _compact_pr9(data)

    def _get_episode(self, arguments: dict[str, Any]) -> str:
        from spot_backend.spotify_tools import _normalize_spotify_id, _pick_arg

        eid = _normalize_spotify_id(_pick_arg(arguments, "episode_id", "id"), "episode")
        if not eid:
            return json.dumps({"error": "episode_id is required", "failure_reason": "validation_error"})
        data = self.client.api_get_cached(f"/episodes/{eid}")
        return _compact_pr9(data)

    def _play_show_latest_episode(self, arguments: dict[str, Any]) -> str:
        from spot_backend.spotify_tools import _normalize_spotify_id, _pick_arg

        sid = _normalize_spotify_id(_pick_arg(arguments, "show_id", "id"), "show")
        if not sid:
            return json.dumps({"error": "show_id is required", "failure_reason": "validation_error"})
        from spot_backend.spotify_tools import _normalize_market

        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        ep_market = market if market and len(market) == 2 else "from_token"
        try:
            page = self.client.api_get(
                f"/shows/{sid}/episodes",
                params={"limit": 1, "offset": 0, "market": ep_market},
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return json.dumps(
                    {
                        "ok": False,
                        "failure_reason": "show_not_found",
                        "error": "Spotify could not find that podcast show.",
                        "user_message": "I couldn't find that show's latest episode.",
                        "show_id": sid,
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "ok": False,
                    "failure_reason": "no_episodes_for_show",
                    "error": _spotify_msg(e),
                    "user_message": "I couldn't find that show's latest episode.",
                    "show_id": sid,
                },
                ensure_ascii=False,
            )
        items = page.get("items") if isinstance(page, dict) else None
        if isinstance(items, list):
            items = [it for it in items if isinstance(it, dict)]
        if not isinstance(items, list) or not items or not isinstance(items[0], dict):
            return json.dumps(
                {
                    "ok": False,
                    "failure_reason": "no_episodes_for_show",
                    "error": "No episodes returned for that show.",
                    "user_message": "I couldn't find that show's latest episode.",
                    "show_id": sid,
                },
                ensure_ascii=False,
            )
        ep = items[0]
        uri = ep.get("uri")
        eid = ep.get("id")
        if not isinstance(uri, str):
            return json.dumps({"ok": False, "error": "Episode has no uri", "failure_reason": "parse_error"})
        if isinstance(eid, str):
            self._record_library_mutation("show", [sid])
            self._record_library_mutation("episode", [eid])
            self._session_known_ids.add(sid)
            self._session_known_ids.add(eid)
        play_args: dict[str, Any] = {"uris": [uri], "playback_request_label": "that episode"}
        device_id = _pick_arg(arguments, "device_id")
        if device_id:
            play_args["device_id"] = device_id
        raw = self._start_playback(play_args)
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return raw
        if isinstance(payload, dict) and payload.get("ok") is False:
            payload.setdefault("failure_reason", "playback_not_verified")
            ep_title = ep.get("name") if isinstance(ep.get("name"), str) else "the latest episode"
            payload["episode_name"] = ep_title
            payload["user_message"] = self._playback_not_verified_user_message(ep_title)
            return json.dumps(payload, ensure_ascii=False)
        if isinstance(payload, dict) and payload.get("playback_verified") is False:
            payload["ok"] = False
            payload.setdefault("failure_reason", "playback_not_verified")
            ep_title = ep.get("name") if isinstance(ep.get("name"), str) else "the latest episode"
            payload["episode_name"] = ep_title
            payload["user_message"] = self._playback_not_verified_user_message(ep_title)
            return json.dumps(payload, ensure_ascii=False)
        return raw

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
        if not isinstance(data, dict):
            return _compact_pr9(data)
        items_out: list[dict[str, Any]] = []
        raw = data.get("items")
        if isinstance(raw, list):
            for row in raw[:50]:
                if not isinstance(row, dict):
                    continue
                show = row.get("show") if isinstance(row.get("show"), dict) else row
                if not isinstance(show, dict):
                    continue
                items_out.append(
                    {
                        "added_at": row.get("added_at"),
                        "id": show.get("id"),
                        "name": show.get("name"),
                        "uri": show.get("uri"),
                        "publisher": show.get("publisher"),
                    }
                )
        from spot_backend.list_format import format_indexed_name_detail

        lines = [
            format_indexed_name_detail(
                offset + i + 1,
                str(r.get("name") or ""),
                str(r.get("publisher") or "") if r.get("publisher") else None,
                default_name="Show",
            )
            for i, r in enumerate(items_out)
        ]
        total = data.get("total")
        summary = "\n".join(lines) if lines else "You have no saved shows on this page."
        if isinstance(total, int) and total > len(items_out):
            more = total - len(items_out)
            summary += f"\n({total} saved shows total — and {more} more not listed here.)"
        elif data.get("next"):
            summary += "\n(and more saved shows not listed here.)"
        payload = {
            "ok": True,
            "total": data.get("total"),
            "offset": offset,
            "returned_count": len(items_out),
            "items": items_out,
            "summary_lines": lines,
            "user_message": summary,
        }
        return _compact_pr9(payload)

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

    def _builder_theme_blob(self, arguments: dict[str, Any]) -> str:
        from spot_backend.spotify_tools import _coerce_str, _pick_arg

        parts: list[str] = []
        theme = _coerce_str(
            _pick_arg(arguments, "theme", "description", "vibe", "name", "playlist_name"),
            "",
        ).strip()
        if theme:
            parts.append(theme)
        raw_queries = arguments.get("track_queries") or arguments.get("tracks") or []
        if isinstance(raw_queries, str):
            raw_queries = [raw_queries]
        if isinstance(raw_queries, list):
            for item in raw_queries:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
        return " ".join(parts).strip().lower()

    def _preview_row_fits_theme(self, row: dict[str, Any], theme_blob: str) -> bool:
        if not theme_blob:
            return True
        release = row.get("release_date")
        if isinstance(release, str) and release.strip():
            fake = {
                "name": row.get("name") or "",
                "album": {"name": "", "release_date": release},
            }
            return self._track_fits_builder_theme(fake, theme_blob)
        uri = row.get("uri")
        if isinstance(uri, str) and uri.startswith("spotify:track:"):
            bare = uri.split(":")[-1]
            try:
                track = self.client.api_get_cached(f"/tracks/{bare}")
            except Exception:
                return True
            if isinstance(track, dict):
                return self._track_fits_builder_theme(track, theme_blob)
        return True

    def _filter_preview_tracks_for_theme(
        self, tracks: list[dict[str, Any]], theme_blob: str
    ) -> list[dict[str, Any]]:
        if not theme_blob:
            return tracks
        return [row for row in tracks if isinstance(row, dict) and self._preview_row_fits_theme(row, theme_blob)]

    def _track_fits_builder_theme(self, track: dict[str, Any], theme_blob: str) -> bool:
        if not theme_blob:
            return True
        name = str(track.get("name") or "").lower()
        album = track.get("album") if isinstance(track.get("album"), dict) else {}
        album_name = str(album.get("name") or "").lower()
        combined = f"{name} {album_name}"
        if any(marker in combined for marker in _SOUNDTRACK_MARKERS):
            if "soundtrack" not in theme_blob and "game" not in theme_blob and "theme" not in theme_blob:
                return False
        if "90" in theme_blob or "1990" in theme_blob or "nineties" in theme_blob:
            release = str(album.get("release_date") or "")
            year_match = re.match(r"(\d{4})", release)
            if year_match:
                year = int(year_match.group(1))
                if year < 1990 or year > 1999:
                    return False
        return True

    def _title_mostly_echoes_query(self, track: dict[str, Any], query: str) -> bool:
        name = str(track.get("name") or "").lower()
        if not name:
            return False
        q_words = [w for w in re.findall(r"[a-z0-9']+", query.lower()) if len(w) > 2]
        if not q_words:
            return False
        hits = sum(1 for w in q_words if w in name)
        generic_hits = sum(1 for w in _GENERIC_QUERY_ECHO_WORDS if w in name)
        if hits >= max(2, len(q_words) // 2) and generic_hits >= 2:
            return True
        if len(name.split()) <= 4 and hits == len(q_words) and generic_hits >= 1:
            return True
        return False

    def _renumber_preview_tracks(self, tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i, row in enumerate(tracks):
            if not isinstance(row, dict):
                continue
            copy = dict(row)
            copy["n"] = i + 1
            out.append(copy)
        return out

    def _preview_payload_from_tracks(self, name: str, tracks: list[dict[str, Any]]) -> dict[str, Any]:
        numbered = self._renumber_preview_tracks(tracks)
        lines = [f'Proposed playlist "{name}" ({len(numbered)} tracks):']
        for row in numbered:
            lines.append(f"{row['n']}. {row.get('name')} — {row.get('artist')}")
        lines.append("Reply yes / make it to create this playlist, or ask to edit the list.")
        preview_text = "\n".join(lines)
        return {
            "ok": True,
            "preview": {"proposed_name": name, "tracks": numbered, "awaiting_approval": True},
            "preview_text": preview_text,
            "awaiting_approval": True,
            "message": preview_text,
        }

    def _resolve_track_query(
        self,
        query: str,
        market: str,
        *,
        pick_index: int = 0,
        theme_blob: str = "",
    ) -> dict[str, Any] | None:
        from spot_backend.spotify_tools import _normalize_market

        q = query.strip()
        if not q:
            return None
        if _theme_requests_nineties(theme_blob) and "year:" not in q.lower():
            q = f"{q} year:1990-1999"
        import os

        if os.environ.get("SPOT_DEBUG_BUILDER_Q", "").strip() in ("1", "true", "yes"):
            logger.warning("playlist_builder_search q=%s", q)
        else:
            logger.info("playlist_builder_search q=%s", q)
        if os.environ.get("SPOT_DEBUG_BUILDER_Q", "").strip() in ("1", "true", "yes"):
            trace_dir = getattr(self.settings, "data_dir", None)
            if trace_dir is not None:
                from pathlib import Path

                from spot_backend.reply_tool_trace import append_tool_trace_record

                append_tool_trace_record(
                    Path(trace_dir),
                    conversation_id=self.conversation_id,
                    tool_name="playlist_builder_search",
                    args_summary="{}",
                    outcome="ok",
                    trace_fields={"q": q},
                )
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
        if not isinstance(items, list) or not items:
            return None
        idx = pick_index % len(items)
        row = items[idx]
        if isinstance(row, dict) and row.get("uri"):
            return row
        for alt in items:
            if isinstance(alt, dict) and alt.get("uri"):
                return alt
        return None

    def _builder_seed_queries(self, arguments: dict[str, Any]) -> list[str]:
        from spot_backend.spotify_tools import _coerce_str, _pick_arg

        raw = arguments.get("track_queries") or arguments.get("tracks") or []
        if isinstance(raw, str):
            raw = [raw]
        seeds: list[str] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    seeds.append(item.strip())
        theme = _coerce_str(
            _pick_arg(arguments, "theme", "description", "vibe", "name", "playlist_name"),
            "",
        ).strip()
        blob = theme.lower()
        if _theme_requests_nineties(blob):
            seeds.append("year:1990-1999 genre:r&b")
            seeds.append("year:1990-1999 genre:rock")
            decades = ["1990s", "1991", "1994", "1995", "1997", "1998"]
            moods = ["chill", "relax", "slow", "soft", "easy listening"]
            for d in decades:
                for m in moods:
                    seeds.append(f"{d} {m}")
                    if len(seeds) >= 40:
                        break
                if len(seeds) >= 40:
                    break
        if "chill" in blob:
            seeds.extend(
                [
                    "chill vibes",
                    "chillout classics",
                    "relaxing hits",
                    "mellow 90s",
                    "downtempo",
                    "lofi chill",
                    "ambient pop",
                ]
            )
        generic = [
            "chill track",
            "relaxing song",
            "soft rock 90s",
            "easy listening",
            "mellow hits",
            "slow jam",
            "acoustic chill",
            "indie chill",
            "rnb chill",
            "soul mellow",
        ]
        seeds.extend(generic)
        seen: set[str] = set()
        out: list[str] = []
        for s in seeds:
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    def _playlist_builder_preview(self, arguments: dict[str, Any]) -> str:
        from spot_backend.spotify_tools import _coerce_str, _normalize_market, _pick_arg

        name = _coerce_str(_pick_arg(arguments, "name", "playlist_name"), "New mix").strip()
        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        theme_blob = self._builder_theme_blob(arguments)
        queries = self._builder_seed_queries(arguments)
        resolved: list[dict[str, Any]] = []
        seen_uris: set[str] = set()
        for raw_q in queries:
            q = raw_q.strip()
            if not q:
                continue
            if len(resolved) >= _MAX_BUILDER_TRACKS:
                break
            for pick_index in range(12):
                if len(resolved) >= _MAX_BUILDER_TRACKS:
                    break
                track = self._resolve_track_query(q, market, pick_index=pick_index, theme_blob=theme_blob)
                if not track:
                    break
                if not self._track_fits_builder_theme(track, theme_blob):
                    continue
                if self._title_mostly_echoes_query(track, q):
                    continue
                uri = track.get("uri")
                if not isinstance(uri, str) or uri in seen_uris:
                    continue
                seen_uris.add(uri)
                artists = track.get("artists") if isinstance(track.get("artists"), list) else []
                artist_name = ""
                if artists and isinstance(artists[0], dict):
                    artist_name = str(artists[0].get("name") or "")
                album = track.get("album") if isinstance(track.get("album"), dict) else {}
                release_date = album.get("release_date") if isinstance(album.get("release_date"), str) else ""
                resolved.append(
                    {
                        "n": len(resolved) + 1,
                        "uri": uri,
                        "name": track.get("name"),
                        "artist": artist_name,
                        "query": q,
                        "release_date": release_date,
                    }
                )
                break
        if len(resolved) < _MIN_BUILDER_TRACKS:
            return json.dumps(
                {
                    "ok": False,
                    "failure_reason": "playlist_preview_insufficient",
                    "error": (
                        f"Resolved only {len(resolved)} tracks (need at least {_MIN_BUILDER_TRACKS}). "
                        "Try broader theme keywords or different decades."
                    ),
                    "resolved_count": len(resolved),
                },
                ensure_ascii=False,
            )
        preview = {
            "proposed_name": name,
            "tracks": resolved,
            "awaiting_approval": True,
            "theme_blob": theme_blob,
        }
        save_playlist_preview(self.conversation_id, preview)
        return json.dumps(self._preview_payload_from_tracks(name, resolved), ensure_ascii=False)

    def _playlist_builder_edit(self, arguments: dict[str, Any]) -> str:
        from spot_backend.spotify_tools import _coerce_str, _normalize_market, _pick_arg, _safe_int

        preview = load_playlist_preview(self.conversation_id)
        if not preview:
            return json.dumps(
                {
                    "error": "No pending playlist preview in this conversation.",
                    "failure_reason": "validation_error",
                    "hint": "Call spotify_playlist_builder_preview first.",
                },
                ensure_ascii=False,
            )
        name = _coerce_str(preview.get("proposed_name"), "New mix").strip()
        tracks = preview.get("tracks")
        if not isinstance(tracks, list):
            tracks = []
        working = [dict(t) for t in tracks if isinstance(t, dict)]

        remove_raw = arguments.get("remove_indices") or arguments.get("remove") or arguments.get("drop")
        if isinstance(remove_raw, int):
            remove_raw = [remove_raw]
        if isinstance(remove_raw, list):
            to_drop: set[int] = set()
            for item in remove_raw:
                idx = _safe_int(item, -1, lo=1, hi=500)
                if idx > 0:
                    to_drop.add(idx)
            if to_drop:
                working = [row for row in working if int(row.get("n") or 0) not in to_drop]

        add_queries = arguments.get("add_queries") or arguments.get("add_tracks") or []
        if isinstance(add_queries, str):
            add_queries = [add_queries]
        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        theme_blob = (
            self._builder_theme_blob(arguments)
            or str(preview.get("theme_blob") or "").strip()
            or name.lower()
        )
        seen_uris = {str(r.get("uri")) for r in working if r.get("uri")}
        if isinstance(add_queries, list):
            for raw_q in add_queries:
                if not isinstance(raw_q, str) or not raw_q.strip():
                    continue
                track = self._resolve_track_query(raw_q.strip(), market, pick_index=len(working), theme_blob=theme_blob)
                if not track or not self._track_fits_builder_theme(track, theme_blob):
                    continue
                if self._title_mostly_echoes_query(track, raw_q.strip()):
                    continue
                uri = track.get("uri")
                if not isinstance(uri, str) or uri in seen_uris:
                    continue
                seen_uris.add(uri)
                artists = track.get("artists") if isinstance(track.get("artists"), list) else []
                artist_name = str(artists[0].get("name") or "") if artists and isinstance(artists[0], dict) else ""
                working.append(
                    {
                        "uri": uri,
                        "name": track.get("name"),
                        "artist": artist_name,
                        "query": raw_q.strip(),
                    }
                )

        replace_index = _safe_int(arguments.get("replace_index"), 0, lo=0, hi=500)
        replace_query = _coerce_str(arguments.get("replace_query"), "").strip()
        if replace_index > 0 and replace_query:
            if 1 <= replace_index <= len(working):
                track = self._resolve_track_query(replace_query, market, pick_index=replace_index - 1, theme_blob=theme_blob)
                if track and self._track_fits_builder_theme(track, theme_blob):
                    if self._title_mostly_echoes_query(track, replace_query):
                        track = None
                if track and self._track_fits_builder_theme(track, theme_blob):
                    uri = track.get("uri")
                    if isinstance(uri, str):
                        artists = track.get("artists") if isinstance(track.get("artists"), list) else []
                        artist_name = (
                            str(artists[0].get("name") or "") if artists and isinstance(artists[0], dict) else ""
                        )
                        working[replace_index - 1] = {
                            "uri": uri,
                            "name": track.get("name"),
                            "artist": artist_name,
                            "query": replace_query,
                        }

        working = self._filter_preview_tracks_for_theme(working, theme_blob)
        working = self._renumber_preview_tracks(working)
        if not working:
            return json.dumps(
                {
                    "ok": False,
                    "failure_reason": "playlist_preview_insufficient",
                    "error": "Preview would have no tracks after edit.",
                    "resolved_count": 0,
                },
                ensure_ascii=False,
            )
        save_playlist_preview(
            self.conversation_id,
            {"proposed_name": name, "tracks": working, "awaiting_approval": True},
        )
        return json.dumps(self._preview_payload_from_tracks(name, working), ensure_ascii=False)

    def _ensure_playlist_private(self, playlist_id: str) -> tuple[bool, bool | None, str | None]:
        """Return (verified_private, public_readback, user_note)."""
        pid = (playlist_id or "").strip()
        if not pid:
            return False, None, None
        public_flag: bool | None = None
        try:
            meta = self.client.api_get(f"/playlists/{pid}", params={"fields": "id,public,collaborative"})
            if isinstance(meta, dict) and isinstance(meta.get("public"), bool):
                public_flag = meta.get("public")
        except httpx.HTTPStatusError:
            return False, None, None
        if public_flag is False:
            return True, False, None
        try:
            self.client.api_put(f"/playlists/{pid}", json_body={"public": False, "collaborative": False})
        except httpx.HTTPStatusError:
            return False, public_flag, None
        for delay in (0.35, 0.75, 1.5):
            time.sleep(delay)
            try:
                meta2 = self.client.api_get(f"/playlists/{pid}", params={"fields": "id,public"})
                if isinstance(meta2, dict) and isinstance(meta2.get("public"), bool):
                    if meta2.get("public") is False:
                        return True, False, None
                    public_flag = meta2.get("public")
            except httpx.HTTPStatusError:
                break
        note = _PLAYLIST_PRIVACY_USER_NOTE
        return False, public_flag, note

    def _playlist_builder_commit(self, arguments: dict[str, Any]) -> str:
        from spot_backend.spotify_tools import _coerce_str, _pick_arg

        preview = load_playlist_preview(self.conversation_id)
        if not preview:
            return json.dumps(
                {
                    "error": "No pending playlist preview in this conversation.",
                    "failure_reason": "validation_error",
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
        theme_blob = str(preview.get("theme_blob") or name).strip().lower()
        tracks = preview.get("tracks")
        if not isinstance(tracks, list) or not tracks:
            return json.dumps({"error": "Preview has no tracks", "failure_reason": "validation_error"})
        tracks = self._filter_preview_tracks_for_theme(
            [t for t in tracks if isinstance(t, dict)],
            theme_blob,
        )
        if not tracks:
            return json.dumps(
                {
                    "ok": False,
                    "failure_reason": "playlist_preview_insufficient",
                    "error": "No tracks left in the preview after applying the theme filter.",
                },
                ensure_ascii=False,
            )
        uris = [t.get("uri") for t in tracks if isinstance(t, dict) and t.get("uri")]
        uris = [u for u in uris if isinstance(u, str)]
        if not uris:
            return json.dumps({"error": "Preview has no track URIs", "failure_reason": "validation_error"})
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
                add_error = f"HTTP {e.response.status_code}: {_spotify_msg(e)}"
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
        verified_private, public_readback, privacy_note = self._ensure_playlist_private(pid)
        if hasattr(self, "note_session_playlist_id"):
            self.note_session_playlist_id(pid)
        payload: dict[str, Any] = {
            "ok": True,
            "playlist_id": pid,
            "name": name,
            "public": False if verified_private else public_readback,
            "verified_private": verified_private,
            "track_count": len(uris),
        }
        if not verified_private:
            payload["privacy_warning"] = _PLAYLIST_PRIVACY_USER_NOTE
            payload["user_message"] = f'Created playlist "{name}" with {len(uris)} tracks.'
        else:
            payload["user_message"] = f'Created private playlist "{name}" with {len(uris)} tracks.'
        return json.dumps(payload, ensure_ascii=False)

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


def _spotify_msg(exc: httpx.HTTPStatusError) -> str:
    try:
        payload = exc.response.json()
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict) and isinstance(err.get("message"), str):
                return err["message"]
    except (json.JSONDecodeError, ValueError):
        pass
    return (exc.response.text or "")[:200]
