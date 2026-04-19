"""Spot-AI-fy: Spotify Web API tool definitions and executor (shared by MCP, Ollama, Gemini)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from spot_backend.config import Settings, get_settings
from spot_backend.spotify_client import SpotifyAuthError, SpotifyClient
from spot_backend.token_store import load_device

logger = logging.getLogger(__name__)


def _compact(data: Any, limit: int = 6000) -> str:
    s = json.dumps(data, ensure_ascii=False)
    if len(s) > limit:
        return s[:limit] + "\n... (truncated)"
    return s


def _coerce_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip() or default
    if isinstance(v, list):
        parts = [str(x).strip() for x in v if x is not None and str(x).strip()]
        return ",".join(parts) if parts else default
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)
    s = str(v).strip()
    return s or default


def _safe_int(v: Any, default: int, *, lo: int | None = None, hi: int | None = None) -> int:
    try:
        x = int(float(v))
    except (TypeError, ValueError):
        return default
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x


def _pick_arg(arguments: dict[str, Any], *keys: str, default: str = "") -> str:
    for k in keys:
        s = _coerce_str(arguments.get(k), "")
        if s:
            return s
    return default


def _normalize_spotify_id(raw: str, segment: str) -> str:
    """Strip spotify: URIs and open.spotify.com URLs so /v1 paths use bare ids."""
    s = raw.strip()
    if not s:
        return s
    low = s.lower()
    prefix = f"spotify:{segment}:"
    if low.startswith(prefix):
        return s[len(prefix) :].split("?", 1)[0].split("/")[0]
    for base in (f"https://open.spotify.com/{segment}/", f"http://open.spotify.com/{segment}/"):
        bl = base.lower()
        if low.startswith(bl):
            tail = s[len(base) :].split("?", 1)[0].strip().strip("/")
            return tail.split("/")[0] if tail else s
    return s.split("?", 1)[0].strip()


def _normalize_include_groups(s: str) -> str:
    """Spotify only accepts album, single, appears_on, compilation."""
    raw = (s or "").strip().replace(" ", "")
    if not raw:
        return "album,single"
    allowed = {"album", "single", "appears_on", "compilation"}
    parts = [p for p in raw.split(",") if p in allowed]
    return ",".join(parts) if parts else "album,single"


def _looks_like_spotify_catalog_id(s: str) -> bool:
    """Spotify track/artist/album ids are 22-char base62-ish strings."""
    if len(s) != 22:
        return False
    return all(c.isalnum() for c in s)


def _normalize_market(m: str) -> str:
    """Spotify expects ISO 3166-1 alpha-2 or the literal from_token."""
    s = (m or "").strip()
    if not s:
        return "from_token"
    if s.lower() == "from_token":
        return "from_token"
    if len(s) == 2 and s.isalpha():
        return s.upper()
    return "from_token"


def _shrink_user_playlists_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Strip heavy fields so local LLMs are not fed megabytes of playlist metadata."""
    items_out: list[dict[str, Any]] = []
    raw_items = data.get("items")
    if isinstance(raw_items, list):
        for it in raw_items[:50]:
            if not isinstance(it, dict):
                continue
            pid = it.get("id")
            name = it.get("name")
            owner = it.get("owner") if isinstance(it.get("owner"), dict) else {}
            row: dict[str, Any] = {
                "id": pid if isinstance(pid, str) else None,
                "name": str(name) if isinstance(name, str) else "",
                "owner_id": owner.get("id") if isinstance(owner.get("id"), str) else None,
                "collaborative": bool(it.get("collaborative")),
                "public": it.get("public"),
            }
            if not isinstance(row["id"], str):
                continue
            items_out.append(row)
    return {
        "total": data.get("total"),
        "limit": data.get("limit"),
        "offset": data.get("offset"),
        "has_next_page": bool(data.get("next")),
        "note": (
            "owner_id identifies the playlist owner. Compare to spotify_me.id to decide if you can "
            "write to it (add/remove tracks). Playlists you merely follow list here too."
        ),
        "items": items_out,
    }


def _shrink_playlist_tracks_items(data: dict[str, Any]) -> dict[str, Any]:
    """Lightweight track rows for LLM context.

    Accepts both legacy `{items: [{track: {...}}]}` and the Feb-2026 renamed shape
    `{items: [{item: {...}}]}` where row.track → row.item.
    """
    raw_items = data.get("items")
    out_items: list[dict[str, Any]] = []
    if isinstance(raw_items, list):
        for row in raw_items[:100]:
            if not isinstance(row, dict):
                continue
            tr = row.get("track") if isinstance(row.get("track"), dict) else row.get("item")
            if not isinstance(tr, dict) or tr.get("id") is None:
                out_items.append({"track": None, "is_local": row.get("is_local")})
                continue
            artists = tr.get("artists")
            anames: list[str] = []
            if isinstance(artists, list):
                for a in artists:
                    if isinstance(a, dict) and a.get("name"):
                        anames.append(str(a["name"]))
            out_items.append(
                {
                    "name": tr.get("name"),
                    "id": tr.get("id"),
                    "uri": tr.get("uri"),
                    "duration_ms": tr.get("duration_ms"),
                    "artists": anames,
                }
            )
    return {
        "total": data.get("total"),
        "limit": data.get("limit"),
        "offset": data.get("offset"),
        "has_next_page": bool(data.get("next")),
        "items": out_items,
    }


def _shrink_playlist_object(data: dict[str, Any]) -> dict[str, Any]:
    """Playlist metadata + slim track page (when present).

    The Feb-2026 Spotify rename changed the playlist object field `tracks` to `items`
    and may omit it entirely for playlists the user does not own. Accept both shapes.
    """
    owner = data.get("owner")
    owner_out: dict[str, Any] = {}
    if isinstance(owner, dict):
        owner_out = {"id": owner.get("id"), "display_name": owner.get("display_name")}
    page = data.get("items") if isinstance(data.get("items"), dict) else data.get("tracks")
    tracks_out: dict[str, Any] = {}
    if isinstance(page, dict):
        tracks_out = _shrink_playlist_tracks_items(page)
    return {
        "id": data.get("id"),
        "name": data.get("name"),
        "description": (data.get("description") or "")[:500],
        "public": data.get("public"),
        "collaborative": data.get("collaborative"),
        "snapshot_id": data.get("snapshot_id"),
        "owner": owner_out,
        "tracks": tracks_out,
    }


def _shrink_saved_tracks_page(data: dict[str, Any]) -> dict[str, Any]:
    out_items: list[dict[str, Any]] = []
    raw = data.get("items")
    if isinstance(raw, list):
        for row in raw[:50]:
            if not isinstance(row, dict):
                continue
            tr = row.get("track")
            if not isinstance(tr, dict):
                continue
            artists = tr.get("artists")
            anames: list[str] = []
            if isinstance(artists, list):
                for a in artists:
                    if isinstance(a, dict) and a.get("name"):
                        anames.append(str(a["name"]))
            out_items.append(
                {
                    "added_at": row.get("added_at"),
                    "name": tr.get("name"),
                    "id": tr.get("id"),
                    "uri": tr.get("uri"),
                    "duration_ms": tr.get("duration_ms"),
                    "artists": anames,
                }
            )
    return {
        "total": data.get("total"),
        "limit": data.get("limit"),
        "offset": data.get("offset"),
        "has_next_page": bool(data.get("next")),
        "items": out_items,
    }


def _coerce_track_uri_list(uris: Any) -> list[str] | None:
    """Build spotify:track: URIs from strings, bare ids, or track-shaped dicts (search results).

    Only yields URIs whose id portion is a 22-char base62 catalog id. This prevents
    the caller from smuggling in an artist/album/episode id under `spotify:track:…`,
    which Spotify accepts as a playlist add but stores as an empty (ghost) row.
    """
    if not isinstance(uris, list) or not uris:
        return None
    out: list[str] = []
    for u in uris:
        if isinstance(u, dict):
            tr = u.get("track")
            if isinstance(tr, dict):
                raw = u.get("uri") or u.get("id") or tr.get("uri") or tr.get("id")
                u_type = (u.get("type") or tr.get("type") or "").lower()
            else:
                raw = u.get("uri") or u.get("id")
                u_type = (u.get("type") or "").lower()
            if u_type and u_type != "track":
                continue
            if raw is None:
                continue
            s = str(raw).strip()
        else:
            s = str(u).strip()
        if not s or s.startswith("{"):
            continue
        low = s.lower()
        if low.startswith("spotify:track:"):
            tid = s[len("spotify:track:") :].split("?", 1)[0].split("/")[0]
            if _looks_like_spotify_catalog_id(tid):
                out.append(f"spotify:track:{tid}")
            continue
        tid = _normalize_spotify_id(s, "track")
        if tid and _looks_like_spotify_catalog_id(tid):
            out.append(f"spotify:track:{tid}")
    return out if out else None


def _verify_tracks_exist(
    client: SpotifyClient, uri_list: list[str]
) -> tuple[list[str], list[str]]:
    """Check that each spotify:track: URI resolves to a real track before POST.

    Prevents ghost rows when a caller sent `spotify:track:{id}` where the id is actually
    an artist/album/episode id. Uses single-track GET /tracks/{id} because Spotify's
    Feb 2026 dev-mode rules block the batch GET /tracks?ids endpoint (403). A 404 is
    treated as "not a track" (invalid). Any other network/HTTP failure falls back to
    "valid" for that uri to avoid blocking adds during Spotify outages.
    """
    if not uri_list:
        return [], []
    valid: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for uri in uri_list:
        if uri in seen:
            continue
        seen.add(uri)
        tid = uri.split(":")[-1]
        try:
            obj = client.api_get(f"/tracks/{tid}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                invalid.append(uri)
                continue
            valid.append(uri)
            continue
        if isinstance(obj, dict) and obj.get("type") == "track" and isinstance(obj.get("id"), str):
            valid.append(uri)
        else:
            invalid.append(uri)
    return valid, invalid


def _flatten_spotify_error_text(detail: Any, spotify_api_message: str | None) -> str:
    """Lowercased blob for heuristic matching (Spotify Web API error shapes vary)."""
    parts: list[str] = []
    if spotify_api_message:
        parts.append(spotify_api_message)
    if isinstance(detail, dict):
        inner = detail.get("error")
        if isinstance(inner, dict) and inner.get("message"):
            parts.append(str(inner["message"]))
        if isinstance(detail.get("error_description"), str):
            parts.append(detail["error_description"])
    elif isinstance(detail, str):
        parts.append(detail)
    return " ".join(parts).lower()


def _spotify_error_suggests_reauth_or_scope(detail: Any, spotify_api_message: str | None) -> bool:
    """True when Spotify's payload wording suggests missing scope, bad token, or re-consent — not heuristics on status alone."""
    blob = _flatten_spotify_error_text(detail, spotify_api_message)
    if not blob.strip():
        return False
    phrases = (
        "insufficient client scope",
        "insufficient scope",
        "missing scope",
        "invalid scope",
        "bad or expired token",
        "expired token",
        "invalid token",
        "bad oauth",
        "invalid oauth",
        "not authorized",
        "authorisation required",
        "authorization required",
        "re-authorization",
        "reauthorization",
        "consent required",
        "invalid_grant",
    )
    return any(p in blob for p in phrases)


def _spotify_403_message_is_scope_ambiguous(spotify_api_message: str | None) -> bool:
    """Spotify often returns only 'Forbidden' with no scope hint — treat as ambiguous (id vs OAuth)."""
    if spotify_api_message is None:
        return True
    s = spotify_api_message.strip().lower().rstrip(".")
    if not s:
        return True
    return s in ("forbidden", "not allowed", "access denied")


# Scopes each tool needs Spotify to have granted on the token's initial consent.
# Spotify refresh tokens do NOT upgrade to newer scopes added later — if the user's
# original /authorize consent was narrower, they must Sign out → Connect to re-consent.
_MODIFY_PLAYLIST_SCOPES = ("playlist-modify-public", "playlist-modify-private")
_READ_PLAYLIST_SCOPES = ("playlist-read-private", "playlist-read-collaborative")

_TOOL_REQUIRED_SCOPES: dict[str, tuple[str, ...]] = {
    "spotify_add_tracks_to_playlist": _MODIFY_PLAYLIST_SCOPES,
    "spotify_remove_playlist_tracks": _MODIFY_PLAYLIST_SCOPES,
    "spotify_replace_playlist_tracks": _MODIFY_PLAYLIST_SCOPES,
    "spotify_reorder_playlist_tracks": _MODIFY_PLAYLIST_SCOPES,
    "spotify_update_playlist": _MODIFY_PLAYLIST_SCOPES,
    "spotify_create_playlist": _MODIFY_PLAYLIST_SCOPES,
    "spotify_playlist_tracks": _READ_PLAYLIST_SCOPES,
    "spotify_get_playlist": _READ_PLAYLIST_SCOPES,
}


def _missing_any_of(granted: set[str], required_any_of: tuple[str, ...]) -> list[str]:
    """Return the scope list if none of them are granted (empty list = at least one granted)."""
    if not required_any_of:
        return []
    if any(s in granted for s in required_any_of):
        return []
    return list(required_any_of)


_PLAYLIST_ID_ARG_TOOLS = frozenset(
    {
        "spotify_add_tracks_to_playlist",
        "spotify_playlist_tracks",
        "spotify_get_playlist",
        "spotify_remove_playlist_tracks",
        "spotify_replace_playlist_tracks",
        "spotify_reorder_playlist_tracks",
        "spotify_update_playlist",
        "spotify_unfollow_playlist",
    }
)

_PLAYLIST_READ_ASSISTANT_GUIDANCE = (
    "This HTTP error applies only to the playlist_id in this request. If spotify_user_playlists already returned "
    "items in this chat, you MUST NOT tell the user you cannot access, list, or count their playlists in general — "
    "you can list them. Retry spotify_playlist_tracks or spotify_get_playlist with a different id from "
    "spotify_user_playlists (paginate offset). To play a playlist you do not need track listing: use "
    "spotify_start_resume_playback with context_uri spotify:playlist:<id> and an active device. "
    "If read_403_ambiguous is true, do not open with Sign out or blame 'collaborative' — check owner vs spotify_me first."
)


_ADD_TRACKS_ASSISTANT_GUIDANCE = (
    "Do NOT tell the user the playlist is inaccessible, locked, or that you lack access to it, and do not pivot to "
    "another artist or 'play without a playlist' unless they asked. "
    "Do NOT offer to create a new, replacement, or differently named playlist as your first suggestion after a "
    "failed add — the user already chose a target playlist. Instead: paginate spotify_user_playlists until the "
    "exact name matches and use that item's `id`; or reuse `id` / `playlist_id_for_add_tracks` from "
    "spotify_create_playlist in this chat. If you still see HTTP 403, call spotify_get_playlist with the id you "
    "used and spotify_me — compare playlist owner id to the current user id before claiming anything about access. "
    "Do NOT say you lack permission, that the playlist is not owned by the user, or that this 'usually happens' "
    "because of ownership — that is often wrong: the usual fix is a bad playlist_id (not from spotify_user_playlists "
    "or spotify_create_playlist) or an empty/malformed track list. A matching playlist *name* is not proof of id. "
    "Only mention ownership if you quote Spotify's spotify_api_message verbatim and it explicitly says so. "
    "HTTP 401 always means re-authenticate in this app. On HTTP 403, if `reauth_may_resolve` is true: you may suggest "
    "Sign out → Connect when spotify_api_message matched explicit scope/token wording, OR when `reauth_heuristic_ambiguous_403` "
    "is true (Spotify returned a generic Forbidden) — in the ambiguous case, tell the user to re-fetch playlist_id from "
    "spotify_user_playlists first (22-char track vs playlist ids look identical), then try Sign out → Connect to pick up "
    "scopes like playlist-read-collaborative. If `reauth_may_resolve` is false on 403, fix playlist_id and track URIs first. "
    "Do NOT claim you set the playlist collaborative, changed settings, or 'confirmed' ownership unless those "
    "exact tool results appear in the conversation (e.g. spotify_update_playlist ok, get_playlist.owner.id vs me.id). "
    "Quote spotify_api_message/detail when helpful. Retry spotify_add_tracks_to_playlist with the corrected "
    "playlist_id plus tracks from spotify_search (each item's `uri` or `id`; pass `tracks`, `track_uris`, `uris`, "
    "or `track_ids`; a search paging object `{items: [...]}` under any of those keys is OK)."
)


def _extend_search_trackish_bucket(combined: list[Any], v: Any) -> None:
    """List of strings/objects, or spotify_search-style object { \"items\": [ ... ] }, or one URI/id string."""
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return
        low = s.lower()
        if low.startswith("spotify:track:") or _looks_like_spotify_catalog_id(s):
            combined.append(s)
        return
    if isinstance(v, list) and v:
        combined.extend(v)
    elif isinstance(v, dict):
        items = v.get("items")
        if isinstance(items, list) and items:
            combined.extend(items)


def _combined_track_inputs(arguments: dict[str, Any]) -> list[Any]:
    """Merge common LLM argument shapes for track lists."""
    combined: list[Any] = []
    for key in ("track_uris", "uris", "track_ids", "tracks"):
        _extend_search_trackish_bucket(combined, arguments.get(key))
    return combined


class SpotifyToolRunner:
    def __init__(self, client: SpotifyClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or SpotifyClient(settings=settings or get_settings())
        self.settings = self.client.settings

    def close(self) -> None:
        self.client.close()

    def _device_id(self) -> str | None:
        d = load_device(self.settings.resolved_device_path)
        return d.device_id if d else None

    def _playlist_owner_snapshot(self, playlist_id: str) -> dict[str, Any]:
        """Rich pre-flight diagnostic for playlist tools.

        Keys:
          me_id, me_status: current user id and /me HTTP status
          owner_id, owner_name, playlist_name, playlist_status: from /playlists/{id} lookup
          is_owned: bool when both ids known; else None (unknown)
          granted_scopes: scopes Spotify granted the stored token
        """
        out: dict[str, Any] = {
            "me_id": None,
            "me_status": None,
            "owner_id": None,
            "owner_name": None,
            "playlist_name": None,
            "playlist_status": None,
            "is_owned": None,
            "granted_scopes": sorted(self.client.get_token_scopes()),
        }
        try:
            me_data = self.client.api_get("/me")
            out["me_status"] = 200
            if isinstance(me_data, dict) and isinstance(me_data.get("id"), str):
                out["me_id"] = me_data["id"]
        except httpx.HTTPStatusError as e:
            out["me_status"] = e.response.status_code
        try:
            pl_data = self.client.api_get(
                f"/playlists/{playlist_id}",
                params={"fields": "id,name,owner(id,display_name)"},
            )
            out["playlist_status"] = 200
            if isinstance(pl_data, dict):
                out["playlist_name"] = pl_data.get("name")
                owner = pl_data.get("owner")
                if isinstance(owner, dict):
                    if isinstance(owner.get("id"), str):
                        out["owner_id"] = owner["id"]
                    if isinstance(owner.get("display_name"), str):
                        out["owner_name"] = owner["display_name"]
        except httpx.HTTPStatusError as e:
            out["playlist_status"] = e.response.status_code
        if isinstance(out["me_id"], str) and isinstance(out["owner_id"], str):
            out["is_owned"] = out["me_id"] == out["owner_id"]
        return out

    def run(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            return self._dispatch(name, arguments)
        except SpotifyAuthError as e:
            return json.dumps({"error": str(e)})
        except httpx.HTTPStatusError as e:
            detail: Any
            try:
                detail = e.response.json()
            except Exception:
                detail = ((e.response.text or "")[:800] or str(e))
            err: dict[str, Any] = {
                "error": f"Spotify HTTP {e.response.status_code} for {name}",
                "detail": detail,
            }
            if isinstance(detail, dict):
                inner = detail.get("error")
                if isinstance(inner, dict):
                    msg = inner.get("message")
                    if isinstance(msg, str) and msg.strip():
                        err["spotify_api_message"] = msg.strip()
            spot_msg = err.get("spotify_api_message")
            if e.response.status_code == 401:
                err["reauth_may_resolve"] = True
            elif e.response.status_code == 403:
                if _spotify_error_suggests_reauth_or_scope(detail, spot_msg):
                    err["reauth_may_resolve"] = True
                elif _spotify_403_message_is_scope_ambiguous(spot_msg) and name in _PLAYLIST_ID_ARG_TOOLS:
                    # Generic "Forbidden" on *read* is often followed-not-owned or wrong id — do not set reauth flags
                    # (avoids the model defaulting to Sign out). Write tools still get reauth_heuristic_ambiguous_403.
                    if name in ("spotify_playlist_tracks", "spotify_get_playlist"):
                        err["read_403_ambiguous"] = True
                    else:
                        err["reauth_may_resolve"] = True
                        err["reauth_heuristic_ambiguous_403"] = True
            if e.response.status_code == 403:
                playlist_write = name in (
                    "spotify_add_tracks_to_playlist",
                    "spotify_remove_playlist_tracks",
                    "spotify_replace_playlist_tracks",
                    "spotify_reorder_playlist_tracks",
                    "spotify_update_playlist",
                )
                if playlist_write:
                    err["explain_playlist_id_before_reconnect"] = True
                    if name == "spotify_add_tracks_to_playlist":
                        reauth = bool(err.get("reauth_may_resolve"))
                        head = (
                            "403 on spotify_add_tracks_to_playlist: most often playlist_id is wrong for writes "
                            "(not the exact `id` from spotify_create_playlist in this chat, or not from "
                            "spotify_user_playlists — e.g. a catalog/search id, or a stale/wrong id). "
                            "Do not assume 'not owned' from the playlist title alone. "
                            "Do not suggest creating a substitute playlist — paginate spotify_user_playlists, match "
                            "the exact name to `id`, then retry add; optionally spotify_get_playlist + spotify_me to "
                            "compare owner id. "
                            "If spotify_create_playlist already succeeded for this user request, retry add with that "
                            "response `id` and non-empty track URIs or track_ids from spotify_search. "
                        )
                        if reauth:
                            if err.get("reauth_heuristic_ambiguous_403"):
                                err["hint"] = (
                                    head
                                    + " Spotify returned a generic Forbidden (reauth_heuristic_ambiguous_403). "
                                    "Common causes: (1) playlist you follow but do not own — spotify_user_playlists "
                                    "lists both; use spotify_get_playlist and compare owner.id to spotify_me.id before "
                                    "adding. (2) wrong 22-char id (track vs playlist). (3) missing playlist-modify-* "
                                    "scopes — Sign out → Connect. Dashboard: https://developer.spotify.com/dashboard"
                                )
                            else:
                                err["hint"] = (
                                    head
                                    + " Spotify's error text (spotify_api_message) suggests OAuth scope or token — "
                                    "this JSON has reauth_may_resolve: true; Sign out → Connect in this app may help. "
                                    "Dashboard: https://developer.spotify.com/dashboard"
                                )
                        else:
                            err["hint"] = (
                                head
                                + " This JSON has no reauth_may_resolve — fix playlist_id and tracks before "
                                "suggesting sign-out. Dashboard: https://developer.spotify.com/dashboard"
                            )
                    else:
                        err["hint"] = (
                            "403 on this playlist call: most often the playlist_id is not writable for this user "
                            "(not from spotify_create_playlist / spotify_user_playlists, or someone else's playlist). "
                            "Retry with the id returned by spotify_create_playlist or listed in spotify_user_playlists. "
                            "Only if the id is definitely the user's own playlist, treat as OAuth scopes or stale token: "
                            "Sign out → Connect Spotify again in this app. Dashboard: https://developer.spotify.com/dashboard"
                        )
                elif name in ("spotify_playlist_tracks", "spotify_get_playlist"):
                    if err.get("read_403_ambiguous"):
                        err["hint"] = (
                            "403 reading playlist (read_403_ambiguous): Spotify returned a generic Forbidden. "
                            "Do not blame 'collaborative' or lead with Sign out. Check spotify_get_playlist.owner.id vs "
                            "spotify_me.id — followed playlists you do not own often cannot be read track-by-track via "
                            "this API; use spotify_start_resume_playback with context_uri instead. If you own it, "
                            "re-verify id from spotify_user_playlists (22-char mix-ups), then Sign out → Connect only "
                            "if owner matches and it still fails (playlist-read-collaborative / read-private). "
                            "Dashboard: https://developer.spotify.com/dashboard"
                        )
                    else:
                        err["hint"] = (
                            "403 reading playlist: Spotify's error text suggests scope or token — Sign out → Connect "
                            "in this app may help; quote spotify_api_message. Also verify playlist_id from "
                            "spotify_user_playlists. Dashboard: https://developer.spotify.com/dashboard"
                        )
                else:
                    err["hint"] = (
                        "403: missing scopes, wrong resource, or not allowed. Check ownership and required scopes; "
                        "if scopes may be missing, Sign out → Connect Spotify again. "
                        "Dashboard: https://developer.spotify.com/dashboard"
                    )
            elif e.response.status_code == 401:
                err["hint"] = "401: sign in again (Connect Spotify) or refresh may have failed."
            if (
                name == "spotify_add_tracks_to_playlist"
                and e.response.status_code == 404
                and "hint" not in err
            ):
                err["hint"] = (
                    "404: no playlist with this id for the current user — use the exact `id` from "
                    "spotify_create_playlist or an id from spotify_user_playlists."
                )
            if e.response.status_code not in (401, 403):
                err["reconnect_spotify_unnecessary"] = True
            if name == "spotify_add_tracks_to_playlist":
                err["assistant_guidance"] = _ADD_TRACKS_ASSISTANT_GUIDANCE
                err["do_not_claim_ownership_issue"] = True
                if e.response.status_code == 401 or err.get("reauth_may_resolve"):
                    err["suggest_sign_out_of_spotify"] = True
                    err["sign_out_not_recommended"] = False
                else:
                    err["suggest_sign_out_of_spotify"] = False
                    err["sign_out_not_recommended"] = True
            elif name in ("spotify_playlist_tracks", "spotify_get_playlist") and e.response.status_code == 403:
                if err.get("read_403_ambiguous"):
                    err["suggest_sign_out_of_spotify"] = False
                    err["sign_out_not_recommended"] = True
                elif err.get("reauth_may_resolve"):
                    err["suggest_sign_out_of_spotify"] = True
                    err["sign_out_not_recommended"] = False
            playlist_id_len = 0
            n_track_inputs = 0
            arg_keys: list[str] = []
            if isinstance(arguments, dict):
                arg_keys = sorted(str(k) for k in arguments.keys())
                if name in _PLAYLIST_ID_ARG_TOOLS:
                    pl = _normalize_spotify_id(
                        _pick_arg(arguments, "playlist_id", "playlistId", "id"), "playlist"
                    )
                    playlist_id_len = len(pl)
                if name == "spotify_add_tracks_to_playlist":
                    n_track_inputs = len(_combined_track_inputs(arguments))
            if name in ("spotify_playlist_tracks", "spotify_get_playlist"):
                err["do_not_generalize_to_all_playlists"] = True
                err["assistant_guidance_playlist_read"] = _PLAYLIST_READ_ASSISTANT_GUIDANCE
            # Attach scope proof: what Spotify actually granted vs. what this tool requires.
            # Distinguishes stale-consent (scope truly missing) from scope-is-fine (other cause).
            required_any_of = _TOOL_REQUIRED_SCOPES.get(name)
            if e.response.status_code in (401, 403) and required_any_of is not None:
                try:
                    granted = self.client.get_token_scopes()
                except Exception:
                    granted = set()
                missing = _missing_any_of(granted, required_any_of)
                err["granted_scopes"] = sorted(granted)
                err["required_any_of_scopes"] = list(required_any_of)
                err["missing_scopes"] = missing
                if missing:
                    err["stale_scopes_need_reauth"] = True
                    err["suggest_sign_out_of_spotify"] = True
                    err["sign_out_not_recommended"] = False
                    err["reauth_may_resolve"] = True
                elif e.response.status_code == 403 and not _spotify_error_suggests_reauth_or_scope(
                    detail, spot_msg
                ):
                    err["scopes_appear_sufficient"] = True
                    # If scopes are provably fine, an ambiguous-wording-based reauth heuristic is
                    # misleading — clear it so the LLM does not get contradictory advice.
                    err.pop("reauth_may_resolve", None)
                    err.pop("reauth_heuristic_ambiguous_403", None)
                    err["suggest_sign_out_of_spotify"] = False
                    err["sign_out_not_recommended"] = True
                    # Spotify's Feb-2026 migration renamed several write endpoints (/tracks -> /items)
                    # and removed others (e.g. /artists/{id}/top-tracks). A 403 on an otherwise-valid
                    # call with correct scopes and ownership usually means we called a removed/renamed
                    # endpoint. Note it for the LLM rather than defaulting to sign-out.
                    err["spotify_feb_2026_migration_possible"] = True
                    err.setdefault(
                        "hint",
                        "",
                    )
                    migration_hint = (
                        " Spotify's Feb-2026 Web API migration removed/renamed endpoints for "
                        "dev-mode apps (e.g. /playlists/{id}/tracks -> /playlists/{id}/items, "
                        "GET /artists/{id}/top-tracks removed). If the backend has not been updated "
                        "for this tool, that is the likely cause — not auth. Do NOT tell the user to "
                        "sign out; ask them to retry so we can log the exact failing path."
                    )
                    if migration_hint.strip() not in err["hint"]:
                        err["hint"] = (err["hint"] + migration_hint).strip()
            logger.warning(
                "spotify_tool_http_error tool=%s http_status=%s reauth_may_resolve=%s "
                "reauth_heuristic_ambiguous_403=%s read_403_ambiguous=%s stale_scopes_need_reauth=%s "
                "granted_scopes=%s missing_scopes=%s spotify_api_message=%r arg_keys=%s "
                "playlist_id_len=%s n_track_inputs=%s",
                name,
                e.response.status_code,
                err.get("reauth_may_resolve"),
                err.get("reauth_heuristic_ambiguous_403"),
                err.get("read_403_ambiguous"),
                err.get("stale_scopes_need_reauth"),
                err.get("granted_scopes"),
                err.get("missing_scopes"),
                err.get("spotify_api_message"),
                arg_keys,
                playlist_id_len,
                n_track_inputs,
            )
            return json.dumps(err)
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    def _dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        match name:
            case "spotify_search":
                return self._search(arguments)
            case "spotify_me":
                return self._me()
            case "spotify_user_playlists":
                return self._user_playlists(arguments)
            case "spotify_playlist_tracks":
                return self._playlist_tracks(arguments)
            case "spotify_get_album":
                return self._get_album(arguments)
            case "spotify_get_track":
                return self._get_track(arguments)
            case "spotify_artist_albums":
                return self._artist_albums(arguments)
            case "spotify_get_artist":
                return self._get_artist(arguments)
            case "spotify_artist_top_tracks":
                return self._artist_top_tracks(arguments)
            case "spotify_get_playlist":
                return self._get_playlist(arguments)
            case "spotify_update_playlist":
                return self._update_playlist(arguments)
            case "spotify_remove_playlist_tracks":
                return self._remove_playlist_tracks(arguments)
            case "spotify_reorder_playlist_tracks":
                return self._reorder_playlist_tracks(arguments)
            case "spotify_replace_playlist_tracks":
                return self._replace_playlist_tracks(arguments)
            case "spotify_unfollow_playlist":
                return self._unfollow_playlist(arguments)
            case "spotify_user_saved_tracks":
                return self._user_saved_tracks(arguments)
            case "spotify_create_playlist":
                return self._create_playlist(arguments)
            case "spotify_add_tracks_to_playlist":
                return self._add_tracks(arguments)
            case "spotify_add_tracks_by_query":
                return self._add_tracks_by_query(arguments)
            case "spotify_play_playlist":
                return self._play_playlist(arguments)
            case "spotify_devices":
                return _compact(self.client.api_get("/me/player/devices"))
            case "spotify_playback_state":
                return _compact(self.client.api_get("/me/player"))
            case "spotify_transfer_playback":
                return self._transfer(arguments)
            case "spotify_start_resume_playback":
                return self._start_playback(arguments)
            case "spotify_pause":
                self.client.api_put("/me/player/pause")
                return json.dumps({"ok": True})
            case "spotify_skip_next":
                self.client.api_post("/me/player/next")
                return json.dumps({"ok": True})
            case "spotify_skip_previous":
                self.client.api_post("/me/player/previous")
                return json.dumps({"ok": True})
            case "spotify_add_to_queue":
                return self._add_to_queue(arguments)
            case "spotify_play_next":
                return self._add_to_queue(arguments)
            case "spotify_set_repeat":
                return self._set_repeat(arguments)
            case "spotify_set_shuffle":
                return self._set_shuffle(arguments)
            case "spotify_seek":
                return self._seek(arguments)
            case "spotify_set_volume":
                return self._set_volume(arguments)
            case _:
                return json.dumps({"error": f"Unknown tool: {name}"})

    def _first_artist_id_from_search(self, query: str, market: str) -> str | None:
        q = query.strip()
        if not q:
            return None
        data = self.client.api_get(
            "/search",
            params={"q": q, "type": "artist", "limit": 10, "market": market},
        )
        if not isinstance(data, dict):
            return None
        artists = data.get("artists")
        if not isinstance(artists, dict):
            return None
        items = artists.get("items")
        if not isinstance(items, list):
            return None
        for it in items:
            if isinstance(it, dict):
                aid = it.get("id")
                if isinstance(aid, str) and _looks_like_spotify_catalog_id(aid):
                    return aid
        return None

    def _canonical_artist_id(self, normalized_artist: str, market: str) -> str | None:
        if not normalized_artist:
            return None
        if _looks_like_spotify_catalog_id(normalized_artist):
            return normalized_artist
        return self._first_artist_id_from_search(normalized_artist, market)

    def _me(self) -> str:
        data = self.client.api_get("/me")
        granted = sorted(self.client.get_token_scopes())
        missing_modify = _missing_any_of(set(granted), _MODIFY_PLAYLIST_SCOPES)
        missing_read = _missing_any_of(set(granted), _READ_PLAYLIST_SCOPES)
        if isinstance(data, dict):
            data["granted_scopes"] = granted
            data["missing_playlist_modify_scopes"] = missing_modify
            data["missing_playlist_read_scopes"] = missing_read
            data["stale_scopes_need_reauth"] = bool(missing_modify or missing_read)
        return _compact(data)

    def _search(self, arguments: dict[str, Any]) -> str:
        q = _pick_arg(arguments, "query", "q", "search_query")
        types = _coerce_str(arguments.get("types"), "track,artist,album").replace(" ", "")
        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        # Feb-2026 migration: /search `limit` max dropped from 50 to 10 for dev-mode apps; default 5.
        limit = _safe_int(arguments.get("limit"), 5, lo=1, hi=10)
        offset = _safe_int(arguments.get("offset"), 0, lo=0, hi=950)
        if not q:
            return json.dumps({"error": "query is required"})
        data = self.client.api_get(
            "/search",
            params={"q": q, "type": types, "market": market, "limit": limit, "offset": offset},
        )
        return _compact(data)

    def _user_playlists(self, arguments: dict[str, Any]) -> str:
        limit = _safe_int(arguments.get("limit"), 20, lo=1, hi=50)
        offset = _safe_int(arguments.get("offset"), 0, lo=0, hi=900_000)
        data = self.client.api_get("/me/playlists", params={"limit": limit, "offset": offset})
        if isinstance(data, dict):
            data = _shrink_user_playlists_payload(data)
        return _compact(data, limit=4500)

    def _playlist_tracks(self, arguments: dict[str, Any]) -> str:
        pid = _normalize_spotify_id(
            _pick_arg(arguments, "playlist_id", "playlistId", "id"),
            "playlist",
        )
        if not pid:
            return json.dumps({"error": "playlist_id is required"})
        snap = self._playlist_owner_snapshot(pid)
        granted = set(snap.get("granted_scopes") or [])
        missing_read = _missing_any_of(granted, _READ_PLAYLIST_SCOPES)
        if snap.get("me_status") == 200 and missing_read:
            logger.info(
                "spotify_playlist_tracks_stale_scopes granted=%s missing=%s me_id=%s",
                sorted(granted),
                missing_read,
                snap.get("me_id"),
            )
            return json.dumps(
                {
                    "error": (
                        "Cannot list tracks: the signed-in token was not granted any playlist-read scope. "
                        "Your consent predates this app's required scopes; refresh tokens cannot upgrade — sign out and reconnect."
                    ),
                    "hint": "Sign out → Connect Spotify to re-consent. missing_scopes lists what is needed.",
                    "granted_scopes": sorted(granted),
                    "missing_scopes": missing_read,
                    "stale_scopes_need_reauth": True,
                    "suggest_sign_out_of_spotify": True,
                    "sign_out_not_recommended": False,
                    "reauth_may_resolve": True,
                    "assistant_guidance_playlist_read": _PLAYLIST_READ_ASSISTANT_GUIDANCE,
                },
                ensure_ascii=False,
            )
        if snap.get("is_owned") is False:
            logger.info(
                "spotify_playlist_tracks_blocked_not_owner playlist_name=%r owner_id=%s me_id=%s",
                snap.get("playlist_name"),
                snap.get("owner_id"),
                snap.get("me_id"),
            )
            return json.dumps(
                {
                    "error": (
                        "Cannot list tracks: this playlist is not owned by the signed-in user — "
                        "followed playlists may return 403 for playlist_tracks even when playback works"
                    ),
                    "hint": (
                        "Compare spotify_get_playlist.owner.id to spotify_me.id. For playlists you only follow, use "
                        "spotify_start_resume_playback with context_uri spotify:playlist:<id> without listing tracks. "
                        "Do not tell the user to sign out first for this case."
                    ),
                    "playlist_not_owned_by_user": True,
                    "playlist_owner_id": snap["owner_id"],
                    "current_user_id": snap["me_id"],
                    "playlist_name": snap.get("playlist_name"),
                    "suggest_sign_out_of_spotify": False,
                    "sign_out_not_recommended": True,
                    "reconnect_spotify_unnecessary": True,
                    "assistant_guidance_playlist_read": _PLAYLIST_READ_ASSISTANT_GUIDANCE,
                    "do_not_generalize_to_all_playlists": True,
                },
                ensure_ascii=False,
            )
        limit = _safe_int(arguments.get("limit"), 50, lo=1, hi=100)
        offset = _safe_int(arguments.get("offset"), 0, lo=0, hi=900_000)
        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        params: dict[str, Any] = {"limit": limit, "offset": offset, "market": market}
        # Spotify Feb-2026 migration: `/tracks` was renamed to `/items` for this endpoint.
        data = self.client.api_get(f"/playlists/{pid}/items", params=params)
        if isinstance(data, dict):
            data = _shrink_playlist_tracks_items(data)
        return _compact(data, limit=8000)

    def _get_album(self, arguments: dict[str, Any]) -> str:
        aid = _normalize_spotify_id(_pick_arg(arguments, "album_id", "albumId", "id"), "album")
        if not aid:
            return json.dumps({"error": "album_id is required"})
        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        return _compact(self.client.api_get(f"/albums/{aid}", params={"market": market}))

    def _get_track(self, arguments: dict[str, Any]) -> str:
        tid = _normalize_spotify_id(_pick_arg(arguments, "track_id", "trackId", "id"), "track")
        if not tid:
            return json.dumps({"error": "track_id is required"})
        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        return _compact(self.client.api_get(f"/tracks/{tid}", params={"market": market}))

    def _artist_albums(self, arguments: dict[str, Any]) -> str:
        raw_id = _pick_arg(arguments, "artist_id", "artistId", "id")
        artist_id = _normalize_spotify_id(raw_id, "artist")
        if not artist_id:
            return json.dumps({"error": "artist_id is required"})
        include_groups = _normalize_include_groups(_coerce_str(arguments.get("include_groups"), "album,single"))
        limit = _safe_int(arguments.get("limit"), 50, lo=1, hi=50)
        offset = _safe_int(arguments.get("offset"), 0, lo=0, hi=900_000)
        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        canonical_id = self._canonical_artist_id(artist_id, market)
        if not canonical_id:
            return json.dumps(
                {
                    "error": "Could not resolve artist_id to a Spotify catalog id",
                    "hint": "Use artists.items[0].id from spotify_search, or pass a recognizable artist name.",
                    "query_tried": artist_id,
                }
            )
        return _compact(
            self.client.api_get(
                f"/artists/{canonical_id}/albums",
                params={
                    "include_groups": include_groups,
                    "limit": limit,
                    "offset": offset,
                    "market": market,
                },
            )
        )

    def _get_artist(self, arguments: dict[str, Any]) -> str:
        raw = _pick_arg(arguments, "artist_id", "artistId", "id")
        norm = _normalize_spotify_id(raw, "artist")
        if not norm:
            return json.dumps({"error": "artist_id is required"})
        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        cid = self._canonical_artist_id(norm, market)
        if not cid:
            return json.dumps(
                {
                    "error": "Could not resolve artist_id",
                    "hint": "Pass a catalog id or artist name.",
                    "query_tried": norm,
                }
            )
        return _compact(self.client.api_get(f"/artists/{cid}"))

    def _artist_top_tracks(self, arguments: dict[str, Any]) -> str:
        raw = _pick_arg(arguments, "artist_id", "artistId", "id")
        norm = _normalize_spotify_id(raw, "artist")
        if not norm:
            return json.dumps({"error": "artist_id is required"})
        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        cid = self._canonical_artist_id(norm, market)
        if not cid:
            return json.dumps({"error": "Could not resolve artist_id", "query_tried": norm})
        try:
            return _compact(
                self.client.api_get(f"/artists/{cid}/top-tracks", params={"market": market})
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in (403, 404):
                raise
            # Spotify removed GET /artists/{id}/top-tracks for dev-mode apps in Feb-2026.
            # Fall back to a search by artist name — Spotify's relevance sort surfaces popular tracks.
            artist_name = ""
            try:
                art = self.client.api_get(f"/artists/{cid}")
                if isinstance(art, dict) and isinstance(art.get("name"), str):
                    artist_name = art["name"]
            except httpx.HTTPStatusError:
                pass
            query = f'artist:"{artist_name}"' if artist_name else norm
            search = self.client.api_get(
                "/search",
                params={"q": query, "type": "track", "market": market, "limit": 10},
            )
            tracks_obj = search.get("tracks") if isinstance(search, dict) else None
            items = tracks_obj.get("items") if isinstance(tracks_obj, dict) else None
            if not isinstance(items, list):
                items = []
            return json.dumps(
                {
                    "tracks": items,
                    "note": (
                        "Spotify removed GET /artists/{id}/top-tracks in the Feb-2026 Web API migration for "
                        "development-mode apps; these are the top search results for the artist as a fallback."
                    ),
                    "fallback_used": "search_by_artist_name",
                    "artist_id": cid,
                    "artist_name": artist_name,
                },
                ensure_ascii=False,
            )

    def _get_playlist(self, arguments: dict[str, Any]) -> str:
        pid = _normalize_spotify_id(
            _pick_arg(arguments, "playlist_id", "playlistId", "id"),
            "playlist",
        )
        if not pid:
            return json.dumps({"error": "playlist_id is required"})
        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        # Feb-2026: playlist object `tracks` field renamed to `items` and row `track` -> `item`.
        # Request both names so we work with either response shape.
        fields_new = (
            "collaborative,description,name,public,id,snapshot_id,"
            "owner(display_name,id),items(total,offset,next,limit,"
            "items(added_at,item(name,id,uri,duration_ms,artists(name))))"
        )
        try:
            data = self.client.api_get(
                f"/playlists/{pid}",
                params={"fields": fields_new, "market": market},
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 400:
                raise
            # Some Spotify tiers still use the legacy `tracks` name; retry with old fields spec.
            fields_legacy = (
                "collaborative,description,name,public,id,snapshot_id,"
                "owner(display_name,id),tracks(total,offset,next,limit,"
                "items(added_at,track(name,id,uri,duration_ms,artists(name))))"
            )
            data = self.client.api_get(
                f"/playlists/{pid}",
                params={"fields": fields_legacy, "market": market},
            )
        if isinstance(data, dict):
            data = _shrink_playlist_object(data)
        return _compact(data, limit=8000)

    def _update_playlist(self, arguments: dict[str, Any]) -> str:
        pid = _normalize_spotify_id(
            _pick_arg(arguments, "playlist_id", "playlistId", "id"),
            "playlist",
        )
        if not pid:
            return json.dumps({"error": "playlist_id is required"})
        body: dict[str, Any] = {}
        if "name" in arguments and arguments.get("name") is not None:
            n = str(arguments.get("name", "")).strip()
            if n:
                body["name"] = n
        if "description" in arguments and arguments.get("description") is not None:
            body["description"] = str(arguments.get("description", ""))
        if "public" in arguments and arguments.get("public") is not None:
            body["public"] = bool(arguments.get("public"))
        if "collaborative" in arguments and arguments.get("collaborative") is not None:
            body["collaborative"] = bool(arguments.get("collaborative"))
        if not body:
            return json.dumps(
                {"error": "Provide at least one of: name, description, public, collaborative (non-null)."}
            )
        self.client.api_put(f"/playlists/{pid}", json_body=body)
        return json.dumps({"ok": True, "updated_fields": list(body.keys())})

    def _remove_playlist_tracks(self, arguments: dict[str, Any]) -> str:
        pid = _normalize_spotify_id(
            _pick_arg(arguments, "playlist_id", "playlistId", "id"),
            "playlist",
        )
        if not pid:
            return json.dumps({"error": "playlist_id is required"})
        uri_list = _coerce_track_uri_list(_combined_track_inputs(arguments))
        if not uri_list:
            return json.dumps({"error": "track_uris / track_ids / tracks (non-empty list) is required"})
        chunk = uri_list[:100]
        # Feb-2026 Spotify rename: body param `tracks` -> `items`, path `/tracks` -> `/items`.
        body: dict[str, Any] = {"items": [{"uri": u} for u in chunk]}
        snap = _coerce_str(arguments.get("snapshot_id"), "")
        if snap:
            body["snapshot_id"] = snap
        data = self.client.api_delete(f"/playlists/{pid}/items", json_body=body)
        return _compact(data if data is not None else {"ok": True})

    def _reorder_playlist_tracks(self, arguments: dict[str, Any]) -> str:
        pid = _normalize_spotify_id(
            _pick_arg(arguments, "playlist_id", "playlistId", "id"),
            "playlist",
        )
        if not pid:
            return json.dumps({"error": "playlist_id is required"})
        if "insert_before" not in arguments or arguments.get("insert_before") is None:
            return json.dumps({"error": "insert_before is required (0-based index in the playlist)"})
        insert_before = _safe_int(arguments.get("insert_before"), 0, lo=0, hi=10_000)
        range_start = _safe_int(arguments.get("range_start"), 0, lo=0, hi=10_000)
        range_length = _safe_int(arguments.get("range_length"), 1, lo=1, hi=100)
        params: dict[str, Any] = {
            "range_start": range_start,
            "insert_before": insert_before,
            "range_length": range_length,
        }
        snap = _coerce_str(arguments.get("snapshot_id"), "")
        if snap:
            params["snapshot_id"] = snap
        data = self.client.api_put(f"/playlists/{pid}/items", json_body=None, params=params)
        return _compact(data if data is not None else {"ok": True})

    def _replace_playlist_tracks(self, arguments: dict[str, Any]) -> str:
        pid = _normalize_spotify_id(
            _pick_arg(arguments, "playlist_id", "playlistId", "id"),
            "playlist",
        )
        if not pid:
            return json.dumps({"error": "playlist_id is required"})
        uri_list = _coerce_track_uri_list(_combined_track_inputs(arguments))
        if not uri_list:
            return json.dumps(
                {
                    "error": "track_uris / track_ids / tracks must be a non-empty list (max 100 URIs per call; repeat to replace more).",
                }
            )
        chunk = uri_list[:100]
        valid_uris, invalid_uris = _verify_tracks_exist(self.client, chunk)
        if invalid_uris:
            logger.warning(
                "spotify_replace_playlist_tracks_rejected_non_track_uris pid=%s invalid=%s",
                pid,
                invalid_uris,
            )
        if not valid_uris:
            return json.dumps(
                {
                    "error": "None of the supplied ids resolve to real Spotify tracks.",
                    "hint": "Only use track ids from spotify_search results (tracks.items[i].id/uri).",
                    "rejected_uris": invalid_uris,
                    "reconnect_spotify_unnecessary": True,
                    "sign_out_not_recommended": True,
                }
            )
        data = self.client.api_put(f"/playlists/{pid}/items", json_body={"uris": valid_uris})
        result: dict[str, Any] = {"ok": True, "replaced_count": len(valid_uris)}
        if isinstance(data, dict) and isinstance(data.get("snapshot_id"), str):
            result["snapshot_id"] = data["snapshot_id"]
        if invalid_uris:
            result["skipped_count"] = len(invalid_uris)
            result["skipped_uris"] = invalid_uris
        return _compact(result)

    def _unfollow_playlist(self, arguments: dict[str, Any]) -> str:
        pid = _normalize_spotify_id(
            _pick_arg(arguments, "playlist_id", "playlistId", "id"),
            "playlist",
        )
        if not pid:
            return json.dumps({"error": "playlist_id is required"})
        self.client.api_delete(f"/playlists/{pid}/followers")
        return json.dumps({"ok": True, "playlist_id": pid})

    def _user_saved_tracks(self, arguments: dict[str, Any]) -> str:
        limit = _safe_int(arguments.get("limit"), 50, lo=1, hi=50)
        offset = _safe_int(arguments.get("offset"), 0, lo=0, hi=900_000)
        data = self.client.api_get("/me/tracks", params={"limit": limit, "offset": offset})
        if isinstance(data, dict):
            data = _shrink_saved_tracks_page(data)
        return _compact(data, limit=8000)

    def _create_playlist(self, arguments: dict[str, Any]) -> str:
        name = str(arguments.get("name", "")).strip()
        if not name:
            return json.dumps({"error": "name is required"})
        public = bool(arguments.get("public", True))
        collaborative = bool(arguments.get("collaborative", False))
        if collaborative:
            public = False
        description = str(arguments.get("description", ""))
        body: dict[str, Any] = {"name": name, "public": public, "description": description}
        if collaborative:
            body["collaborative"] = True
        data = self.client.api_post("/me/playlists", json_body=body)
        if not isinstance(data, dict):
            return _compact(data)
        pid = data.get("id")
        mini: dict[str, Any] = {
            "id": pid,
            "name": data.get("name"),
            "uri": data.get("uri"),
            "playlist_id_for_add_tracks": pid,
            "hint": "Next: spotify_add_tracks_to_playlist with playlist_id = id above (string), plus track_uris / track_ids / tracks from spotify_search.",
        }
        if isinstance(data.get("snapshot_id"), str):
            mini["snapshot_id"] = data["snapshot_id"]
        return json.dumps(mini, ensure_ascii=False)

    def _add_tracks(self, arguments: dict[str, Any]) -> str:
        pid = _normalize_spotify_id(
            _pick_arg(arguments, "playlist_id", "playlistId", "id"),
            "playlist",
        )
        combined = _combined_track_inputs(arguments)
        uri_list = _coerce_track_uri_list(combined) if combined else None
        if combined and not uri_list:
            logger.info(
                "spotify_add_tracks_validation_failed reason=no_valid_track_uris keys=%s playlist_id_len=%s n_combined=%s",
                sorted(arguments.keys()),
                len(pid or ""),
                len(combined),
            )
            return json.dumps(
                {
                    "error": "tracks payload had entries but none became valid spotify:track URIs",
                    "hint": "Use track objects with uri or id (22-char catalog id), or strings spotify:track:… / bare ids. "
                    "You may pass tracks as the search object {items: [...]} or a flat list of track objects.",
                    "keys_seen": sorted(str(k) for k in arguments.keys()),
                    "reconnect_spotify_unnecessary": True,
                    "suggest_sign_out_of_spotify": False,
                    "sign_out_not_recommended": True,
                    "assistant_guidance": _ADD_TRACKS_ASSISTANT_GUIDANCE,
                    "do_not_claim_ownership_issue": True,
                }
            )
        if not pid or not uri_list:
            logger.info(
                "spotify_add_tracks_validation_failed reason=missing_playlist_id_or_tracks keys=%s "
                "playlist_id_len=%s n_combined=%s has_uri_list=%s",
                sorted(arguments.keys()),
                len(pid or ""),
                len(combined),
                bool(uri_list),
            )
            return json.dumps(
                {
                    "error": "playlist_id and a non-empty list of tracks are required",
                    "hint": "Use playlist_id from spotify_create_playlist (id or playlist_id_for_add_tracks) or spotify_user_playlists. "
                    "Pass track_uris, track_ids, or tracks — tracks may be an array of search track objects OR {items: [...]} from spotify_search. "
                    "The same {items: [...]} shape may be placed under track_uris or uris if the model used the wrong key.",
                    "keys_seen": sorted(str(k) for k in arguments.keys()),
                    "reconnect_spotify_unnecessary": True,
                    "suggest_sign_out_of_spotify": False,
                    "sign_out_not_recommended": True,
                    "assistant_guidance": _ADD_TRACKS_ASSISTANT_GUIDANCE,
                    "do_not_claim_ownership_issue": True,
                }
            )
        snap = self._playlist_owner_snapshot(pid)
        granted = set(snap.get("granted_scopes") or [])
        missing_modify = _missing_any_of(granted, _MODIFY_PLAYLIST_SCOPES)
        if snap.get("me_status") == 200 and missing_modify:
            # Definitive: token was issued without any playlist-modify scope. Refresh won't add it — must re-consent.
            logger.info(
                "spotify_add_tracks_stale_scopes granted=%s missing=%s me_id=%s",
                sorted(granted),
                missing_modify,
                snap.get("me_id"),
            )
            return json.dumps(
                {
                    "error": (
                        "Cannot add tracks: the signed-in token was not granted any playlist-modify scope. "
                        "Your consent was issued before this app required those scopes; refresh tokens cannot "
                        "upgrade scopes — you must sign out and connect again."
                    ),
                    "hint": (
                        "Sign out → Connect Spotify in this app to re-consent with the current scope set. "
                        "Missing scopes are listed under missing_scopes."
                    ),
                    "granted_scopes": sorted(granted),
                    "missing_scopes": missing_modify,
                    "stale_scopes_need_reauth": True,
                    "suggest_sign_out_of_spotify": True,
                    "sign_out_not_recommended": False,
                    "reauth_may_resolve": True,
                    "assistant_guidance": _ADD_TRACKS_ASSISTANT_GUIDANCE,
                },
                ensure_ascii=False,
            )
        if snap.get("is_owned") is False:
            logger.info(
                "spotify_add_tracks_blocked_not_owner playlist_name=%r owner_id=%s me_id=%s",
                snap.get("playlist_name"),
                snap.get("owner_id"),
                snap.get("me_id"),
            )
            return json.dumps(
                {
                    "error": (
                        "Cannot add tracks: this playlist is not owned by the signed-in user "
                        "(Spotify Web API only allows the owner to add tracks)"
                    ),
                    "hint": (
                        "spotify_user_playlists returns playlists you own and playlists you follow. "
                        "Call spotify_get_playlist with this id and compare owner.id to spotify_me.id. "
                        "If they differ, use spotify_create_playlist to make your own copy, or add only to "
                        "playlists you own."
                    ),
                    "playlist_not_owned_by_user": True,
                    "playlist_owner_id": snap["owner_id"],
                    "current_user_id": snap["me_id"],
                    "playlist_name": snap.get("playlist_name"),
                    "suggest_sign_out_of_spotify": False,
                    "sign_out_not_recommended": True,
                    "reconnect_spotify_unnecessary": True,
                    "assistant_guidance": _ADD_TRACKS_ASSISTANT_GUIDANCE,
                    "do_not_claim_ownership_issue": True,
                },
                ensure_ascii=False,
            )
        requested = uri_list[:100]
        valid_uris, invalid_uris = _verify_tracks_exist(self.client, requested)
        if invalid_uris:
            logger.warning(
                "spotify_add_tracks_rejected_non_track_uris pid=%s invalid=%s valid_count=%d",
                pid,
                invalid_uris,
                len(valid_uris),
            )
        if not valid_uris:
            return json.dumps(
                {
                    "error": (
                        "None of the supplied ids resolve to real Spotify tracks. Spotify would accept "
                        "them as empty (ghost) rows, so the add was blocked."
                    ),
                    "hint": (
                        "Do NOT retry with different invented ids. Either (a) call spotify_search first "
                        "and pass uri/id from tracks.items where type=='track', or (b) use the composite "
                        "tool spotify_add_tracks_by_query with {playlist_id, query, count, min_year?} — "
                        "it searches and adds real URIs for you."
                    ),
                    "try_instead": "spotify_add_tracks_by_query",
                    "rejected_uris": invalid_uris,
                    "requested_count": len(requested),
                    "added_count": 0,
                    "reconnect_spotify_unnecessary": True,
                    "suggest_sign_out_of_spotify": False,
                    "sign_out_not_recommended": True,
                    "assistant_guidance": _ADD_TRACKS_ASSISTANT_GUIDANCE,
                    "do_not_claim_ownership_issue": True,
                    "do_not_claim_success_without_added_count": True,
                },
                ensure_ascii=False,
            )
        snapshot = self.client.api_post(
            f"/playlists/{pid}/items",
            json_body={"uris": valid_uris},
        )
        result: dict[str, Any] = {
            "ok": True,
            "playlist_id": pid,
            "requested_count": len(requested),
            "added_count": len(valid_uris),
            "added_uris": valid_uris,
        }
        if isinstance(snapshot, dict) and isinstance(snapshot.get("snapshot_id"), str):
            result["snapshot_id"] = snapshot["snapshot_id"]
        if invalid_uris:
            result["skipped_count"] = len(invalid_uris)
            result["skipped_uris"] = invalid_uris
            result["skipped_reason"] = (
                "ids did not resolve to spotify tracks (likely artist/album ids or bad ids) — not added "
                "to avoid creating ghost rows."
            )
        result["verify_hint"] = (
            "Call spotify_playlist_tracks with this playlist_id now and report the tracks actually present "
            "(name/uri) — do not claim adds the user cannot see."
        )
        return _compact(result)

    def _add_tracks_by_query(self, arguments: dict[str, Any]) -> str:
        """Composite tool: search → (optional) year filter → dedupe vs playlist → add real URIs.

        Prefer this over spotify_search + spotify_add_tracks_to_playlist when the user says
        "add a/some <artist|song> to <playlist>". It guarantees only real Spotify track ids
        are added and skips duplicates already on the playlist.
        """
        pid = _normalize_spotify_id(
            _pick_arg(arguments, "playlist_id", "playlistId", "id"),
            "playlist",
        )
        query = _coerce_str(_pick_arg(arguments, "query", "q", "search_query"))
        if not pid or not query:
            return json.dumps(
                {
                    "error": "playlist_id and query are required",
                    "hint": (
                        "Call with playlist_id (owned playlist) and a natural-language query such as "
                        "'SZA 2024' or 'john mayer gravity'. Optional: count (default 1, max 10), "
                        "min_year (e.g. 2024), market (defaults to from_token), avoid_duplicates (default true)."
                    ),
                    "reconnect_spotify_unnecessary": True,
                    "sign_out_not_recommended": True,
                }
            )
        count = _safe_int(arguments.get("count"), 1, lo=1, hi=10)
        min_year_raw = arguments.get("min_year") or arguments.get("year_min") or arguments.get("min_release_year")
        min_year = _safe_int(min_year_raw, 0, lo=0, hi=3000) if min_year_raw is not None else 0
        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        avoid_dup_raw = arguments.get("avoid_duplicates")
        avoid_duplicates = True if avoid_dup_raw is None else bool(avoid_dup_raw)

        snap = self._playlist_owner_snapshot(pid)
        granted = set(snap.get("granted_scopes") or [])
        missing_modify = _missing_any_of(granted, _MODIFY_PLAYLIST_SCOPES)
        if snap.get("me_status") == 200 and missing_modify:
            return json.dumps(
                {
                    "error": (
                        "Cannot add tracks: the signed-in token was not granted any playlist-modify scope. "
                        "Sign out → Connect Spotify to re-consent."
                    ),
                    "granted_scopes": sorted(granted),
                    "missing_scopes": missing_modify,
                    "stale_scopes_need_reauth": True,
                    "suggest_sign_out_of_spotify": True,
                    "reauth_may_resolve": True,
                    "assistant_guidance": _ADD_TRACKS_ASSISTANT_GUIDANCE,
                },
                ensure_ascii=False,
            )
        if snap.get("is_owned") is False:
            return json.dumps(
                {
                    "error": (
                        "Cannot add tracks: this playlist is not owned by the signed-in user."
                    ),
                    "playlist_not_owned_by_user": True,
                    "playlist_owner_id": snap.get("owner_id"),
                    "current_user_id": snap.get("me_id"),
                    "playlist_name": snap.get("playlist_name"),
                    "reconnect_spotify_unnecessary": True,
                    "sign_out_not_recommended": True,
                    "assistant_guidance": _ADD_TRACKS_ASSISTANT_GUIDANCE,
                    "do_not_claim_ownership_issue": True,
                },
                ensure_ascii=False,
            )

        # Search for track candidates. Pull a wider pool than needed so dedupe+year still has options.
        search = self.client.api_get(
            "/search",
            params={"q": query, "type": "track", "market": market, "limit": 10},
        )
        items_raw = []
        if isinstance(search, dict):
            tr = search.get("tracks") or {}
            if isinstance(tr, dict):
                items_raw = tr.get("items") or []
        candidates: list[dict[str, Any]] = [it for it in items_raw if isinstance(it, dict) and it.get("type") == "track" and isinstance(it.get("uri"), str)]

        def _release_year(track_obj: dict[str, Any]) -> int:
            album = track_obj.get("album") or {}
            rd = (album.get("release_date") or "")[:4] if isinstance(album, dict) else ""
            try:
                return int(rd) if rd.isdigit() else 0
            except Exception:
                return 0

        pool = candidates[:]
        year_filtered: list[dict[str, Any]] = []
        if min_year:
            year_filtered = [c for c in pool if _release_year(c) >= min_year]
            if year_filtered:
                # Prefer year-matched, but keep others as fallback if we don't find enough unique tracks.
                pool = year_filtered + [c for c in pool if c not in year_filtered]

        # Fetch the existing playlist tracks once for dedup (first 500 entries; enough for typical playlists).
        existing_uris: set[str] = set()
        if avoid_duplicates:
            try:
                offset_p = 0
                while offset_p < 500:
                    page = self.client.api_get(
                        f"/playlists/{pid}/items",
                        params={
                            "limit": 100,
                            "offset": offset_p,
                            "market": market,
                            "fields": "items(track(uri),item(uri)),next",
                        },
                    )
                    if not isinstance(page, dict):
                        break
                    rows = page.get("items") or []
                    if not isinstance(rows, list) or not rows:
                        break
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        tr = row.get("track") or row.get("item") or {}
                        u = tr.get("uri") if isinstance(tr, dict) else None
                        if isinstance(u, str) and u.startswith("spotify:track:"):
                            existing_uris.add(u)
                    if not page.get("next"):
                        break
                    offset_p += 100
            except httpx.HTTPStatusError:
                # If we can't read the playlist, don't block the add — Spotify dedupes nothing but at least we try.
                pass

        picked: list[dict[str, Any]] = []
        seen_in_batch: set[str] = set()
        for cand in pool:
            uri = cand.get("uri")
            if not isinstance(uri, str) or not uri.startswith("spotify:track:"):
                continue
            if uri in seen_in_batch:
                continue
            if avoid_duplicates and uri in existing_uris:
                continue
            picked.append(cand)
            seen_in_batch.add(uri)
            if len(picked) >= count:
                break

        if not picked:
            return json.dumps(
                {
                    "error": (
                        f"No new tracks matched query '{query}' that are not already on the playlist."
                        + (f" (min_year={min_year} filter applied)" if min_year else "")
                    ),
                    "hint": (
                        "Loosen the query, drop min_year, or set avoid_duplicates=false. You can also call "
                        "spotify_search directly for a wider look."
                    ),
                    "search_result_count": len(candidates),
                    "year_filtered_count": len(year_filtered) if min_year else None,
                    "existing_playlist_track_count_seen": len(existing_uris),
                    "query_used": query,
                    "min_year": min_year or None,
                    "reconnect_spotify_unnecessary": True,
                    "sign_out_not_recommended": True,
                },
                ensure_ascii=False,
            )

        uri_list = [p["uri"] for p in picked]
        snapshot_resp = self.client.api_post(
            f"/playlists/{pid}/items",
            json_body={"uris": uri_list},
        )

        def _artist_names(tr_obj: dict[str, Any]) -> list[str]:
            arts = tr_obj.get("artists")
            if not isinstance(arts, list):
                return []
            return [a.get("name", "") for a in arts if isinstance(a, dict) and isinstance(a.get("name"), str)]

        added_tracks = [
            {
                "name": p.get("name", ""),
                "artists": _artist_names(p),
                "uri": p.get("uri", ""),
                "id": p.get("id", ""),
                "release_year": _release_year(p),
                "album": (p.get("album") or {}).get("name", "") if isinstance(p.get("album"), dict) else "",
            }
            for p in picked
        ]

        result: dict[str, Any] = {
            "ok": True,
            "playlist_id": pid,
            "query_used": query,
            "requested_count": count,
            "added_count": len(picked),
            "added_tracks": added_tracks,
            "skipped_as_duplicates_count": max(0, len(candidates) - len(pool)) if avoid_duplicates else 0,
            "search_result_count": len(candidates),
            "min_year": min_year or None,
            "market": market,
        }
        if isinstance(snapshot_resp, dict) and isinstance(snapshot_resp.get("snapshot_id"), str):
            result["snapshot_id"] = snapshot_resp["snapshot_id"]
        result["verify_hint"] = (
            "The added_tracks list above is the authoritative source for what was just added. "
            "Report those names/artists verbatim to the user — do not substitute other song names."
        )
        return _compact(result)

    def _transfer(self, arguments: dict[str, Any]) -> str:
        did = str(arguments.get("device_id", "")).strip()
        if not did:
            return json.dumps({"error": "device_id is required"})
        self.client.api_put("/me/player", json_body={"device_ids": [did], "play": False})
        return json.dumps({"ok": True, "device_id": did})

    def _force_to_target_track(
        self,
        body: dict[str, Any],
        *,
        device_id: str = "",
        max_skips: int = 6,
    ) -> bool:
        """Salvage a queue-like state into an immediate play.

        When /me/player/play with context_uri+offset was accepted but the device treats the
        offset as "up next" (the target uri shows up in /me/player/queue rather than becoming
        the current item), or when another track is holding playback, we can issue
        POST /me/player/next a few times to advance to the target. Stops early when the
        current item URI matches the requested offset.uri / uris[0].
        """
        want_first_uri = ""
        want_offset_uri = ""
        uris = body.get("uris") if isinstance(body, dict) else None
        if isinstance(uris, list) and uris:
            w = uris[0]
            if isinstance(w, str):
                want_first_uri = w.strip()
        off = body.get("offset") if isinstance(body, dict) else None
        if isinstance(off, dict):
            ou = off.get("uri")
            if isinstance(ou, str):
                want_offset_uri = ou.strip()
        target = want_offset_uri or want_first_uri
        if not target:
            return False
        # Early-out check: already on target?
        snap = self._current_playback_snapshot()
        if snap.get("item_uri") == target:
            return True
        for _ in range(max(1, max_skips)):
            params: dict[str, str] = {}
            if device_id:
                params["device_id"] = device_id
            try:
                if params:
                    self.client.api_post("/me/player/next", params=params)
                else:
                    self.client.api_post("/me/player/next")
            except httpx.HTTPStatusError:
                return False
            time.sleep(0.6)
            snap = self._current_playback_snapshot()
            if snap.get("item_uri") == target and snap.get("is_playing"):
                return True
        return False

    def _resolve_target_device(self, preferred: str = "") -> str:
        """Pick the best device id to target.

        Preference order: explicit preferred id > /me/player device > the single
        non-restricted available device. Returns "" if nothing usable is available.
        """
        if preferred:
            return preferred
        try:
            ps = self.client.api_get("/me/player")
            if isinstance(ps, dict):
                dev = ps.get("device") or {}
                if isinstance(dev, dict) and isinstance(dev.get("id"), str) and dev.get("id"):
                    return dev["id"]
        except httpx.HTTPStatusError:
            pass
        try:
            data = self.client.api_get("/me/player/devices") or {}
        except httpx.HTTPStatusError:
            return ""
        devices = data.get("devices") if isinstance(data, dict) else []
        if not isinstance(devices, list):
            return ""
        non_restricted = [
            d for d in devices
            if isinstance(d, dict) and not d.get("is_restricted") and isinstance(d.get("id"), str)
        ]
        if len(non_restricted) == 1:
            return non_restricted[0]["id"]
        active = [d for d in non_restricted if d.get("is_active")]
        if active:
            return active[0]["id"]
        return ""

    def _start_playback(self, arguments: dict[str, Any]) -> str:
        device_id = str(arguments.get("device_id", "")).strip() or self._device_id() or ""
        body: dict[str, Any] = {}
        uris = arguments.get("uris")
        context_uri = arguments.get("context_uri")
        offset = arguments.get("offset")
        if isinstance(uris, list) and uris:
            body["uris"] = [str(u) for u in uris]
        if isinstance(context_uri, str) and context_uri.strip():
            body["context_uri"] = context_uri.strip()
        if isinstance(offset, dict):
            body["offset"] = offset
        want_verification = bool(
            body.get("context_uri")
            or body.get("uris")
            or (isinstance(body.get("offset"), dict) and body["offset"].get("uri"))
        )
        try:
            self._try_play(device_id, body)
            if not want_verification:
                return json.dumps({"ok": True, "device_id": device_id or None, "body": body})
            # Spotify frequently returns 200 while the device controller keeps playing the
            # previous track/context. Verify the current track/context actually switched —
            # if not, force a transfer to the intended device and retry once, then verify.
            if self._playback_matches(body, attempts=6, delay_s=0.5):
                return json.dumps(
                    {
                        "ok": True,
                        "device_id": device_id or None,
                        "body": body,
                        "playback_verified": True,
                    }
                )
            chosen = self._resolve_target_device(device_id)
            if chosen:
                try:
                    self.client.api_put(
                        "/me/player",
                        json_body={"device_ids": [chosen], "play": False},
                    )
                except httpx.HTTPStatusError:
                    pass
                time.sleep(0.4)
                try:
                    self._try_play(chosen, body)
                except httpx.HTTPStatusError:
                    pass
                if self._playback_matches(body, attempts=8, delay_s=0.5):
                    return json.dumps(
                        {
                            "ok": True,
                            "device_id": chosen,
                            "body": body,
                            "playback_verified": True,
                            "note": "Playback did not switch on the first attempt; forced a transfer to the target device and retried.",
                        }
                    )
            # Last resort: if Spotify treated the offset as "up next" (common when already
            # playing from the same context), skip forward until the target track becomes current.
            if self._force_to_target_track(body, device_id=chosen or device_id):
                return json.dumps(
                    {
                        "ok": True,
                        "device_id": chosen or device_id or None,
                        "body": body,
                        "playback_verified": True,
                        "note": (
                            "Spotify queued the target as 'up next' instead of jumping — forced skip(s) "
                            "to advance to the requested track."
                        ),
                    }
                )
            snapshot = self._current_playback_snapshot()
            return json.dumps(
                {
                    "ok": False,
                    "error": (
                        "Spotify accepted the play command (HTTP 200) but playback did not switch "
                        "to the requested context/track. The previous song is still playing."
                    ),
                    "hint": (
                        "A different device or session may be holding playback. Ask the user to tap "
                        "play briefly in Spotify on the intended device, or call spotify_devices + "
                        "spotify_transfer_playback to move control, then retry. Do not tell the user "
                        "playback started — it did not."
                    ),
                    "requested_body": body,
                    "current_state": snapshot,
                    "device_id": device_id or None,
                    "playback_verified": False,
                    "reconnect_spotify_unnecessary": True,
                    "sign_out_not_recommended": True,
                }
            )
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            # Spotify's edge frequently returns 502/503/504 on /me/player/play even when the
            # request reaches the player — confirm via /me/player before giving up. If playback
            # actually matches what we asked for, treat as success.
            if status in (500, 502, 503, 504):
                confirmed = self._playback_matches(body)
                if confirmed:
                    return json.dumps(
                        {
                            "ok": True,
                            "device_id": device_id or None,
                            "body": body,
                            "playback_verified": True,
                            "note": (
                                f"Spotify edge returned HTTP {status} but playback state confirms "
                                "the command was applied."
                            ),
                        }
                    )
                # Retry once after a short pause.
                time.sleep(0.8)
                try:
                    self._try_play(device_id, body)
                    if not want_verification or self._playback_matches(body, attempts=6, delay_s=0.5):
                        return json.dumps(
                            {
                                "ok": True,
                                "device_id": device_id or None,
                                "body": body,
                                "playback_verified": bool(want_verification),
                                "note": f"Succeeded on retry after Spotify HTTP {status}.",
                            }
                        )
                    # 200 came back but playback still did not switch. Try force-skipping to target.
                    if self._force_to_target_track(body, device_id=device_id):
                        return json.dumps(
                            {
                                "ok": True,
                                "device_id": device_id or None,
                                "body": body,
                                "playback_verified": True,
                                "note": (
                                    f"Spotify returned HTTP {status}, retry succeeded, target was "
                                    "queued instead of current — forced skip(s) to advance to it."
                                ),
                            }
                        )
                    snapshot = self._current_playback_snapshot()
                    return json.dumps(
                        {
                            "ok": False,
                            "error": (
                                f"Spotify returned HTTP {status} then a retry succeeded, but the "
                                "device is still playing the previous track. Playback did not switch."
                            ),
                            "hint": (
                                "Ask the user to tap play on the intended device, or use "
                                "spotify_transfer_playback to move control, then retry."
                            ),
                            "requested_body": body,
                            "current_state": snapshot,
                            "device_id": device_id or None,
                            "playback_verified": False,
                            "reconnect_spotify_unnecessary": True,
                            "sign_out_not_recommended": True,
                        }
                    )
                except httpx.HTTPStatusError as e2:
                    if e2.response.status_code in (500, 502, 503, 504):
                        confirmed = self._playback_matches(body)
                        if confirmed:
                            return json.dumps(
                                {
                                    "ok": True,
                                    "device_id": device_id or None,
                                    "body": body,
                                    "playback_verified": True,
                                    "note": (
                                        f"Spotify edge returned HTTP {e2.response.status_code} "
                                        "twice but playback state confirms the command was applied."
                                    ),
                                }
                            )
                    if e2.response.status_code == 404:
                        e = e2  # fall through to 404 handling below
                    else:
                        raise
            if e.response.status_code != 404:
                raise
            # 404 => no active device. Try to pick a single available idle device, transfer, retry.
            try:
                dev = self.client.api_get("/me/player/devices") or {}
            except httpx.HTTPStatusError:
                dev = {}
            devices = dev.get("devices") if isinstance(dev, dict) else []
            devices = devices if isinstance(devices, list) else []
            non_restricted = [d for d in devices if isinstance(d, dict) and not d.get("is_restricted")]
            chosen = None
            if device_id:
                chosen = device_id
            elif len(non_restricted) == 1 and isinstance(non_restricted[0].get("id"), str):
                chosen = non_restricted[0]["id"]
            if chosen:
                try:
                    self.client.api_put(
                        "/me/player",
                        json_body={"device_ids": [chosen], "play": False},
                    )
                except httpx.HTTPStatusError:
                    pass
                try:
                    self._try_play(chosen, body)
                    verified = (
                        True
                        if not want_verification
                        else self._playback_matches(body, attempts=6, delay_s=0.5)
                    )
                    return json.dumps(
                        {
                            "ok": True,
                            "device_id": chosen,
                            "body": body,
                            "playback_verified": bool(verified),
                            "note": "Transferred playback to the only available device before playing.",
                        }
                    )
                except httpx.HTTPStatusError:
                    pass
            return json.dumps(
                {
                    "error": (
                        "Spotify has no active device to play on. Open Spotify on a device "
                        "(desktop/mobile/web player) and press play briefly, or pick one in the "
                        "app's device selector, then retry."
                    ),
                    "hint": (
                        "If the user has multiple idle devices, call spotify_devices and "
                        "re-invoke spotify_start_resume_playback with device_id explicitly."
                    ),
                    "devices": devices,
                    "reconnect_spotify_unnecessary": True,
                    "sign_out_not_recommended": True,
                },
                ensure_ascii=False,
            )

    def _try_play(self, device_id: str, body: dict[str, Any]) -> None:
        path = "/me/player/play"
        if device_id:
            path = f"{path}?device_id={device_id}"
        self.client.api_put(path, json_body=body if body else {})

    def _playback_matches(
        self,
        body: dict[str, Any],
        *,
        attempts: int = 6,
        delay_s: float = 0.5,
    ) -> bool:
        """Poll /me/player briefly and check whether playback reflects the request we sent.

        When the caller specified offset.uri or uris[0], the CURRENT TRACK must match that
        URI — matching the context alone is not enough (Spotify may report a stale context
        while the old track keeps playing). Returns True only when is_playing AND
        (context matches if requested) AND (current track matches the requested start uri
        if requested).
        """
        want_context = (body.get("context_uri") or "").strip() if isinstance(body, dict) else ""
        want_uris = body.get("uris") if isinstance(body, dict) else None
        want_first_uri = (
            str(want_uris[0]).strip()
            if isinstance(want_uris, list) and want_uris and want_uris[0] is not None
            else ""
        )
        want_offset = body.get("offset") if isinstance(body, dict) else None
        want_offset_uri = ""
        if isinstance(want_offset, dict):
            o_uri = want_offset.get("uri")
            if isinstance(o_uri, str):
                want_offset_uri = o_uri.strip()
        want_track_uri = want_offset_uri or want_first_uri
        for _ in range(max(1, attempts)):
            try:
                ps = self.client.api_get("/me/player")
            except httpx.HTTPStatusError:
                return False
            if not isinstance(ps, dict):
                time.sleep(delay_s)
                continue
            if ps.get("is_playing"):
                ctx = (ps.get("context") or {}) if isinstance(ps.get("context"), dict) else {}
                ctx_uri = (ctx.get("uri") or "").strip()
                item = ps.get("item") or {}
                cur_uri = (item.get("uri") or "").strip() if isinstance(item, dict) else ""
                ctx_ok = True if not want_context else (ctx_uri == want_context)
                track_ok = True if not want_track_uri else (cur_uri == want_track_uri)
                if ctx_ok and track_ok:
                    return True
            time.sleep(delay_s)
        return False

    def _current_playback_snapshot(self) -> dict[str, Any]:
        """Return a small snapshot of /me/player for diagnostics (never raises)."""
        try:
            ps = self.client.api_get("/me/player")
        except httpx.HTTPStatusError:
            return {}
        if not isinstance(ps, dict):
            return {}
        ctx = ps.get("context") if isinstance(ps.get("context"), dict) else {}
        item = ps.get("item") if isinstance(ps.get("item"), dict) else {}
        device = ps.get("device") if isinstance(ps.get("device"), dict) else {}
        return {
            "is_playing": bool(ps.get("is_playing")),
            "context_uri": (ctx.get("uri") if isinstance(ctx, dict) else None),
            "item_uri": (item.get("uri") if isinstance(item, dict) else None),
            "item_name": (item.get("name") if isinstance(item, dict) else None),
            "device_id": (device.get("id") if isinstance(device, dict) else None),
            "device_name": (device.get("name") if isinstance(device, dict) else None),
            "device_is_restricted": (device.get("is_restricted") if isinstance(device, dict) else None),
        }

    def _play_playlist(self, arguments: dict[str, Any]) -> str:
        """Composite tool: start a playlist (optionally at a specific track) and set repeat/shuffle in one call.

        Prefer this over chaining spotify_start_resume_playback + spotify_set_repeat when the user
        says something like "play RNB2025 starting at Kill Bill with repeat on". It plays the
        context, then (if the play call returned ok) applies repeat and shuffle.
        """
        pid = _normalize_spotify_id(
            _pick_arg(arguments, "playlist_id", "playlistId", "id"),
            "playlist",
        )
        if not pid:
            return json.dumps(
                {
                    "error": "playlist_id is required",
                    "hint": "Pass playlist_id (the id alone, or as spotify:playlist:<id>).",
                }
            )

        start_at_uri_raw = _pick_arg(arguments, "start_at_uri", "track_uri", "offset_uri")
        start_at_uri = start_at_uri_raw.strip() if start_at_uri_raw else ""
        if start_at_uri and not start_at_uri.startswith("spotify:track:"):
            tid = _normalize_spotify_id(start_at_uri, "track")
            if _looks_like_spotify_catalog_id(tid):
                start_at_uri = f"spotify:track:{tid}"
            else:
                start_at_uri = ""

        start_at_position_raw = arguments.get("start_at_position")
        start_at_position = (
            _safe_int(start_at_position_raw, -1, lo=0, hi=10_000)
            if start_at_position_raw is not None
            else -1
        )

        context_uri = f"spotify:playlist:{pid}"
        play_args: dict[str, Any] = {"context_uri": context_uri}
        if start_at_uri:
            play_args["offset"] = {"uri": start_at_uri}
        elif start_at_position >= 0:
            play_args["offset"] = {"position": start_at_position}
        device_id = _coerce_str(arguments.get("device_id"))
        if device_id:
            play_args["device_id"] = device_id

        play_raw = self._start_playback(play_args)
        try:
            play_result = json.loads(play_raw)
        except (json.JSONDecodeError, ValueError):
            play_result = {"raw": play_raw}

        summary: dict[str, Any] = {
            "playlist_id": pid,
            "context_uri": context_uri,
            "start_at_uri": start_at_uri or None,
            "start_at_position": start_at_position if start_at_position >= 0 else None,
            "playback": play_result,
        }

        play_ok = isinstance(play_result, dict) and play_result.get("ok") is True
        if not play_ok:
            summary["ok"] = False
            summary["error"] = (
                "Failed to start playback — repeat/shuffle were not applied. See playback.error/detail."
            )
            return _compact(summary)

        # Apply repeat if requested.
        repeat_raw = arguments.get("repeat")
        repeat_applied: Any = None
        if repeat_raw is not None and not (isinstance(repeat_raw, str) and not repeat_raw.strip()):
            repeat_arg: dict[str, Any] = {"state": repeat_raw}
            if device_id:
                repeat_arg["device_id"] = device_id
            try:
                repeat_raw_out = self._set_repeat(repeat_arg)
                repeat_applied = json.loads(repeat_raw_out)
            except (httpx.HTTPStatusError, json.JSONDecodeError, ValueError) as exc:  # pragma: no cover - network
                repeat_applied = {"ok": False, "error": str(exc)}
        summary["repeat"] = repeat_applied

        # Apply shuffle if requested.
        shuffle_raw = arguments.get("shuffle")
        shuffle_applied: Any = None
        if shuffle_raw is not None:
            shuffle_arg: dict[str, Any] = {"state": shuffle_raw}
            if device_id:
                shuffle_arg["device_id"] = device_id
            try:
                shuffle_raw_out = self._set_shuffle(shuffle_arg)
                shuffle_applied = json.loads(shuffle_raw_out)
            except (httpx.HTTPStatusError, json.JSONDecodeError, ValueError) as exc:  # pragma: no cover - network
                shuffle_applied = {"ok": False, "error": str(exc)}
        summary["shuffle"] = shuffle_applied

        summary["ok"] = True
        return _compact(summary)

    def _add_to_queue(self, arguments: dict[str, Any]) -> str:
        uri = str(arguments.get("uri", "")).strip()
        if not uri:
            return json.dumps({"error": "uri is required"})
        device_id = str(arguments.get("device_id", "")).strip() or self._device_id()
        params: dict[str, str] = {"uri": uri}
        if device_id:
            params["device_id"] = device_id
        self.client.api_post("/me/player/queue", params=params)
        return json.dumps({"ok": True})

    def _set_repeat(self, arguments: dict[str, Any]) -> str:
        raw = str(arguments.get("state", "")).strip().lower()
        aliases = {
            "playlist": "context",
            "all": "context",
            "album": "context",
            "queue": "context",
            "on": "context",
            "true": "context",
            "one": "track",
            "song": "track",
            "single": "track",
            "none": "off",
            "false": "off",
        }
        state = aliases.get(raw, raw)
        if state not in ("track", "context", "off"):
            return json.dumps(
                {
                    "error": "state must be one of: track, context, off",
                    "hint": "Use 'context' to repeat the whole playlist/album, 'track' to loop the current song, 'off' to stop repeating.",
                }
            )
        device_id = str(arguments.get("device_id", "")).strip() or self._device_id()
        params: dict[str, str] = {"state": state}
        if device_id:
            params["device_id"] = device_id
        self.client.api_put("/me/player/repeat", params=params)
        return json.dumps({"ok": True, "state": state, "device_id": device_id or None})

    def _set_shuffle(self, arguments: dict[str, Any]) -> str:
        raw = arguments.get("state")
        if isinstance(raw, str):
            s = raw.strip().lower()
            val = s in ("true", "on", "1", "yes", "shuffle")
        else:
            val = bool(raw)
        device_id = str(arguments.get("device_id", "")).strip() or self._device_id()
        params: dict[str, str] = {"state": "true" if val else "false"}
        if device_id:
            params["device_id"] = device_id
        self.client.api_put("/me/player/shuffle", params=params)
        return json.dumps({"ok": True, "state": val, "device_id": device_id or None})

    def _seek(self, arguments: dict[str, Any]) -> str:
        raw = arguments.get("position_ms", arguments.get("position"))
        pos = _safe_int(raw, -1, lo=0, hi=10 * 60 * 60 * 1000)
        if pos < 0:
            return json.dumps({"error": "position_ms (non-negative integer) is required"})
        device_id = str(arguments.get("device_id", "")).strip() or self._device_id()
        params: dict[str, Any] = {"position_ms": pos}
        if device_id:
            params["device_id"] = device_id
        self.client.api_put("/me/player/seek", params=params)
        return json.dumps({"ok": True, "position_ms": pos})

    def _set_volume(self, arguments: dict[str, Any]) -> str:
        vol = _safe_int(arguments.get("volume_percent", arguments.get("volume")), -1, lo=0, hi=100)
        if vol < 0:
            return json.dumps({"error": "volume_percent is required (0-100)"})
        device_id = str(arguments.get("device_id", "")).strip() or self._device_id()
        params: dict[str, Any] = {"volume_percent": vol}
        if device_id:
            params["device_id"] = device_id
        self.client.api_put("/me/player/volume", params=params)
        return json.dumps({"ok": True, "volume_percent": vol})


OLLAMA_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "spotify_search",
            "description": "Search the Spotify catalog for tracks, artists, or albums (use for vague names before play or add). Use bare ids from results for follow-ups; URIs are normalized by other tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "types": {"type": "string", "description": "Comma-separated spotify types, e.g. track,artist,album"},
                    "market": {"type": "string"},
                    "limit": {"type": "integer"},
                    "offset": {"type": "integer", "description": "Pagination offset (search supports paging)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_me",
            "description": "Get the current Spotify user profile (id, display name, country) plus granted_scopes and missing_playlist_* scope diagnostics. Use to verify the signed-in account's id vs. a playlist's owner.id and to check if consent covers playlist modify/read before blaming Spotify.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "description": "List the signed-in user's playlists with id, name, owner_id, collaborative, public. owner_id lets you decide write-ability: only ids where owner_id == spotify_me.id are writable via spotify_add_tracks_to_playlist. Success here proves you can enumerate — a later 403 on one id is per-playlist (ownership, wrong id, or scopes), not a blanket block. Paginate with offset.",
            "name": "spotify_user_playlists",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                    "offset": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_playlist_tracks",
            "description": "List tracks in one playlist by id (paginate). A 403 here does not mean all playlists are unreadable — try other ids from spotify_user_playlists or play with start_resume_playback context_uri without listing tracks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_id": {"type": "string"},
                    "limit": {"type": "integer"},
                    "offset": {"type": "integer"},
                    "market": {"type": "string"},
                },
                "required": ["playlist_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_get_playlist",
            "description": "Read one playlist's metadata and a page of tracks. A 403 is for this id only — try other ids from spotify_user_playlists or play via start_resume_playback context_uri without reading tracks first.",
            "parameters": {
                "type": "object",
                "properties": {"playlist_id": {"type": "string"}, "market": {"type": "string"}},
                "required": ["playlist_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_get_album",
            "description": "Get album details including release date and tracks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "album_id": {"type": "string"},
                    "market": {"type": "string"},
                },
                "required": ["album_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_get_track",
            "description": "Get a single track's metadata (name, artists, album, duration).",
            "parameters": {
                "type": "object",
                "properties": {"track_id": {"type": "string"}, "market": {"type": "string"}},
                "required": ["track_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_artist_albums",
            "description": "List albums and singles for an artist. Pass artists.items[0].id from spotify_search, or the artist's display name (the tool resolves names via search).",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_id": {"type": "string"},
                    "include_groups": {"type": "string"},
                    "limit": {"type": "integer"},
                    "offset": {"type": "integer"},
                    "market": {"type": "string"},
                },
                "required": ["artist_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_get_artist",
            "description": "Get artist profile (genres, popularity, followers). Pass catalog id or artist name.",
            "parameters": {
                "type": "object",
                "properties": {"artist_id": {"type": "string"}, "market": {"type": "string"}},
                "required": ["artist_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_artist_top_tracks",
            "description": "Get an artist's top tracks in a market. Pass catalog id or artist name.",
            "parameters": {
                "type": "object",
                "properties": {"artist_id": {"type": "string"}, "market": {"type": "string"}},
                "required": ["artist_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_user_saved_tracks",
            "description": "List the user's saved (liked) tracks, paginated.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}, "offset": {"type": "integer"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_create_playlist",
            "description": "Create a new empty playlist for the signed-in user (needs playlist-modify scopes on the token). Then add tracks with spotify_add_tracks_to_playlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "public": {"type": "boolean"},
                    "description": {"type": "string"},
                    "collaborative": {"type": "boolean"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_add_tracks_to_playlist",
            "description": "Append tracks only to playlists **owned** by the user. spotify_user_playlists includes followed playlists too — those ids 403 on add unless you own them; compare get_playlist.owner.id to spotify_me.id. playlist_id from spotify_create_playlist is always writable. Tracks: track_uris, track_ids, tracks, or uris (array or single spotify:track: string). Max 100 per call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_id": {"type": "string"},
                    "track_uris": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "spotify:track: URIs, bare ids, or (if your stack allows) pass search paging via tracks/uris instead",
                    },
                    "track_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Bare 22-char Spotify track ids (optional alternative to track_uris)",
                    },
                    "uris": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Same role as track_uris; some models use this key for track URIs",
                    },
                    "tracks": {
                        "type": "array",
                        "description": "Array of track objects from spotify_search tracks.items (uri or id). Same as passing the search tracks object: the server also accepts {items: [...]} if sent as JSON (unwrap). Prefer a flat array of items here.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "uri": {"type": "string"},
                                "id": {"type": "string"},
                                "track": {
                                    "type": "object",
                                    "properties": {
                                        "uri": {"type": "string"},
                                        "id": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
                "required": ["playlist_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_add_tracks_by_query",
            "description": (
                "COMPOSITE: search Spotify for tracks matching `query` and add real Spotify track URIs "
                "to the playlist in one call — never fabricated ids. Optional `min_year` filters to "
                "tracks released in/after that year when possible. Optional `count` (default 1, max 10). "
                "Defaults to skipping tracks already on the playlist (avoid_duplicates=true). "
                "Prefer this over chaining spotify_search + spotify_add_tracks_to_playlist for natural "
                "requests like \"add a SZA song from 2024 to RNB2025\" — the response's `added_tracks` "
                "array is the authoritative list of what was added."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_id": {
                        "type": "string",
                        "description": "Owned playlist id (from spotify_user_playlists or spotify_create_playlist).",
                    },
                    "query": {
                        "type": "string",
                        "description": "Natural-language Spotify search query, e.g. 'SZA', 'john mayer gravity', 'drake 2024'.",
                    },
                    "count": {
                        "type": "integer",
                        "description": "How many unique tracks to add (default 1, max 10).",
                    },
                    "min_year": {
                        "type": "integer",
                        "description": "Prefer tracks whose album.release_date year is >= this (e.g. 2024). Falls back to any match if no year-matched track is new to the playlist.",
                    },
                    "avoid_duplicates": {
                        "type": "boolean",
                        "description": "If true (default), skip tracks already on the playlist.",
                    },
                    "market": {
                        "type": "string",
                        "description": "ISO country code or 'from_token' (default).",
                    },
                },
                "required": ["playlist_id", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_play_playlist",
            "description": (
                "PLAY NOW — immediate interruption. COMPOSITE tool: start a playlist (optionally at a "
                "specific track via start_at_uri) AND set repeat/shuffle in one call. This is the RIGHT "
                "tool for ALL of these phrases: 'play [playlist]', 'start playing [playlist]', 'play "
                "[playlist] at [track]', 'play [playlist] starting with [track]', 'begin [playlist] "
                "with [track]', 'start playing [playlist] beginning with [track]'. It interrupts the "
                "current song and jumps to start_at_uri — if Spotify initially queues the target "
                "instead of jumping, the tool force-skips to it and reports playback_verified=true. "
                "The response has ok=true only when playback actually switched on the device. For "
                "repeat use 'context' (whole playlist), 'track' (one song), or 'off'. This is NOT for "
                "queueing — use spotify_play_next / spotify_add_to_queue ONLY for literal 'play X next' "
                "/ 'queue X' phrasing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_id": {"type": "string", "description": "Playlist id to play."},
                    "start_at_uri": {
                        "type": "string",
                        "description": "Optional spotify:track:<id> to start the playlist at.",
                    },
                    "start_at_position": {
                        "type": "integer",
                        "description": "Optional 0-based track position to start at (used only if start_at_uri is absent).",
                    },
                    "repeat": {
                        "type": "string",
                        "description": "Optional repeat mode: 'context' | 'track' | 'off' (aliases: on/all/playlist → context, one/song → track, none/false → off).",
                    },
                    "shuffle": {
                        "type": "boolean",
                        "description": "Optional shuffle on/off.",
                    },
                    "device_id": {"type": "string"},
                },
                "required": ["playlist_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_update_playlist",
            "description": "Update playlist name, description, public, and/or collaborative flags.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "public": {"type": "boolean"},
                    "collaborative": {"type": "boolean"},
                },
                "required": ["playlist_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_remove_playlist_tracks",
            "description": "Remove up to 100 tracks from a playlist by URI or 22-char track id. Optional snapshot_id for concurrent edits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_id": {"type": "string"},
                    "track_uris": {"type": "array", "items": {"type": "string"}},
                    "snapshot_id": {"type": "string"},
                },
                "required": ["playlist_id", "track_uris"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_reorder_playlist_tracks",
            "description": "Move a contiguous block of tracks: range_start, range_length, insert_before (0-based indices). Optional snapshot_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_id": {"type": "string"},
                    "range_start": {"type": "integer"},
                    "range_length": {"type": "integer"},
                    "insert_before": {"type": "integer"},
                    "snapshot_id": {"type": "string"},
                },
                "required": ["playlist_id", "insert_before"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_replace_playlist_tracks",
            "description": "Replace ALL tracks in the playlist with up to 100 URIs (call again for larger lists).",
            "parameters": {
                "type": "object",
                "properties": {"playlist_id": {"type": "string"}, "track_uris": {"type": "array", "items": {"type": "string"}}},
                "required": ["playlist_id", "track_uris"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_unfollow_playlist",
            "description": "Remove the playlist from the user's library (delete/unfollow for the current user).",
            "parameters": {
                "type": "object",
                "properties": {"playlist_id": {"type": "string"}},
                "required": ["playlist_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_devices",
            "description": "List available Spotify Connect devices.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_playback_state",
            "description": "Get the current playback state (track, progress, is_playing).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_transfer_playback",
            "description": "Transfer playback to a device id without starting playback.",
            "parameters": {
                "type": "object",
                "properties": {"device_id": {"type": "string"}},
                "required": ["device_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_start_resume_playback",
            "description": (
                "PLAY NOW (interrupts current playback immediately). Start or resume playback. "
                "Use uris: [spotify:track:...] for explicit tracks, or context_uri: spotify:album:..., "
                "spotify:playlist:..., spotify:artist:... for album/playlist/artist context. To start "
                "a playlist AT a specific track, send context_uri plus offset: {\"uri\": \"spotify:track:<id>\"} "
                "(or offset: {\"position\": N} for a 0-based index). The tool verifies that playback "
                "actually switched to the requested context/track — if Spotify accepts the command but "
                "the device keeps playing the old song, the response will have ok=false and "
                "playback_verified=false. Do NOT use this for 'play X next' / 'queue X' — use "
                "spotify_add_to_queue / spotify_play_next for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string"},
                    "uris": {"type": "array", "items": {"type": "string"}},
                    "context_uri": {"type": "string"},
                    "offset": {
                        "type": "object",
                        "description": "Start at a specific track within context_uri. Shape: {uri: 'spotify:track:<id>'} OR {position: N}.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_pause",
            "description": "Pause playback.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_skip_next",
            "description": "Skip to next track.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_skip_previous",
            "description": "Skip to previous track.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_add_to_queue",
            "description": (
                "PLAY NEXT ONLY (append to up-next queue, does NOT interrupt). The current song keeps "
                "playing; the supplied uri plays AFTER it finishes. ONLY use this when the user "
                "EXPLICITLY says 'play X NEXT', 'queue X', 'add X to the queue', 'put X up next', "
                "'after this song play X'. DO NOT use this for 'play X', 'start playing X', 'play X "
                "now', 'play [playlist] at [track]', 'start playing [playlist] beginning with [track]', "
                "'begin [playlist] with [track]' — those ALL require immediate interruption and MUST go "
                "through spotify_start_resume_playback or spotify_play_playlist. If you are unsure, "
                "default to spotify_play_playlist (for playlists) or spotify_start_resume_playback, "
                "not this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {"uri": {"type": "string"}, "device_id": {"type": "string"}},
                "required": ["uri"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_play_next",
            "description": (
                "PLAY NEXT ONLY (semantic alias for spotify_add_to_queue). Appends to the up-next queue; "
                "does NOT interrupt. ONLY use when the user explicitly says 'play X next' / 'queue X' / "
                "'after this one play X'. NEVER use this for 'play X', 'start playing X', 'play X now', "
                "'play [playlist] at [track]' — those require spotify_start_resume_playback or "
                "spotify_play_playlist (immediate interruption)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"uri": {"type": "string"}, "device_id": {"type": "string"}},
                "required": ["uri"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_set_repeat",
            "description": "Set Spotify repeat mode on the active device. Use 'context' to loop the current playlist/album, 'track' to loop the current song, 'off' to stop repeating. Call this AFTER spotify_start_resume_playback when the user asks to repeat/loop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "description": "One of: track, context, off. Aliases accepted: playlist/all/on → context, one/song → track, none → off.",
                    },
                    "device_id": {"type": "string"},
                },
                "required": ["state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_set_shuffle",
            "description": "Toggle Spotify shuffle mode on the active device.",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {"type": "boolean", "description": "true to shuffle, false to turn off."},
                    "device_id": {"type": "string"},
                },
                "required": ["state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_seek",
            "description": "Seek to a position in the currently playing track (position_ms from the start).",
            "parameters": {
                "type": "object",
                "properties": {
                    "position_ms": {"type": "integer"},
                    "device_id": {"type": "string"},
                },
                "required": ["position_ms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_set_volume",
            "description": "Set the playback volume (0-100) on the active device.",
            "parameters": {
                "type": "object",
                "properties": {
                    "volume_percent": {"type": "integer"},
                    "device_id": {"type": "string"},
                },
                "required": ["volume_percent"],
            },
        },
    },
]
