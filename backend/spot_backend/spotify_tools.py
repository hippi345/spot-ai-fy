"""Spot-AI-fy: Spotify Web API tool definitions and executor (shared by MCP, Ollama, Gemini)."""

from __future__ import annotations

import json
import logging
import re
import time

# Delay before a single 404 device-not-active play retry (tests patch this).
PLAY_DEVICE_404_RETRY_DELAY_SECONDS = 0.75
from typing import Any

import httpx

from spot_backend.config import Settings, get_settings
from spot_backend.artist_name_match import (
    BARE_ARTIST_SHORT_QUERY_MAX_LEN,
    artist_name_similarity,
    artist_names_match,
    artist_query_matches_candidate_name,
    normalize_artist_name_for_match,
    resolve_artist_from_search_items,
)
from spot_backend.llm_secret_safety import redact_known_api_keys
from spot_backend.playlist_pick import (
    playlist_id_is_spotify_curated,
    playlist_row_playable_owned,
)
from spot_backend.spotify_dev_limits import (
    SPOTIFY_DEV_MAX_PAGE,
    SPOTIFY_SEARCH_DEFAULT_LIMIT,
    clamp_spotify_page_limit,
)
from spot_backend.spotify_client import (
    SpotifyAuthError,
    SpotifyClient,
    SpotifyQuotaExceededError,
    SpotifyRateLimitError,
)
from spot_backend.spotify_removed_routes import SpotifyRemovedRouteError
from spot_backend.spotify_tools_pr9 import SpotifyToolRunnerPr9Mixin
from spot_backend.token_store import load_device

logger = logging.getLogger(__name__)


def _arguments_include_playlist_tracks(arguments: dict[str, Any]) -> bool:
    for key in ("tracks", "track_uris", "track_ids", "uris", "items"):
        val = arguments.get(key)
        if isinstance(val, list) and val:
            return True
        if isinstance(val, str) and val.strip():
            return True
    return False


def _compact(data: Any, limit: int = 6000) -> str:
    s = json.dumps(data, ensure_ascii=False)
    if len(s) > limit:
        return s[:limit] + "\n... (truncated)"
    return s


_UNWANTED_TRACK_VARIANT_RE = re.compile(
    r"\b(karaoke|cover version|\bcover\b|tribute|instrumental|live at|live from|in the style of)\b",
    re.I,
)

_REMIX_VARIANT_RE = re.compile(r"\b(remix|rework|edit|mix|version)\b", re.I)
_FEAT_VARIANT_RE = re.compile(r"\b(feat\.?|ft\.?|with)\b", re.I)
_LIVE_VARIANT_RE = re.compile(r"\b(live)\b", re.I)


def _normalize_track_title(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


_ALBUM_EDITION_SUFFIX_RE = re.compile(
    r"(?:"
    r"\s*[\(\[]\s*(?:deluxe|expanded|clean|explicit|standard|anniversary|"
    r"remaster(?:ed)?|special\s+edition)\s*(?:edition)?\s*[\)\]]"
    r"|"
    r"\s*-\s*(?:deluxe|expanded|clean|explicit|standard|anniversary|remaster(?:ed)?)\s*$"
    r")",
    re.I,
)


def _normalize_album_title_for_dedupe(name: str) -> str:
    t = (name or "").strip().lower()
    for _ in range(4):
        new = _ALBUM_EDITION_SUFFIX_RE.sub("", t).strip()
        if new == t:
            break
        t = new
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def _dedupe_album_items_by_title(items: list[Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        key = _normalize_album_title_for_dedupe(str(it.get("name") or ""))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _best_artist_name_for_track_search(*candidates: str | None) -> str:
    """Pick a Spotify `artist:"..."` query name; prefer fuller alias forms (e.g. Kanye West over Ye)."""
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        label = candidate.strip()
        if not label or _looks_like_spotify_catalog_id(label):
            continue
        key = normalize_artist_name_for_match(label)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(label)
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    alias_linked = [
        name
        for name in unique
        if any(
            name != other
            and (
                artist_query_matches_candidate_name(name, other)
                or artist_query_matches_candidate_name(other, name)
            )
            for other in unique
        )
    ]
    pool = alias_linked or unique
    return max(pool, key=len)


def _track_matches_artist(
    track: dict[str, Any],
    *,
    artist_id: str,
    artist_name: str,
    name_hints: tuple[str, ...] = (),
) -> bool:
    artists = track.get("artists") if isinstance(track.get("artists"), list) else []
    if not artists:
        # artist:"..." search results are already scoped; some API tiers omit artist objects.
        return True
    aid = (artist_id or "").strip()
    for row in artists:
        if not isinstance(row, dict):
            continue
        credited_id = str(row.get("id") or "").strip()
        if aid and credited_id == aid:
            return True
    name_pool = tuple(
        n.strip()
        for n in (artist_name,) + name_hints
        if isinstance(n, str) and n.strip() and not _looks_like_spotify_catalog_id(n.strip())
    )
    for row in artists:
        if not isinstance(row, dict):
            continue
        an = str(row.get("name") or "").strip()
        if not an:
            continue
        for hint in name_pool:
            if an.casefold() == hint.casefold():
                return True
            if artist_query_matches_candidate_name(hint, an):
                return True
            if artist_names_match(hint, an):
                return True
    return False


def _pick_track_by_popularity(tracks: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not tracks:
        return None
    scored: list[tuple[int, dict[str, Any]]] = []
    for tr in tracks:
        pop = tr.get("popularity")
        if isinstance(pop, int):
            scored.append((pop, tr))
    if scored:
        return max(scored, key=lambda pair: pair[0])[1]
    return tracks[0]


def _slim_recently_played_payload(data: dict[str, Any]) -> dict[str, Any]:
    raw_items = data.get("items") if isinstance(data.get("items"), list) else []
    items: list[dict[str, Any]] = []
    for row in raw_items:
        if not isinstance(row, dict):
            continue
        track = row.get("track") if isinstance(row.get("track"), dict) else {}
        artists_raw = track.get("artists") if isinstance(track.get("artists"), list) else []
        artists = [
            {"name": a.get("name")}
            for a in artists_raw
            if isinstance(a, dict) and isinstance(a.get("name"), str)
        ]
        items.append(
            {
                "played_at": row.get("played_at"),
                "track": {
                    "name": track.get("name"),
                    "uri": track.get("uri"),
                    "id": track.get("id"),
                    "artists": artists,
                },
            }
        )
    out: dict[str, Any] = {"items": items}
    if isinstance(data.get("next"), str):
        out["next"] = data["next"]
    cursors = data.get("cursors")
    if isinstance(cursors, dict):
        out["cursors"] = cursors
    return out


def _score_track_search_candidate(
    track: dict[str, Any],
    *,
    want_title: str = "",
    want_artist: str = "",
) -> int:
    name = str(track.get("name") or "")
    name_low = name.lower()
    norm_name = _normalize_track_title(name)
    score = int(track.get("popularity") or 0)
    if _UNWANTED_TRACK_VARIANT_RE.search(name_low):
        score -= 120
    title = want_title.strip()
    artist = want_artist.strip()
    title_low = title.lower()
    artist_low = artist.lower()
    norm_want = _normalize_track_title(title)
    if norm_want and norm_name == norm_want:
        score += 220
    elif norm_want and norm_want in norm_name:
        score += 90
    elif title_low and title_low in name_low:
        score += 70
    if title_low and name_low == title_low:
        score += 50
    want_remix = bool(_REMIX_VARIANT_RE.search(title_low))
    if _REMIX_VARIANT_RE.search(name_low) and not want_remix:
        score -= 160
    if _FEAT_VARIANT_RE.search(name_low) and not _FEAT_VARIANT_RE.search(title_low):
        score -= 100
    if _LIVE_VARIANT_RE.search(name_low) and "live" not in title_low:
        score -= 90
    artists = track.get("artists") if isinstance(track.get("artists"), list) else []
    artist_names = [
        str(a.get("name") or "").lower()
        for a in artists
        if isinstance(a, dict)
    ]
    if artist_low:
        if artist_names and artist_names[0] == artist_low:
            score += 140
        elif any(an == artist_low for an in artist_names):
            score += 110
        elif any(artist_low in an for an in artist_names):
            score += 45
        if len(artist_names) > 1 and artist_names[0] != artist_low:
            score -= 40
    return score


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


_PLACEHOLDER_DEVICE_IDS = frozenset(
    {"default", "active", "auto", "none", "any", "current", "primary"}
)


def _sanitize_model_device_id(raw: str) -> str:
    """Drop model-invented device placeholders; only real saved ids should be sent."""
    s = (raw or "").strip()
    if not s:
        return ""
    if s.lower() in _PLACEHOLDER_DEVICE_IDS:
        return ""
    return s


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


_LIBRARY_URI_CHUNK = 40


def _release_date_sort_key(release_date: str) -> tuple[int, int, int]:
    parts = (release_date or "0000").split("-")
    year = int(parts[0]) if parts and parts[0].isdigit() else 0
    month = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
    day = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
    return (year, month, day)


def pick_latest_album_release(
    items: list[Any],
    *,
    on_or_before: tuple[int, int, int] | None = None,
) -> dict[str, Any] | None:
    """Pick the newest album/single by release_date (deluxe reissues beat older originals)."""
    candidates = [it for it in items if isinstance(it, dict) and it.get("name")]
    if not candidates:
        return None
    if on_or_before is not None:
        candidates = [
            it
            for it in candidates
            if _release_date_sort_key(str(it.get("release_date") or "")) <= on_or_before
        ]
        if not candidates:
            return None
    return max(
        candidates,
        key=lambda album: _release_date_sort_key(str(album.get("release_date") or "")),
    )


def _spotify_uris_csv(segment: str, ids: list[str]) -> str:
    return ",".join(f"spotify:{segment}:{i}" for i in ids)


def _looks_like_spotify_catalog_id(s: str) -> bool:
    """Spotify track/artist/album ids are 22-char base62-ish strings."""
    if len(s) != 22:
        return False
    return all(c.isalnum() for c in s)


def _catalog_id_from_string(val: str) -> str | None:
    parsed = _parse_spotify_context_ref(val)
    if parsed:
        return parsed[1]
    s = val.strip()
    if _looks_like_spotify_catalog_id(s):
        return s
    return None


def collect_catalog_ids_from_tool_json(obj: Any) -> set[str]:
    """Walk any tool JSON and collect Spotify catalog ids / URI bare ids."""
    found: set[str] = set()

    def walk(node: Any) -> None:  # pylint: disable=too-many-nested-blocks
        if isinstance(node, dict):
            for key, val in node.items():
                if key in ("id", "uri", "context_uri") and isinstance(val, str):
                    cid = _catalog_id_from_string(val)
                    if cid:
                        found.add(cid)
                elif key == "external_urls" and isinstance(val, dict):
                    url = val.get("spotify")
                    if isinstance(url, str):
                        for segment in ("track", "album", "artist", "playlist"):
                            marker = f"/{segment}/"
                            if marker in url:
                                tail = url.split(marker, 1)[1].split("?", 1)[0].strip("/")
                                if _looks_like_spotify_catalog_id(tail):
                                    found.add(tail)
                else:
                    walk(val)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str):
            cid = _catalog_id_from_string(node)
            if cid:
                found.add(cid)

    walk(obj)
    return found


_THIS_PLAYBACK_MARKERS = frozenset(
    {
        "this",
        "this track",
        "this song",
        "this album",
        "this show",
        "this episode",
        "this podcast",
        "current",
        "current track",
        "current album",
        "current show",
        "current episode",
        "playing",
        "now playing",
    }
)

_UNDO_LIBRARY_MARKERS = frozenset(
    {
        "that",
        "that track",
        "that song",
        "that album",
        "that show",
        "that episode",
        "that playlist",
        "undo that",
        "the last one",
        "it",
        "this",
    }
)

_PLAYLIST_VISIBILITY_MISMATCH_NOTE = (
    "I asked Spotify to make it private, but Spotify still shows it as public "
    "(this can lag or be a known Spotify API quirk)."
)

_PLAYBACK_START_FAILED_USER_MESSAGE = (
    "I couldn't start playback on your device, so repeat and shuffle weren't applied. "
    "Open Spotify on your phone or computer, press play on any song, then ask again."
)

_PLAYLIST_PLAY_UNAVAILABLE_USER_MESSAGE = (
    "I couldn't start that playlist from here — try one you created in your library."
)

_GENERIC_PLAYBY_TRACK_TITLES = frozenset(
    {"songs", "song", "music", "tracks", "track", "something"}
)


def _parse_spotify_context_ref(raw: str) -> tuple[str, str] | None:
    """Return (playlist|album|artist|track, bare_id) or None when malformed."""
    s = (raw or "").strip()
    if not s:
        return None
    low = s.lower()
    for segment in ("playlist", "album", "artist", "track"):
        prefix = f"spotify:{segment}:"
        if low.startswith(prefix):
            tail = _normalize_spotify_id(s, segment)
            if _looks_like_spotify_catalog_id(tail):
                return segment, tail
            return None
    bare = s.split("?", 1)[0].strip()
    if _looks_like_spotify_catalog_id(bare):
        return "playlist", bare
    return None


def _spotify_http_error_fields(exc: httpx.HTTPStatusError) -> dict[str, Any]:
    body_redacted = redact_known_api_keys((exc.response.text or "")[:2000])
    return {
        "spotify_http_status": exc.response.status_code,
        "spotify_error_body_redacted": body_redacted[:800],
    }


def _spotify_http_message(exc: httpx.HTTPStatusError) -> str:
    try:
        payload = exc.response.json()
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict) and isinstance(err.get("message"), str):
                return err["message"]
    except (json.JSONDecodeError, ValueError):
        pass
    return (exc.response.text or "")[:400]


def _http_failure_reason(status: int, spot_msg: str | None = None) -> str:
    msg = (spot_msg or "").lower()
    if status == 400:
        if "scope" in msg or "missing scope" in msg:
            return "missing_scope"
        return "http_400_bad_request"
    if status == 401:
        return "http_401_unauthorized"
    if status == 403:
        return "http_403_forbidden"
    if status == 404:
        return "http_404_not_found"
    if status == 429:
        return "rate_limited"
    if status in (408, 504):
        return "timeout"
    if status >= 500:
        return "http_5xx_server_error"
    return f"http_{status}"


def _attach_failure_reason(payload: dict[str, Any], *, default: str | None = None) -> None:
    if payload.get("failure_reason"):
        return
    status = payload.get("spotify_http_status")
    if isinstance(status, int):
        payload["failure_reason"] = _http_failure_reason(
            status,
            payload.get("spotify_api_message") if isinstance(payload.get("spotify_api_message"), str) else None,
        )
        return
    if default:
        payload["failure_reason"] = default


def _spotify_error_is_restriction_violated(exc: httpx.HTTPStatusError) -> bool:
    if exc.response.status_code != 403:
        return False
    msg = _spotify_http_message(exc).lower()
    return "restriction" in msg and "violat" in msg


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


def _shrink_user_playlists_payload(
    data: dict[str, Any],
    me_id: str | None = None,
) -> dict[str, Any]:
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
            tracks = it.get("tracks")
            items_page = it.get("items")
            tracks_total: int | None = None
            if isinstance(tracks, dict) and isinstance(tracks.get("total"), int):
                tracks_total = tracks.get("total")
            elif isinstance(items_page, dict) and isinstance(items_page.get("total"), int):
                tracks_total = items_page.get("total")
            row: dict[str, Any] = {
                "id": pid if isinstance(pid, str) else None,
                "name": str(name) if isinstance(name, str) else "",
                "owner_id": owner.get("id") if isinstance(owner.get("id"), str) else None,
                "collaborative": bool(it.get("collaborative")),
                "public": it.get("public"),
            }
            if tracks_total is not None:
                row["tracks_total"] = tracks_total
            if not isinstance(row["id"], str):
                continue
            row["owned_by_me"] = playlist_row_playable_owned(row, me_id or "")
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


def _shrink_saved_albums_page(data: dict[str, Any]) -> dict[str, Any]:
    out_items: list[dict[str, Any]] = []
    raw = data.get("items")
    if isinstance(raw, list):
        for row in raw[:50]:
            if not isinstance(row, dict):
                continue
            alb = row.get("album")
            if not isinstance(alb, dict):
                continue
            artists = alb.get("artists")
            anames: list[str] = []
            if isinstance(artists, list):
                for a in artists:
                    if isinstance(a, dict) and a.get("name"):
                        anames.append(str(a["name"]))
            out_items.append(
                {
                    "added_at": row.get("added_at"),
                    "name": alb.get("name"),
                    "id": alb.get("id"),
                    "uri": alb.get("uri"),
                    "artists": anames,
                    "release_date": alb.get("release_date"),
                }
            )
    return {
        "total": data.get("total"),
        "limit": data.get("limit"),
        "offset": data.get("offset"),
        "has_next_page": bool(data.get("next")),
        "items": out_items,
    }


def _format_ranked_track_lines(items: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        rank = row.get("rank")
        name = str(row.get("name") or "Unknown track")
        artists_raw = row.get("artists")
        anames: list[str] = []
        if isinstance(artists_raw, list):
            for a in artists_raw:
                if isinstance(a, dict) and a.get("name"):
                    anames.append(str(a["name"]))
                elif isinstance(a, str) and a.strip():
                    anames.append(a.strip())
        artist_str = ", ".join(anames) if anames else "Unknown artist"
        suffix = ""
        if row.get("is_playable") is False:
            suffix = " (not playable here)"
        prefix = f"{rank}. " if rank is not None else ""
        lines.append(f"{prefix}{name} — {artist_str}{suffix}")
    return lines


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
_USER_TOP_SCOPES = ("user-top-read",)
_USER_FOLLOW_READ_SCOPES = ("user-follow-read",)
_USER_LIBRARY_READ_SCOPES = ("user-library-read",)
_USER_LIBRARY_MODIFY_SCOPES = ("user-library-modify",)
_USER_RECENTLY_PLAYED_SCOPES = ("user-read-recently-played",)
_USER_FOLLOW_MODIFY_SCOPES = ("user-follow-modify",)
_USER_PLAYBACK_READ_SCOPES = ("user-read-playback-state",)

_SCOPE_FEATURE_LABELS: dict[str, str] = {
    "playlist-modify-public": "editing public playlists",
    "playlist-modify-private": "editing private playlists",
    "playlist-read-private": "reading private playlists",
    "playlist-read-collaborative": "reading collaborative playlists",
    "user-top-read": "your top artists and tracks",
    "user-follow-read": "artists you follow",
    "user-library-read": "your saved albums and liked tracks",
    "user-library-modify": "saving or removing tracks and albums in your library",
    "user-read-recently-played": "recently played history",
    "user-follow-modify": "following or unfollowing artists",
    "user-read-playback-state": "reading the playback queue",
}

_TOOL_FEATURE_NAMES: dict[str, str] = {
    "spotify_recently_played": "recently played history",
    "spotify_save_tracks": "saving tracks to your library",
    "spotify_unsave_tracks": "removing saved tracks",
    "spotify_save_albums": "saving albums to your library",
    "spotify_unsave_albums": "removing saved albums",
    "spotify_saved_albums": "your saved albums",
    "spotify_follow_artist": "following artists",
    "spotify_unfollow_artist": "unfollowing artists",
    "spotify_get_queue": "the playback queue",
    "spotify_playlists_containing_track": "searching your playlists for a track",
}


def _reconnect_for_scopes(missing: list[str]) -> str:
    labels = [_SCOPE_FEATURE_LABELS.get(s, s) for s in missing]
    if len(labels) == 1:
        return f"Reconnect Spotify to enable {labels[0]}."
    return f"Reconnect Spotify to enable: {', '.join(labels)}."


_TOOL_REQUIRED_SCOPES: dict[str, tuple[str, ...]] = {
    "spotify_add_tracks_to_playlist": _MODIFY_PLAYLIST_SCOPES,
    "spotify_remove_playlist_tracks": _MODIFY_PLAYLIST_SCOPES,
    "spotify_replace_playlist_tracks": _MODIFY_PLAYLIST_SCOPES,
    "spotify_reorder_playlist_tracks": _MODIFY_PLAYLIST_SCOPES,
    "spotify_update_playlist": _MODIFY_PLAYLIST_SCOPES,
    "spotify_create_playlist": _MODIFY_PLAYLIST_SCOPES,
    "spotify_duplicate_playlist": _MODIFY_PLAYLIST_SCOPES,
    "spotify_follow_playlist": _MODIFY_PLAYLIST_SCOPES,
    "spotify_playlist_tracks": _READ_PLAYLIST_SCOPES,
    "spotify_get_playlist": _READ_PLAYLIST_SCOPES,
    "spotify_top_artists": _USER_TOP_SCOPES,
    "spotify_top_tracks": _USER_TOP_SCOPES,
    "spotify_followed_artists": _USER_FOLLOW_READ_SCOPES,
    "spotify_user_saved_tracks": _USER_LIBRARY_READ_SCOPES,
    "spotify_recently_played": _USER_RECENTLY_PLAYED_SCOPES,
    "spotify_save_tracks": _USER_LIBRARY_MODIFY_SCOPES,
    "spotify_unsave_tracks": _USER_LIBRARY_MODIFY_SCOPES,
    "spotify_save_albums": _USER_LIBRARY_MODIFY_SCOPES,
    "spotify_unsave_albums": _USER_LIBRARY_MODIFY_SCOPES,
    "spotify_saved_albums": _USER_LIBRARY_READ_SCOPES,
    "spotify_follow_artist": _USER_FOLLOW_MODIFY_SCOPES,
    "spotify_unfollow_artist": _USER_FOLLOW_MODIFY_SCOPES,
    "spotify_get_queue": _USER_PLAYBACK_READ_SCOPES,
    "spotify_playlists_containing_track": _READ_PLAYLIST_SCOPES,
    "spotify_library_contains": _USER_LIBRARY_READ_SCOPES,
    "spotify_library_save": _USER_LIBRARY_MODIFY_SCOPES,
    "spotify_library_remove": _USER_LIBRARY_MODIFY_SCOPES,
    "spotify_user_saved_shows": _USER_LIBRARY_READ_SCOPES,
    "spotify_user_saved_episodes": _USER_LIBRARY_READ_SCOPES,
    "spotify_user_saved_audiobooks": _USER_LIBRARY_READ_SCOPES,
    "spotify_playlist_builder_preview": _MODIFY_PLAYLIST_SCOPES,
    "spotify_playlist_builder_edit": _MODIFY_PLAYLIST_SCOPES,
    "spotify_playlist_builder_commit": _MODIFY_PLAYLIST_SCOPES,
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


class SpotifyToolRunner(SpotifyToolRunnerPr9Mixin):
    def __init__(
        self,
        client: SpotifyClient | None = None,
        settings: Settings | None = None,
        *,
        conversation_id: str | None = None,
    ) -> None:
        self.client = client or SpotifyClient(settings=settings or get_settings())
        self.settings = self.client.settings
        self.conversation_id = (conversation_id or "").strip() or None
        self._session_known_ids: set[str] = set()
        self._last_library_mutation: dict[str, Any] | None = None
        self._last_session_playlist_id: str | None = None
        self._last_primary_artist_id: str | None = None
        self._last_show_search_id: str | None = None
        self._last_playlist_search_query: str | None = None
        from spot_backend.library_mutation_store import load_last_library_mutation

        prior = load_last_library_mutation(self.conversation_id)
        if isinstance(prior, dict):
            self._last_library_mutation = prior
        from spot_backend.library_mutation_store import load_last_playlist_id

        stored_pid = load_last_playlist_id(self.conversation_id)
        if stored_pid:
            self.note_session_playlist_id(stored_pid)

    def last_saved_track_ids_for_undo(self, conversation_id: str | None = None) -> list[str]:
        """Track ids to unsave for undo — persisted session first, else this runner's last save."""
        from spot_backend.library_mutation_store import last_saved_track_ids

        cid = (conversation_id or self.conversation_id or "").strip() or None
        stored = last_saved_track_ids(cid)
        if stored:
            return stored
        mut = self._last_library_mutation
        if isinstance(mut, dict) and mut.get("segment") == "track":
            ids = mut.get("ids")
            if isinstance(ids, list):
                return [str(i) for i in ids if str(i).strip()]
        return []

    def note_session_playlist_id(self, playlist_id: str) -> None:
        pid = (playlist_id or "").strip()
        if _looks_like_spotify_catalog_id(pid):
            self._last_session_playlist_id = pid
            self._session_known_ids.add(pid)
            from spot_backend.library_mutation_store import record_last_playlist_id

            record_last_playlist_id(self.conversation_id, pid)

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

    def _precheck_scopes(self, name: str) -> str | None:
        required = _TOOL_REQUIRED_SCOPES.get(name)
        if required is None:
            return None
        granted = self.client.get_token_scopes()
        if not granted:
            return None
        missing = _missing_any_of(granted, required)
        if not missing:
            return None
        msg = _reconnect_for_scopes(missing)
        return json.dumps(
            {
                "error": msg,
                "reconnect_spotify_message": msg,
                "feature": _TOOL_FEATURE_NAMES.get(name, "this feature"),
                "missing_scopes": missing,
                "granted_scopes": sorted(granted),
                "stale_scopes_need_reauth": True,
                "suggest_sign_out_of_spotify": True,
                "sign_out_not_recommended": False,
                "reauth_may_resolve": True,
            },
            ensure_ascii=False,
        )

    def run(self, name: str, arguments: dict[str, Any]) -> str:
        self.client.clear_metadata_cache()
        pre = self._precheck_scopes(name)
        if pre:
            return pre
        play_guard = self._precheck_play_catalog_id(name, arguments)
        if play_guard:
            return play_guard
        if name == "spotify_create_playlist":
            from spot_backend.playlist_builder_store import load_playlist_preview

            if load_playlist_preview(self.conversation_id):
                return json.dumps(
                    {
                        "ok": False,
                        "failure_reason": "playlist_builder_active",
                        "error": (
                            "A playlist-builder preview is waiting for approval in this chat. "
                            "Do not call spotify_create_playlist — use spotify_playlist_builder_commit "
                            "after the user confirms, or spotify_playlist_builder_preview to revise."
                        ),
                        "reconnect_spotify_unnecessary": True,
                    },
                    ensure_ascii=False,
                )
        try:
            result = self._dispatch(name, arguments)
        except SpotifyAuthError as e:
            return json.dumps({"error": str(e)})
        except SpotifyQuotaExceededError as e:
            return json.dumps(
                {"error": str(e), "failure_reason": getattr(e, "failure_reason", "quota_exceeded")},
                ensure_ascii=False,
            )
        except SpotifyRateLimitError as e:
            return json.dumps(
                {"error": str(e), "failure_reason": "rate_limited"},
                ensure_ascii=False,
            )
        except SpotifyRemovedRouteError as e:
            return json.dumps(
                {
                    "error": str(e),
                    "failure_reason": getattr(e, "failure_reason", "removed_route"),
                },
                ensure_ascii=False,
            )
        except httpx.HTTPStatusError as e:
            detail: Any
            try:
                detail = e.response.json()
            except (json.JSONDecodeError, ValueError):
                detail = (e.response.text or "")[:800]
            body_redacted = redact_known_api_keys((e.response.text or "")[:2000])
            logger.warning(
                "spotify_tool_http_error tool=%s status=%s body=%s",
                name,
                e.response.status_code,
                body_redacted,
            )
            err: dict[str, Any] = {
                "error": f"Spotify HTTP {e.response.status_code}",
                "detail": detail,
                "spotify_error_body_redacted": body_redacted[:800],
                "spotify_http_status": e.response.status_code,
            }
            try:
                tok = self.client.ensure_fresh_access_token()
                if tok:
                    err["spotify_error_body_redacted"] = redact_known_api_keys(
                        err["spotify_error_body_redacted"],
                        [tok],
                    )
            except (SpotifyAuthError, OSError, RuntimeError, ValueError):
                pass
            spot_msg = _spotify_http_message(e)
            if spot_msg:
                err["spotify_api_message"] = spot_msg
            return self._format_http_error(name, arguments, e, err)
        except Exception as e:
            return json.dumps(
                {
                    "error": f"{type(e).__name__}: {e}",
                    "failure_reason": "parse_error",
                }
            )
        self._remember_tool_catalog_ids(name, result)
        return result

    def _verify_catalog_id_on_spotify(self, kind: str, bare_id: str) -> tuple[bool, str | None]:
        """GET Spotify catalog object; return (exists, user-facing error)."""
        path_map = {
            "track": f"/tracks/{bare_id}",
            "album": f"/albums/{bare_id}",
            "artist": f"/artists/{bare_id}",
            "playlist": f"/playlists/{bare_id}",
            "show": f"/shows/{bare_id}",
            "episode": f"/episodes/{bare_id}",
        }
        path = path_map.get(kind)
        if not path:
            return False, f"Unsupported catalog kind {kind!r}"
        try:
            self.client.api_get(path)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (400, 404):
                return False, (
                    f"Spotify has no {kind} with id {bare_id!r} (HTTP {e.response.status_code}). "
                    "Use an id from spotify_search or another lookup tool in this chat."
                )
            raise
        return True, None

    def _validate_playlist_id_for_mutation(self, pid: str) -> tuple[str | None, str | None]:
        """Reject or substitute playlist ids that were not seen in tool results this session."""
        clean = (pid or "").strip()
        if not clean:
            return None, json.dumps({"error": "playlist_id is required"})
        if clean in self._session_known_ids:
            return clean, None
        fallback = self._last_session_playlist_id
        if fallback and _looks_like_spotify_catalog_id(fallback):
            return fallback, None
        return None, json.dumps(
            {
                "ok": False,
                "error": (
                    "That playlist id was not returned by Spotify tools in this conversation — "
                    "I cannot change it until you create or look up the playlist here."
                ),
                "rejected_playlist_id": clean,
                "hint": "Use the id from spotify_create_playlist or spotify_user_playlists in this chat.",
                "reconnect_spotify_unnecessary": True,
                "sign_out_not_recommended": True,
            },
            ensure_ascii=False,
        )

    def _precheck_play_catalog_id(self, name: str, arguments: dict[str, Any]) -> str | None:
        if name not in ("spotify_play_playlist", "spotify_start_resume_playback"):
            return None
        raw = ""
        if name == "spotify_play_playlist":
            raw = _pick_arg(arguments, "playlist_id", "playlistId", "id")
            bare = _normalize_spotify_id(raw, "playlist") if raw else ""
            blocked = self._block_editorial_playlist_id(bare) if bare else None
            if blocked:
                return blocked
        else:
            raw = _pick_arg(arguments, "context_uri", "playlist_id", "album_id", "artist_id", "id")
            if not raw and isinstance(arguments.get("uris"), list) and arguments["uris"]:
                raw = str(arguments["uris"][0])
        parsed = _parse_spotify_context_ref(raw) if raw else None
        if parsed:
            kind, bare = parsed
            if bare in self._session_known_ids:
                return None
            ok, verify_err = self._verify_catalog_id_on_spotify(kind, bare)
            if not ok:
                return json.dumps(
                    {
                        "error": verify_err or f"Unknown {kind} id {bare!r}",
                        "hint": "Call spotify_search, spotify_user_playlists, spotify_get_track, or similar first, "
                        "then replay with an id from that tool output.",
                        "reconnect_spotify_unnecessary": True,
                        "sign_out_not_recommended": True,
                    },
                    ensure_ascii=False,
                )
            return None
        if raw:
            norm = _normalize_spotify_id(raw, "playlist")
            if _looks_like_spotify_catalog_id(norm):
                if norm in self._session_known_ids:
                    return None
                ok, verify_err = self._verify_catalog_id_on_spotify("playlist", norm)
                if not ok:
                    return json.dumps(
                        {
                            "error": verify_err or f"Unknown playlist id {norm!r}",
                            "hint": "Call spotify_user_playlists or spotify_search first.",
                            "reconnect_spotify_unnecessary": True,
                            "sign_out_not_recommended": True,
                        },
                        ensure_ascii=False,
                    )
            elif raw.strip().lower().startswith("spotify:"):
                return json.dumps(
                    {
                        "error": "Malformed Spotify id or URI — use a 22-character id from spotify_search or list tools.",
                        "hint": "Search first, then play using an id returned in the tool JSON.",
                        "reconnect_spotify_unnecessary": True,
                    },
                    ensure_ascii=False,
                )
        return None

    def _remember_tool_catalog_ids(self, name: str, result: str) -> None:
        try:
            data = json.loads(result)
        except (json.JSONDecodeError, TypeError, ValueError):
            return
        self._session_known_ids.update(collect_catalog_ids_from_tool_json(data))
        if name in ("spotify_create_playlist", "spotify_duplicate_playlist", "spotify_playlist_builder_commit"):
            pid = data.get("id") or data.get("new_playlist_id") or data.get("playlist_id") or data.get("playlist_id_for_add_tracks")
            if isinstance(pid, str) and _looks_like_spotify_catalog_id(pid):
                self._last_session_playlist_id = pid
                self.note_session_playlist_id(pid)
        if name == "spotify_search" and isinstance(data, dict):
            artists = data.get("artists")
            if isinstance(artists, dict):
                items = artists.get("items")
                if isinstance(items, list) and items and isinstance(items[0], dict):
                    aid = items[0].get("id")
                    if isinstance(aid, str) and _looks_like_spotify_catalog_id(aid):
                        self._last_primary_artist_id = aid
        if name in (
            "spotify_get_artist",
            "spotify_play_artist_popular_track",
            "spotify_play_artist",
        ) and isinstance(data, dict):
            aid = data.get("id")
            if isinstance(aid, str) and _looks_like_spotify_catalog_id(aid):
                self._last_primary_artist_id = aid

    def _format_http_error(
        self,
        name: str,
        arguments: dict[str, Any],
        e: httpx.HTTPStatusError,
        err: dict[str, Any],
    ) -> str:
        try:
            return self._dispatch_error_json(name, arguments, e, err)
        except Exception:  # pragma: no cover - fallback
            return json.dumps(err, ensure_ascii=False)

    def _dispatch_error_json(
        self,
        name: str,
        arguments: dict[str, Any],
        e: httpx.HTTPStatusError,
        err: dict[str, Any],
    ) -> str:
        detail = err.get("detail")
        if not isinstance(detail, (dict, str)):
            detail = ((e.response.text or "")[:800] or str(e))
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
                reconnect_msg = _reconnect_for_scopes(missing)
                err["error"] = reconnect_msg
                err["reconnect_spotify_message"] = reconnect_msg
                err["feature"] = _TOOL_FEATURE_NAMES.get(name, "this feature")
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
        _attach_failure_reason(err, default="http_error")
        return json.dumps(err)

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
            case "spotify_artist_latest_album":
                return self._artist_latest_album(arguments)
            case "spotify_play_artist_latest_release":
                return self._play_artist_latest_release(arguments)
            case "spotify_get_artist":
                return self._get_artist(arguments)
            case "spotify_artist_top_tracks":
                return self._artist_top_tracks_removed()
            case "spotify_play_artist_popular_track":
                return self._play_artist_popular_track(arguments)
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
            case "spotify_recently_played":
                return self._recently_played(arguments)
            case "spotify_save_tracks":
                return self._save_tracks(arguments)
            case "spotify_unsave_tracks":
                return self._unsave_tracks(arguments)
            case "spotify_save_albums":
                return self._save_albums(arguments)
            case "spotify_unsave_albums":
                return self._unsave_albums(arguments)
            case "spotify_saved_albums":
                return self._saved_albums(arguments)
            case "spotify_follow_artist":
                return self._follow_artist(arguments)
            case "spotify_unfollow_artist":
                return self._unfollow_artist(arguments)
            case "spotify_get_queue":
                return self._get_queue()
            case "spotify_remove_from_queue":
                return self._remove_from_queue(arguments)
            case "spotify_playlists_containing_track":
                return self._playlists_containing_track(arguments)
            case "spotify_top_artists":
                return self._top_artists(arguments)
            case "spotify_top_tracks":
                return self._top_tracks(arguments)
            case "spotify_followed_artists":
                return self._followed_artists(arguments)
            case "spotify_user_public_playlists":
                return self._user_public_playlists(arguments)
            case "spotify_search_playlists":
                return self._search_playlists(arguments)
            case "spotify_follow_playlist":
                return self._follow_playlist(arguments)
            case "spotify_duplicate_playlist":
                return self._duplicate_playlist(arguments)
            case "spotify_create_playlist":
                return self._create_playlist(arguments)
            case "spotify_add_tracks_to_playlist":
                return self._add_tracks(arguments)
            case "spotify_add_tracks_by_query":
                return self._add_tracks_by_query(arguments)
            case "spotify_play_artist":
                return self._play_artist(arguments)
            case "spotify_play_bare":
                return self._play_bare(arguments)
            case "spotify_play_track":
                return self._play_track(arguments)
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
                return self._skip_next(arguments)
            case "spotify_skip_previous":
                return self._skip_previous(arguments)
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
            case "spotify_library_contains":
                return self._library_contains(arguments)
            case "spotify_library_save":
                return self._library_save_uris(arguments)
            case "spotify_library_remove":
                return self._library_remove_uris(arguments)
            case "spotify_get_show":
                return self._get_show(arguments)
            case "spotify_get_show_episodes":
                return self._get_show_episodes(arguments)
            case "spotify_get_episode":
                return self._get_episode(arguments)
            case "spotify_get_audiobook":
                return self._get_audiobook(arguments)
            case "spotify_get_audiobook_chapters":
                return self._get_audiobook_chapters(arguments)
            case "spotify_get_chapter":
                return self._get_chapter(arguments)
            case "spotify_user_saved_shows":
                return self._me_shows(arguments)
            case "spotify_user_saved_episodes":
                return self._me_episodes(arguments)
            case "spotify_user_saved_audiobooks":
                return self._me_audiobooks(arguments)
            case "spotify_playlist_builder_preview":
                return self._playlist_builder_preview(arguments)
            case "spotify_playlist_builder_edit":
                return self._playlist_builder_edit(arguments)
            case "spotify_playlist_builder_commit":
                return self._playlist_builder_commit(arguments)
            case "spotify_play_show_latest_episode":
                return self._play_show_latest_episode(arguments)
            case _:
                return json.dumps({"error": f"Unknown tool: {name}"})

    def _artist_search_items(self, query: str, market: str) -> list[dict[str, Any]]:
        q = query.strip()
        if not q:
            return []
        data = self.client.api_get(
            "/search",
            params={"q": q, "type": "artist", "limit": SPOTIFY_DEV_MAX_PAGE, "market": market},
        )
        if not isinstance(data, dict):
            return []
        artists = data.get("artists")
        if not isinstance(artists, dict):
            return []
        items = artists.get("items")
        if not isinstance(items, list):
            return []
        return [it for it in items if isinstance(it, dict)]

    def _resolve_artist_search_query(
        self, query: str, market: str
    ) -> tuple[str, str, int, str] | None:
        q = query.strip()
        if not q:
            return None
        if _looks_like_spotify_catalog_id(q):
            return (q, "", 0, "catalog_id")
        items = self._artist_search_items(q, market)
        row = resolve_artist_from_search_items(q, items)
        if row:
            return row
        return None

    def _first_artist_id_from_search(self, query: str, market: str) -> str | None:
        resolved = self._resolve_artist_search_query(query, market)
        return resolved[0] if resolved else None

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
        limit = _safe_int(arguments.get("limit"), SPOTIFY_SEARCH_DEFAULT_LIMIT, lo=1, hi=10)
        offset = _safe_int(arguments.get("offset"), 0, lo=0, hi=950)
        if not q:
            return json.dumps({"error": "query is required", "failure_reason": "validation_error"})
        try:
            data = self.client.api_get(
                "/search",
                params={"q": q, "type": types, "market": market, "limit": limit, "offset": offset},
            )
        except httpx.HTTPStatusError as e:
            return json.dumps(
                {
                    "ok": False,
                    "error": _spotify_http_message(e),
                    "failure_reason": _http_failure_reason(e.response.status_code, _spotify_http_message(e)),
                    **_spotify_http_error_fields(e),
                },
                ensure_ascii=False,
            )
        # Spotify dev-mode search responses interleave literal `null` entries into every
        # `items` array (sparsification — see backend/scripts/diag_search_and_owned.py).
        # Small/local models read the nulls as "no results" and hallucinate an empty
        # answer. Strip them before the LLM ever sees the payload and keep a note of the
        # real `total` so the agent can say "found some" even when the page is sparse.
        if isinstance(data, dict):
            for bucket_key in ("tracks", "artists", "albums", "playlists", "shows", "episodes", "audiobooks"):
                bucket = data.get(bucket_key)
                if not isinstance(bucket, dict):
                    continue
                raw_items = bucket.get("items") if isinstance(bucket.get("items"), list) else []
                clean = [it for it in raw_items if isinstance(it, dict)]
                bucket["items"] = clean
                bucket["returned_count"] = len(clean)
                if "total" in bucket:
                    bucket["total_in_catalog"] = bucket.get("total")
                if bucket_key == "shows" and clean:
                    first = clean[0]
                    if isinstance(first, dict) and isinstance(first.get("id"), str):
                        self._last_show_search_id = first["id"]
                        self._session_known_ids.add(first["id"])
        return _compact(data)

    def _user_playlists(self, arguments: dict[str, Any]) -> str:
        limit = clamp_spotify_page_limit(arguments.get("limit"), default=SPOTIFY_DEV_MAX_PAGE)
        offset = _safe_int(arguments.get("offset"), 0, lo=0, hi=900_000)
        data = self.client.api_get("/me/playlists", params={"limit": limit, "offset": offset})
        me_id: str | None = None
        try:
            me = self.client.api_get("/me")
            if isinstance(me, dict) and isinstance(me.get("id"), str):
                me_id = me["id"]
        except httpx.HTTPStatusError:
            me_id = None
        if isinstance(data, dict):
            data = _shrink_user_playlists_payload(data, me_id=me_id)
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
        limit = clamp_spotify_page_limit(arguments.get("limit"), default=SPOTIFY_DEV_MAX_PAGE)
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
        data = self.client.api_get(f"/albums/{aid}", params={"market": market})
        if isinstance(data, dict):
            items, total, truncated = self._fetch_album_tracks_paginated(aid, market=market, cap=50)
            if items:
                track_block: dict[str, Any] = {
                    "items": items,
                    "total": total if total is not None else len(items),
                    "limit": len(items),
                    "offset": 0,
                }
                if truncated:
                    track_block["truncated"] = True
                    track_block["note"] = (
                        f"Album has more than {len(items)} tracks; only the first {len(items)} are listed."
                    )
                data["tracks"] = track_block
        return _compact(data, limit=8000)

    def _fetch_album_tracks_paginated(
        self, album_id: str, *, market: str, cap: int = 50
    ) -> tuple[list[Any], int | None, bool]:
        """Page GET /albums/{id}/tracks until cap or no next page."""
        collected: list[Any] = []
        total: int | None = None
        offset = 0
        truncated = False
        while len(collected) < cap:
            page_limit = min(SPOTIFY_DEV_MAX_PAGE, cap - len(collected))
            page = self.client.api_get(
                f"/albums/{album_id}/tracks",
                params={"limit": page_limit, "offset": offset, "market": market},
            )
            if not isinstance(page, dict):
                break
            if total is None and isinstance(page.get("total"), int):
                total = page["total"]
            batch = page.get("items")
            if not isinstance(batch, list) or not batch:
                break
            collected.extend(batch)
            offset += len(batch)
            next_url = page.get("next")
            if not next_url or len(collected) >= cap:
                if next_url and len(collected) >= cap:
                    truncated = True
                break
            if isinstance(total, int) and offset >= total:
                break
        if isinstance(total, int) and total > len(collected):
            truncated = truncated or total > len(collected)
        return collected, total, truncated

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
            artist_id = _coerce_str(raw_id, "").strip()
        if not artist_id:
            return json.dumps({"error": "artist_id is required"})
        include_groups = _normalize_include_groups(_coerce_str(arguments.get("include_groups"), "album"))
        # Feb-2026 dev-mode: /artists/{id}/albums limit max is 10 (limit>10 → HTTP 400).
        limit = _safe_int(arguments.get("limit"), 10, lo=1, hi=10)
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
        count_studio = arguments.get("count_studio_albums")
        want_studio_count = (
            count_studio is True
            or str(count_studio or "").strip().lower() in ("1", "true", "yes")
            or (offset == 0 and include_groups == "album")
        )
        if want_studio_count and offset == 0:
            try:
                pool = self._paginate_artist_album_items(
                    canonical_id,
                    "album",
                    market,
                    max_pages=40,
                )
            except httpx.HTTPStatusError as e:
                spot_msg = _spotify_http_message(e)
                return json.dumps(
                    {
                        "error": f"Spotify HTTP {e.response.status_code}",
                        "spotify_api_message": spot_msg,
                        "artist_id": canonical_id,
                        "hint": (
                            "Could not list albums for this artist. Do not state an album count — "
                            "say you could not get Spotify's count."
                        ),
                    },
                    ensure_ascii=False,
                )
            studio_items = [
                it
                for it in pool
                if isinstance(it, dict)
                and str(it.get("album_type") or it.get("album_group") or "") == "album"
            ]
            deduped = _dedupe_album_items_by_title(studio_items)
            payload: dict[str, Any] = {
                "artist_id": canonical_id,
                "studio_album_count_deduped": len(deduped),
                "spotify_catalog_note": (
                    "Studio albums on Spotify (include_groups=album), deduped by normalized title "
                    "to collapse deluxe/clean/explicit editions."
                ),
                "discography_counts": {"albums": len(deduped)},
                "assistant_guidance": (
                    "Report studio_album_count_deduped as Drake's studio-album count on Spotify. "
                    "If this tool returned an error, do not state any number."
                ),
                "sample_albums": [
                    {"id": it.get("id"), "name": it.get("name")}
                    for it in deduped[:8]
                    if isinstance(it, dict)
                ],
            }
            if include_groups != "album":
                page = self.client.api_get(
                    f"/artists/{canonical_id}/albums",
                    params={
                        "include_groups": include_groups,
                        "limit": limit,
                        "offset": offset,
                        "market": market,
                    },
                )
                if isinstance(page, dict):
                    payload["items_page"] = {
                        "items": page.get("items"),
                        "total": page.get("total"),
                        "limit": page.get("limit"),
                        "offset": page.get("offset"),
                    }
            return json.dumps(payload, ensure_ascii=False)

        page = self.client.api_get(
            f"/artists/{canonical_id}/albums",
            params={
                "include_groups": include_groups,
                "limit": limit,
                "offset": offset,
                "market": market,
            },
        )
        if not isinstance(page, dict):
            return _compact(page)
        slim: dict[str, Any] = {
            "artist_id": canonical_id,
            "items": page.get("items"),
            "total": page.get("total"),
            "limit": page.get("limit"),
            "offset": page.get("offset"),
            "next": page.get("next"),
        }
        return json.dumps(slim, ensure_ascii=False)

    def _paginate_artist_album_items(
        self,
        canonical_id: str,
        include_groups: str,
        market: str,
        *,
        max_pages: int = 8,
    ) -> list[dict[str, Any]]:
        """Collect artist album/single items across pages (Spotify does not sort by date)."""
        limit = 10
        offset = 0
        collected: list[dict[str, Any]] = []
        total: int | None = None
        for _ in range(max_pages):
            page = self.client.api_get(
                f"/artists/{canonical_id}/albums",
                params={
                    "include_groups": include_groups,
                    "limit": limit,
                    "offset": offset,
                    "market": market,
                },
            )
            if not isinstance(page, dict):
                break
            if total is None and isinstance(page.get("total"), int):
                total = page["total"]
            batch = page.get("items")
            if not isinstance(batch, list) or not batch:
                break
            collected.extend([it for it in batch if isinstance(it, dict)])
            offset += len(batch)
            if isinstance(total, int) and offset >= total:
                break
            if not page.get("next"):
                if total is None or offset >= total:
                    break
        return collected

    def _pick_latest_artist_release(
        self,
        canonical_id: str,
        *,
        prefer: str,
        market: str,
        include_groups: str = "album,single",
    ) -> tuple[dict[str, Any] | None, str]:
        from datetime import date

        today_key = _release_date_sort_key(date.today().isoformat())
        prefer_norm = (prefer or "release").strip().lower()
        pool = self._paginate_artist_album_items(canonical_id, include_groups, market)
        if prefer_norm == "single":
            singles = [it for it in pool if str(it.get("album_type") or "") == "single"]
            if not singles:
                singles = self._paginate_artist_album_items(canonical_id, "single", market)
            latest = pick_latest_album_release(singles, on_or_before=today_key)
            return latest, "single"
        if prefer_norm == "album":
            albums = [it for it in pool if str(it.get("album_type") or "") == "album"]
            if not albums:
                albums = self._paginate_artist_album_items(canonical_id, "album", market)
            latest = pick_latest_album_release(albums, on_or_before=today_key)
            return latest, "album"
        combined = pool or self._paginate_artist_album_items(
            canonical_id, include_groups or "album,single", market
        )
        latest = pick_latest_album_release(combined, on_or_before=today_key)
        kind = str(latest.get("album_type") if latest else "release")
        return latest, kind

    def _artist_latest_album(self, arguments: dict[str, Any]) -> str:
        from datetime import date

        raw_id = _pick_arg(arguments, "artist_id", "artistId", "id", "artist_name")
        artist_id = _normalize_spotify_id(raw_id, "artist")
        if not artist_id:
            artist_id = _coerce_str(raw_id, "").strip()
        include_groups = _normalize_include_groups(
            _coerce_str(arguments.get("include_groups"), "album,single")
        )
        prefer = _coerce_str(arguments.get("prefer"), "album").strip().lower()
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

        latest, kind = self._pick_latest_artist_release(
            canonical_id,
            prefer=prefer,
            market=market,
            include_groups=include_groups,
        )
        if not latest:
            return json.dumps({"error": "No releases found for this artist", "artist_id": canonical_id})
        return json.dumps(
            {
                "ok": True,
                "artist_id": canonical_id,
                "latest_release": latest,
                "latest_album": latest,
                "release_date": latest.get("release_date"),
                "album_type": latest.get("album_type"),
                "prefer": prefer,
                "release_kind": kind,
                "reference_date_utc": date.today().isoformat(),
                "assistant_guidance": (
                    "latest_release is the newest album or single already released (release_date <= today). "
                    "For play requests call spotify_play_artist_latest_release instead of asking the user to confirm."
                ),
            },
            ensure_ascii=False,
        )

    def _play_artist_latest_release(self, arguments: dict[str, Any]) -> str:
        from datetime import date

        raw_id = _pick_arg(arguments, "artist_id", "artistId", "id", "artist_name")
        artist_ref = _coerce_str(raw_id, "").strip()
        kind = _coerce_str(arguments.get("kind"), "single").strip().lower()
        if kind not in ("single", "album", "any"):
            kind = "single"
        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        norm_id = _normalize_spotify_id(artist_ref, "artist")
        canonical_id = self._canonical_artist_id(norm_id or artist_ref, market)
        if not canonical_id:
            return json.dumps(
                {
                    "error": "Could not resolve artist to a Spotify catalog id",
                    "hint": "Pass artist_id from spotify_search or a recognizable artist name.",
                    "query_tried": artist_ref,
                },
                ensure_ascii=False,
            )

        prefer = "release" if kind == "any" else kind
        latest, picked_kind = self._pick_latest_artist_release(
            canonical_id,
            prefer=prefer,
            market=market,
        )
        if not latest and kind == "single":
            latest, picked_kind = self._pick_latest_artist_release(
                canonical_id,
                prefer="release",
                market=market,
            )
        if not latest:
            return json.dumps(
                {"error": "No released singles or albums found for this artist", "artist_id": canonical_id},
                ensure_ascii=False,
            )

        album_id = latest.get("id")
        if not isinstance(album_id, str) or not album_id.strip():
            return json.dumps({"error": "Latest release is missing a catalog id", "latest_release": latest})

        play_raw = self._play_playlist(
            {
                "playlist_id": f"spotify:album:{album_id.strip()}",
                "market": market,
                **{
                    k: v
                    for k, v in arguments.items()
                    if k in ("device_id", "offset", "position_ms", "repeat", "shuffle")
                },
            }
        )
        try:
            play_data = json.loads(play_raw)
        except (json.JSONDecodeError, ValueError):
            play_data = {"raw": play_raw}
        if not isinstance(play_data, dict):
            play_data = {"raw": play_raw}
        play_data["latest_release"] = latest
        play_data["release_kind"] = picked_kind
        play_data["reference_date_utc"] = date.today().isoformat()
        if kind == "single" and picked_kind != "single":
            play_data["assistant_guidance"] = (
                "No released single was found; played the newest release of any type instead."
            )
        elif play_data.get("ok"):
            play_data["assistant_guidance"] = (
                "Playback started for the newest released "
                f"{picked_kind}. Reply in one short sentence — do not ask to confirm."
            )
        return json.dumps(play_data, ensure_ascii=False)

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

    def _artist_top_tracks_removed(self) -> str:
        return json.dumps(
            {
                "error": (
                    "spotify_artist_top_tracks is disabled — Spotify removed GET /artists/{id}/top-tracks "
                    "for development-mode apps (Feb 2026). Use spotify_play_artist_popular_track or spotify_search."
                ),
                "tool_removed": True,
                "reconnect_spotify_unnecessary": True,
            },
            ensure_ascii=False,
        )

    def _resolve_artist_id_and_name(
        self,
        raw_ref: str,
        market: str,
    ) -> tuple[str | None, str]:
        parsed = _parse_spotify_context_ref(raw_ref)
        if parsed and parsed[0] == "artist":
            cid = parsed[1]
            name = ""
            try:
                art = self.client.api_get(f"/artists/{cid}")
                if isinstance(art, dict) and isinstance(art.get("name"), str):
                    name = art["name"]
            except httpx.HTTPStatusError:
                pass
            return cid, name
        cid = self._canonical_artist_id(raw_ref.strip(), market)
        if not cid:
            return None, raw_ref.strip()
        name = raw_ref.strip()
        if _looks_like_spotify_catalog_id(name):
            name = ""
        try:
            art = self.client.api_get(f"/artists/{cid}")
            if isinstance(art, dict) and isinstance(art.get("name"), str):
                name = art["name"]
        except httpx.HTTPStatusError:
            pass
        return cid, name

    def _artist_name_for_track_search(
        self,
        artist_id: str,
        market: str,
        *hints: str | None,
    ) -> str:
        api_name = ""
        try:
            art = self.client.api_get(f"/artists/{artist_id}")
            if isinstance(art, dict) and isinstance(art.get("name"), str):
                api_name = art["name"].strip()
        except httpx.HTTPStatusError:
            pass
        return _best_artist_name_for_track_search(api_name, *hints)

    def _search_artist_tracks_for_popularity(
        self,
        *,
        artist_id: str,
        artist_name: str,
        market: str,
        max_pages: int = 3,
        name_hints: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        search_name = artist_name.strip() or _best_artist_name_for_track_search(*name_hints)
        query = f'artist:"{search_name}"' if search_name else artist_id
        collected: list[dict[str, Any]] = []
        for page_idx in range(max_pages):
            search = self.client.api_get(
                "/search",
                params={
                    "q": query,
                    "type": "track",
                    "market": market,
                    "limit": 10,
                    "offset": page_idx * 10,
                },
            )
            tracks_obj = search.get("tracks") if isinstance(search, dict) else None
            items = tracks_obj.get("items") if isinstance(tracks_obj, dict) else None
            if not isinstance(items, list) or not items:
                break
            for tr in items:
                if isinstance(tr, dict) and _track_matches_artist(
                    tr,
                    artist_id=artist_id,
                    artist_name=search_name or artist_name,
                    name_hints=name_hints,
                ):
                    collected.append(tr)
            if not isinstance(tracks_obj, dict) or not tracks_obj.get("next"):
                break
        return collected

    def _search_tracks_by_artist_id(
        self,
        *,
        artist_id: str,
        market: str,
        limit: int = 10,
        name_hints: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        """Fallback when artist-name scoped search returns no playable rows."""
        aid = (artist_id or "").strip()
        if not _looks_like_spotify_catalog_id(aid):
            return []
        data = self.client.api_get(
            "/search",
            params={"q": aid, "type": "track", "market": market, "limit": limit},
        )
        tracks_obj = data.get("tracks") if isinstance(data, dict) else None
        items = tracks_obj.get("items") if isinstance(tracks_obj, dict) else None
        if not isinstance(items, list):
            return []
        out: list[dict[str, Any]] = []
        for tr in items:
            if isinstance(tr, dict) and _track_matches_artist(
                tr, artist_id=aid, artist_name="", name_hints=name_hints
            ):
                out.append(tr)
        return out

    def _play_artist_via_context_uri(
        self,
        *,
        artist_id: str,
        artist_name: str,
        device_id: str | None,
    ) -> dict[str, Any]:
        play_args: dict[str, Any] = {"context_uri": f"spotify:artist:{artist_id}"}
        if device_id:
            play_args["device_id"] = device_id
        play_args["playback_request_label"] = artist_name or artist_id
        play_raw = self._start_playback(play_args)
        try:
            play_result = json.loads(play_raw)
        except (json.JSONDecodeError, ValueError):
            play_result = {"ok": False, "raw": play_raw}
        play_ok = isinstance(play_result, dict) and play_result.get("ok") is True
        player = self._poll_player_state(attempts=6, delay_s=0.5)
        verified = bool(
            isinstance(play_result, dict) and play_result.get("playback_verified") is True
        )
        if player:
            ctx = player.get("context") if isinstance(player.get("context"), dict) else None
            ctx_uri = str(ctx.get("uri") or "") if ctx else ""
            if ctx_uri == f"spotify:artist:{artist_id}":
                verified = True
            elif self._player_item_features_artist(player, artist_name, artist_id=artist_id):
                verified = True
        summary: dict[str, Any] = {
            "artist_id": artist_id,
            "artist_name": artist_name,
            "play_body": {"context_uri": f"spotify:artist:{artist_id}"},
            "playback": play_result,
            "player_after": player,
            "playback_verified": verified,
            "ok": play_ok or verified,
            "selection_method": "artist context_uri (no track search hits)",
        }
        if not play_ok and not verified:
            user_msg = play_result.get("user_message") if isinstance(play_result, dict) else None
            if isinstance(user_msg, str) and user_msg.strip():
                summary["user_message"] = user_msg.strip()
                summary["error"] = user_msg.strip()
            else:
                summary["error"] = _PLAYBACK_START_FAILED_USER_MESSAGE
            summary["failure_reason"] = (
                self._playback_failure_reason_from_result(play_result)
                if isinstance(play_result, dict)
                else "playback_not_verified"
            )
            if not verified:
                summary["failure_reason"] = "playback_not_verified"
            self._lift_failure_fields(summary, play_result if isinstance(play_result, dict) else {})
        return summary

    def _play_artist_popular_track(self, arguments: dict[str, Any]) -> str:
        raw_id = _pick_arg(arguments, "artist_id", "id")
        norm_id = _normalize_spotify_id(raw_id, "artist") if raw_id else ""
        if norm_id and _looks_like_spotify_catalog_id(norm_id):
            raw_ref = norm_id
        else:
            raw_ref = _pick_arg(
                arguments,
                "artist",
                "artist_name",
                "name",
                "artist_id",
                "id",
            )
        if not raw_ref or not str(raw_ref).strip():
            return json.dumps(
                {
                    "error": "artist is required",
                    "hint": "Pass the artist name (e.g. Band of Horses) from chat context or spotify_search.",
                }
            )
        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        cid, artist_name = self._resolve_artist_id_and_name(str(raw_ref).strip(), market)
        if not cid:
            return json.dumps(
                {
                    "ok": False,
                    "error": "Could not find that artist on Spotify.",
                    "query_tried": str(raw_ref).strip(),
                    "reconnect_spotify_unnecessary": True,
                },
                ensure_ascii=False,
            )
        track_search_name = self._artist_name_for_track_search(
            cid, market, artist_name, str(raw_ref).strip()
        )
        name_hints = tuple(
            h
            for h in (track_search_name, artist_name, str(raw_ref).strip())
            if isinstance(h, str) and h.strip()
        )
        tracks = self._search_artist_tracks_for_popularity(
            artist_id=cid,
            artist_name=track_search_name,
            market=market,
            name_hints=name_hints,
        )
        if not tracks:
            tracks = self._search_tracks_by_artist_id(
                artist_id=cid,
                market=market,
                name_hints=name_hints,
            )
        picked = _pick_track_by_popularity(tracks)
        if not picked:
            device_id = _coerce_str(arguments.get("device_id"))
            fallback = self._play_artist_via_context_uri(
                artist_id=cid,
                artist_name=track_search_name or artist_name,
                device_id=device_id,
            )
            return _compact(fallback)
        play_body = self._album_offset_body_for_track(picked)
        if not play_body:
            return json.dumps(
                {
                    "ok": False,
                    "artist_id": cid,
                    "error": "I couldn't resolve an album context for that track.",
                    "reconnect_spotify_unnecessary": True,
                },
                ensure_ascii=False,
            )
        track_name = str(picked.get("name") or "that track")
        play_args: dict[str, Any] = dict(play_body)
        device_id = _coerce_str(arguments.get("device_id"))
        if device_id:
            play_args["device_id"] = device_id
        play_args["playback_request_label"] = track_name
        play_raw = self._start_playback(play_args)
        try:
            play_result = json.loads(play_raw)
        except (json.JSONDecodeError, ValueError):
            play_result = {"ok": False, "raw": play_raw}
        play_ok = isinstance(play_result, dict) and play_result.get("ok") is True
        player = self._poll_player_state(attempts=6, delay_s=0.5)
        verified = bool(
            isinstance(play_result, dict) and play_result.get("playback_verified") is True
        )
        if player and isinstance(picked, dict):
            track_uri = str(picked.get("uri") or "")
            if track_uri and self._track_uri_from_player(player) == track_uri:
                verified = True
        summary: dict[str, Any] = {
            "ok": play_ok or verified,
            "artist_id": cid,
            "artist_name": artist_name,
            "track": {
                "id": picked.get("id"),
                "name": track_name,
                "popularity": picked.get("popularity"),
            },
            "playback": play_result,
            "playback_verified": verified,
            "selection_method": (
                "highest popularity among search results"
                if isinstance(picked.get("popularity"), int)
                else "first relevant search result (Spotify ranking)"
            ),
            "assistant_guidance": (
                f"Reply in one short honest sentence naming the track ({track_name!r}) now playing."
            ),
        }
        if not play_ok and not verified:
            user_msg = play_result.get("user_message") if isinstance(play_result, dict) else None
            if isinstance(user_msg, str) and user_msg.strip():
                summary["user_message"] = user_msg.strip()
                summary["error"] = user_msg.strip()
            else:
                summary["error"] = _PLAYBACK_START_FAILED_USER_MESSAGE
        return _compact(summary)

    def _get_playlist(self, arguments: dict[str, Any]) -> str:
        pid, err = self._playlist_id_from_arguments(arguments)
        if err:
            return err
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
            if e.response.status_code == 404:
                return json.dumps(
                    {
                        "ok": False,
                        "lookup_only": True,
                        "optional_lookup_failure": True,
                        "user_message": _PLAYLIST_PLAY_UNAVAILABLE_USER_MESSAGE,
                        "error": _PLAYLIST_PLAY_UNAVAILABLE_USER_MESSAGE,
                        "reconnect_spotify_unnecessary": True,
                    },
                    ensure_ascii=False,
                )
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
        from spot_backend.chat_tool_state import is_playlist_pronoun_reference

        raw_ref = _pick_arg(arguments, "playlist_id", "playlistId", "id")
        if is_playlist_pronoun_reference(raw_ref or ""):
            pid, err = self._resolve_playlist_id_from_arg(raw_ref or "")
            if err:
                return err
        else:
            pid = _normalize_spotify_id(raw_ref or "", "playlist")
            err = None
        if not pid:
            return json.dumps({"error": "playlist_id is required"})
        validated, reject = self._validate_playlist_id_for_mutation(pid)
        if reject:
            return reject
        pid = validated or pid
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
        out: dict[str, Any] = {"ok": True, "updated_fields": list(body.keys())}
        if "public" in body:
            requested_public = bool(body["public"])
            out["visibility_change_requested"] = True
            checked: dict[str, Any] | None = None
            try:
                checked = self.client.api_get(f"/playlists/{pid}", params={"fields": "id,public,name"})
            except httpx.HTTPStatusError:
                checked = None
            if isinstance(checked, dict) and checked.get("public") is not requested_public:
                out["ok"] = False
                out["visibility_mismatch"] = True
                out["requested_public"] = requested_public
                out["actual_public"] = checked.get("public")
                out["public"] = checked.get("public")
                out["visibility_warning"] = _PLAYLIST_VISIBILITY_MISMATCH_NOTE
                out["visibility_result"] = (
                    "Update request sent, but Spotify still reports this playlist as public; "
                    "do not say the playlist is private or that the change succeeded."
                )
                out["assistant_reply_instruction"] = (
                    "The private request was sent, but Spotify still reports public. "
                    "Do not say the playlist is private, is now private, was updated to be private, "
                    "or that the visibility change succeeded. Confirm only that you sent the update; "
                    "the app adds any visibility note."
                )
            elif isinstance(checked, dict) and "public" in checked:
                out["public"] = checked.get("public")
        return json.dumps(out, ensure_ascii=False)

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
        data = self.client.api_put(f"/playlists/{pid}/items", json_body=params)
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

    def _resolve_playlist_id_from_arg(self, raw: str) -> tuple[str | None, str | None]:
        """Resolve playlist id or name from the user's library. Returns (id, error_json)."""
        text = (raw or "").strip()
        if not text:
            return None, json.dumps({"error": "playlist_id is required"})
        from spot_backend.chat_tool_state import is_playlist_pronoun_reference

        if is_playlist_pronoun_reference(text):
            mut = self._last_library_mutation
            if isinstance(mut, dict) and mut.get("segment") == "playlist":
                ids = mut.get("ids")
                if isinstance(ids, list) and ids:
                    last = str(ids[-1]).strip()
                    if _looks_like_spotify_catalog_id(last):
                        return last, None
            if self._last_session_playlist_id:
                return self._last_session_playlist_id, None
            return None, json.dumps(
                {
                    "error": "No playlist from this chat to refer to yet.",
                    "hint": "Create or mention a playlist first, or pass its id from spotify_user_playlists.",
                    "reconnect_spotify_unnecessary": True,
                }
            )
        norm = _normalize_spotify_id(text, "playlist")
        if norm and _looks_like_spotify_catalog_id(norm):
            return norm, None
        want = text.lower()
        page = self.client.api_get(
            "/me/playlists",
            params={"limit": SPOTIFY_DEV_MAX_PAGE},
        )
        items = page.get("items") if isinstance(page, dict) else None
        if not isinstance(items, list):
            return None, json.dumps(
                {
                    "error": f"No playlist named {text!r} in your library",
                    "hint": "Call spotify_user_playlists and pass the exact id, not the display name.",
                }
            )
        matches: list[str] = []
        for row in items:
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            pid = row.get("id")
            if not isinstance(name, str) or not isinstance(pid, str):
                continue
            if name.strip().lower() == want:
                matches.append(pid)
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, json.dumps(
                {
                    "error": f"Multiple playlists named {text!r} — pass the id from spotify_user_playlists",
                    "matching_playlist_ids": matches,
                }
            )
        return None, json.dumps(
            {
                "error": f"No playlist named {text!r} in your library",
                "hint": "Call spotify_user_playlists and pass the exact id.",
            }
        )

    def _playlist_id_from_arguments(self, arguments: dict[str, Any]) -> tuple[str | None, str | None]:
        raw = _pick_arg(arguments, "playlist_id", "playlistId", "id")
        return self._resolve_playlist_id_from_arg(raw)

    def _unfollow_playlist(self, arguments: dict[str, Any]) -> str:
        pid, err = self._playlist_id_from_arguments(arguments)
        if err:
            return err
        if not pid:
            return json.dumps({"error": "playlist_id is required", "failure_reason": "validation_error"})
        blocked = self._block_editorial_playlist_id(pid)
        if blocked:
            return blocked
        ok, verify_err = self._verify_catalog_id_on_spotify("playlist", pid)
        if not ok:
            return json.dumps(
                {
                    "ok": False,
                    "failure_reason": "invalid_playlist_id",
                    "error": verify_err or f"Unknown playlist id {pid!r}",
                },
                ensure_ascii=False,
            )
        self._library_delete_uris([f"spotify:playlist:{pid}"])
        try:
            check = self.client.api_get(
                "/me/library/contains",
                params={"uris": f"spotify:playlist:{pid}"},
            )
            still_saved = isinstance(check, list) and bool(check) and check[0]
        except httpx.HTTPStatusError:
            still_saved = None
        if still_saved is True:
            return json.dumps(
                {
                    "ok": False,
                    "failure_reason": "remove_verify_failed",
                    "error": "Spotify still reports that playlist as saved after removal.",
                    "playlist_id": pid,
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "ok": True,
                "playlist_id": pid,
                "removed_from_library": True,
                "verified_removed": still_saved is False,
            }
        )

    def _user_saved_tracks(self, arguments: dict[str, Any]) -> str:
        limit = clamp_spotify_page_limit(arguments.get("limit"), default=SPOTIFY_DEV_MAX_PAGE)
        offset = _safe_int(arguments.get("offset"), 0, lo=0, hi=900_000)
        data = self.client.api_get("/me/tracks", params={"limit": limit, "offset": offset})
        if isinstance(data, dict):
            data = _shrink_saved_tracks_page(data)
        return _compact(data, limit=8000)

    def _collect_catalog_ids(self, arguments: dict[str, Any], segment: str, *keys: str) -> list[str]:
        out: list[str] = []
        for key in keys:
            raw = arguments.get(key)
            if raw is None:
                continue
            if isinstance(raw, str):
                tid = _normalize_spotify_id(raw, segment)
                if tid:
                    out.append(tid)
            elif isinstance(raw, list):
                for item in raw:
                    if isinstance(item, str):
                        tid = _normalize_spotify_id(item, segment)
                        if tid:
                            out.append(tid)
        # dedupe preserve order
        seen: set[str] = set()
        uniq: list[str] = []
        for i in out:
            if i not in seen:
                seen.add(i)
                uniq.append(i)
        return uniq

    def _recently_played(self, arguments: dict[str, Any]) -> str:
        limit = clamp_spotify_page_limit(arguments.get("limit"), default=SPOTIFY_DEV_MAX_PAGE)
        after = _pick_arg(arguments, "after", "cursor")
        params: dict[str, Any] = {"limit": limit}
        if after:
            params["after"] = after
        before = _pick_arg(arguments, "before")
        if before:
            params["before"] = before
        data = self.client.api_get("/me/player/recently-played", params=params)
        if isinstance(data, dict):
            slim = _slim_recently_played_payload(data)
            items = slim.get("items") if isinstance(slim.get("items"), list) else []
            while True:
                slim["items"] = items
                slim["returned_count"] = len(items)
                if isinstance(data.get("items"), list) and len(data["items"]) > len(items):
                    slim["truncated"] = True
                    slim["total_fetched"] = len(data["items"])
                out = _compact(slim, limit=8000)
                if len(out) <= 8000 and not out.endswith("(truncated)"):
                    return out
                if not items:
                    return out
                items = items[:-1]
        return _compact(data, limit=8000)

    def _library_put_uris(self, uris: list[str]) -> None:
        for offset in range(0, len(uris), _LIBRARY_URI_CHUNK):
            chunk = uris[offset : offset + _LIBRARY_URI_CHUNK]
            self.client.api_put("/me/library", params={"uris": ",".join(chunk)})

    def _library_delete_uris(self, uris: list[str]) -> None:
        for offset in range(0, len(uris), _LIBRARY_URI_CHUNK):
            chunk = uris[offset : offset + _LIBRARY_URI_CHUNK]
            self.client.api_delete("/me/library", params={"uris": ",".join(chunk)})

    def _player_state_snapshot(self) -> dict[str, Any] | None:
        try:
            state = self.client.api_get("/me/player", params={"additional_types": "episode"})
        except httpx.HTTPStatusError:
            try:
                state = self.client.api_get("/me/player")
            except httpx.HTTPStatusError:
                return None
        return state if isinstance(state, dict) else None

    def _playback_catalog_id(self, segment: str) -> str | None:
        state = self._player_state_snapshot()
        if not state:
            return None
        item = state.get("item")
        if not isinstance(item, dict):
            return None
        if segment == "track":
            tid = item.get("id")
            if isinstance(tid, str) and _looks_like_spotify_catalog_id(tid):
                return tid
            return None
        if segment == "album":
            album = item.get("album")
            if isinstance(album, dict):
                aid = album.get("id")
                if isinstance(aid, str) and _looks_like_spotify_catalog_id(aid):
                    return aid
        if segment == "artist":
            artists = item.get("artists")
            if isinstance(artists, list):
                for artist in artists:
                    if isinstance(artist, dict):
                        aid = artist.get("id")
                        if isinstance(aid, str) and _looks_like_spotify_catalog_id(aid):
                            return aid
        if segment == "episode":
            eid = item.get("id")
            if item.get("type") == "episode" and isinstance(eid, str) and _looks_like_spotify_catalog_id(eid):
                return eid
        if segment == "show":
            show = item.get("show")
            if isinstance(show, dict):
                sid = show.get("id")
                if isinstance(sid, str) and _looks_like_spotify_catalog_id(sid):
                    return sid
        return None

    def _verify_library_segment_ids(self, segment: str, ids: list[str]) -> str | None:
        for bare in ids:
            if bare in self._session_known_ids:
                continue
            ok, verify_err = self._verify_catalog_id_on_spotify(segment, bare)
            if not ok:
                return verify_err or f"Spotify has no {segment} with id {bare!r}"
        return None

    def _verify_library_segment_ids_strict(self, segment: str, ids: list[str]) -> str | None:
        for bare in ids:
            ok, verify_err = self._verify_catalog_id_on_spotify(segment, bare)
            if not ok:
                return verify_err or f"Spotify has no {segment} with id {bare!r}"
        return None

    def _record_library_mutation(self, segment: str, ids: list[str]) -> None:
        clean = [i for i in ids if isinstance(i, str) and i.strip()]
        if clean:
            self._last_library_mutation = {"segment": segment, "ids": clean}
            from spot_backend.library_mutation_store import record_library_mutation

            record_library_mutation(self.conversation_id, segment, clean)

    def _resolve_library_segment_ids(
        self, arguments: dict[str, Any], segment: str, *keys: str
    ) -> list[str]:
        for key in keys:
            raw = arguments.get(key)
            if isinstance(raw, str):
                low = raw.strip().lower()
                if low in _UNDO_LIBRARY_MARKERS:
                    mut = self._last_library_mutation
                    if isinstance(mut, dict) and mut.get("segment") == segment:
                        ids = mut.get("ids")
                        if isinstance(ids, list) and ids:
                            return [str(i) for i in ids if str(i).strip()]
                if low in _THIS_PLAYBACK_MARKERS:
                    cid = self._playback_catalog_id(segment)
                    return [cid] if cid else []
            if isinstance(raw, str) and not raw.strip():
                cid = self._playback_catalog_id(segment)
                return [cid] if cid else []
        return self._collect_catalog_ids(arguments, segment, *keys)

    def _save_tracks(self, arguments: dict[str, Any]) -> str:
        raw_single = _pick_arg(arguments, "track_id", "track_ids", "ids")
        if isinstance(raw_single, str) and raw_single.strip():
            low = raw_single.strip().lower()
            if low.startswith("spotify:show:") or low.startswith("spotify:episode:"):
                return json.dumps(
                    {
                        "ok": False,
                        "failure_reason": "invalid_uri_type",
                        "error": "save_tracks only accepts track ids/URIs, not podcast shows.",
                        "hint": "Use spotify_library_save with spotify:show:<id> to save a show.",
                    },
                    ensure_ascii=False,
                )
        ids = self._resolve_library_segment_ids(arguments, "track", "track_ids", "ids", "track_id")
        if not ids:
            return json.dumps(
                {
                    "error": "track_id or track_ids is required",
                    "failure_reason": "empty_args",
                    "hint": "Pass the current song with track_id='' or 'this', or call spotify_playback_state first.",
                }
            )
        if len(ids) > 50:
            return json.dumps({"error": "At most 50 track ids per call"})
        verify_err = self._verify_library_segment_ids("track", ids)
        if verify_err:
            return json.dumps(
                {"ok": False, "error": verify_err, "reconnect_spotify_unnecessary": True},
                ensure_ascii=False,
            )
        self._library_put_uris([f"spotify:track:{i}" for i in ids])
        self._record_library_mutation("track", ids)
        return json.dumps({"ok": True, "saved_track_ids": ids})

    def _unsave_tracks(self, arguments: dict[str, Any]) -> str:
        ids = self._resolve_library_segment_ids(arguments, "track", "track_ids", "ids", "track_id")
        if not ids:
            return json.dumps({"error": "track_id or track_ids is required"})
        if len(ids) > 50:
            return json.dumps({"error": "At most 50 track ids per call"})
        verify_err = self._verify_library_segment_ids_strict("track", ids)
        if verify_err:
            return json.dumps(
                {"ok": False, "error": verify_err, "reconnect_spotify_unnecessary": True},
                ensure_ascii=False,
            )
        self._library_delete_uris([f"spotify:track:{i}" for i in ids])
        return json.dumps({"ok": True, "removed_track_ids": ids})

    def _save_albums(self, arguments: dict[str, Any]) -> str:
        raw_check = _pick_arg(arguments, "album_id", "album_ids", "ids")
        if isinstance(raw_check, str) and "show:" in raw_check.lower():
            return json.dumps(
                {
                    "ok": False,
                    "failure_reason": "invalid_uri_type",
                    "error": "save_albums only accepts album ids/URIs, not podcast shows.",
                    "hint": "Use spotify_library_save with spotify:show:<id> for shows.",
                },
                ensure_ascii=False,
            )
        ids = self._resolve_library_segment_ids(arguments, "album", "album_ids", "ids", "album_id")
        if not ids:
            return json.dumps(
                {
                    "error": "album_id or album_ids is required",
                    "hint": "To save the album of the current song, pass album_id='this album' or call spotify_playback_state first.",
                }
            )
        if len(ids) > 50:
            return json.dumps({"error": "At most 50 album ids per call"})
        for bare in ids:
            if bare not in self._session_known_ids:
                ok, verify_err = self._verify_catalog_id_on_spotify("album", bare)
                if not ok:
                    return json.dumps(
                        {
                            "ok": False,
                            "failure_reason": "unknown_id",
                            "error": verify_err or f"Spotify has no album with id {bare!r}",
                            "reconnect_spotify_unnecessary": True,
                        },
                        ensure_ascii=False,
                    )
        verify_err = self._verify_library_segment_ids("album", ids)
        if verify_err:
            return json.dumps(
                {
                    "ok": False,
                    "failure_reason": "unknown_id",
                    "error": verify_err,
                    "reconnect_spotify_unnecessary": True,
                },
                ensure_ascii=False,
            )
        self._library_put_uris([f"spotify:album:{i}" for i in ids])
        self._record_library_mutation("album", ids)
        return json.dumps({"ok": True, "saved_album_ids": ids})

    def _unsave_albums(self, arguments: dict[str, Any]) -> str:
        ids = self._resolve_library_segment_ids(arguments, "album", "album_ids", "ids", "album_id")
        if not ids:
            return json.dumps({"error": "album_id or album_ids is required"})
        if len(ids) > 50:
            return json.dumps({"error": "At most 50 album ids per call"})
        verify_err = self._verify_library_segment_ids_strict("album", ids)
        if verify_err:
            return json.dumps(
                {"ok": False, "error": verify_err, "reconnect_spotify_unnecessary": True},
                ensure_ascii=False,
            )
        self._library_delete_uris([f"spotify:album:{i}" for i in ids])
        return json.dumps({"ok": True, "removed_album_ids": ids})

    def _saved_albums(self, arguments: dict[str, Any]) -> str:
        limit = clamp_spotify_page_limit(arguments.get("limit"), default=SPOTIFY_DEV_MAX_PAGE)
        offset = _safe_int(arguments.get("offset"), 0, lo=0, hi=900_000)
        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        data = self.client.api_get(
            "/me/albums",
            params={"limit": limit, "offset": offset, "market": market},
        )
        if not isinstance(data, dict):
            return _compact(data, limit=8000)
        slim = _shrink_saved_albums_page(data)
        items = slim.get("items") if isinstance(slim.get("items"), list) else []
        lines: list[str] = []
        for i, row in enumerate(items):
            if not isinstance(row, dict):
                continue
            anames = row.get("artists") if isinstance(row.get("artists"), list) else []
            artist_str = ", ".join(str(a) for a in anames if str(a).strip())
            lines.append(
                f"{offset + i + 1}. {row.get('name') or 'Unknown'} — {artist_str or 'Unknown artist'}"
            )
        summary = "\n".join(lines) if lines else "No saved albums on this page."
        if slim.get("has_next_page"):
            summary += f"\n(More saved albums available — pass offset={offset + len(items)}.)"
        payload = {
            "ok": True,
            "total": slim.get("total"),
            "limit": slim.get("limit"),
            "offset": offset,
            "returned_count": len(items),
            "items": items,
            "summary_lines": lines,
            "user_message": summary,
        }
        return _compact(payload, limit=8000)

    def _follow_artist(self, arguments: dict[str, Any]) -> str:
        ids = self._resolve_library_segment_ids(arguments, "artist", "artist_ids", "ids", "artist_id")
        if not ids:
            return json.dumps({"error": "artist_id or artist_ids is required"})
        if len(ids) > 50:
            return json.dumps({"error": "At most 50 artist ids per call"})
        verify_err = self._verify_library_segment_ids("artist", ids)
        if verify_err:
            return json.dumps(
                {"ok": False, "error": verify_err, "reconnect_spotify_unnecessary": True},
                ensure_ascii=False,
            )
        self._library_put_uris([f"spotify:artist:{i}" for i in ids])
        self._record_library_mutation("artist", ids)
        return json.dumps({"ok": True, "followed_artist_ids": ids})

    def _unfollow_artist(self, arguments: dict[str, Any]) -> str:
        ids = self._resolve_library_segment_ids(arguments, "artist", "artist_ids", "ids", "artist_id")
        if not ids:
            return json.dumps({"error": "artist_id or artist_ids is required"})
        if len(ids) > 50:
            return json.dumps({"error": "At most 50 artist ids per call"})
        verify_err = self._verify_library_segment_ids_strict("artist", ids)
        if verify_err:
            return json.dumps(
                {"ok": False, "error": verify_err, "reconnect_spotify_unnecessary": True},
                ensure_ascii=False,
            )
        self._library_delete_uris([f"spotify:artist:{i}" for i in ids])
        return json.dumps({"ok": True, "unfollowed_artist_ids": ids})

    def _get_queue(self) -> str:
        data = self.client.api_get("/me/player/queue")
        return _compact(data, limit=8000)

    def _remove_from_queue(self, arguments: dict[str, Any]) -> str:
        _ = arguments
        return json.dumps(
            {
                "ok": False,
                "error": "Spotify's Web API cannot remove a specific song from the queue — only skip to the next track.",
                "hint": "Offer spotify_skip_next (or spotify_play_next) to skip what's playing, or explain this limitation.",
                "spotify_api_limitation": True,
                "try_instead": ["spotify_skip_next", "spotify_play_next"],
            },
            ensure_ascii=False,
        )

    def _playlists_containing_track(self, arguments: dict[str, Any]) -> str:
        track_id = _normalize_spotify_id(
            _pick_arg(arguments, "track_id", "trackId", "id", "uri"), "track"
        )
        if not track_id:
            return json.dumps({"error": "track_id is required"})
        max_playlists = _safe_int(arguments.get("max_playlists"), 50, lo=1, hi=200)
        max_pages_per_playlist = _safe_int(arguments.get("max_pages_per_playlist"), 3, lo=1, hi=10)
        want_uri = f"spotify:track:{track_id}"
        matches: list[dict[str, Any]] = []
        offset = 0
        scanned = 0
        pages_fetched = 0
        truncated = False
        page: dict[str, Any] = {}
        while scanned < max_playlists:
            page = self.client.api_get(
                "/me/playlists",
                params={"limit": SPOTIFY_DEV_MAX_PAGE, "offset": offset},
            )
            pages_fetched += 1
            if not isinstance(page, dict):
                break
            items = page.get("items") if isinstance(page.get("items"), list) else []
            if not items:
                break
            for pl in items:
                if scanned >= max_playlists:
                    truncated = True
                    break
                if not isinstance(pl, dict):
                    continue
                pid = pl.get("id")
                if not isinstance(pid, str):
                    continue
                scanned += 1
                name = pl.get("name")
                owner = pl.get("owner") if isinstance(pl.get("owner"), dict) else {}
                found = False
                track_offset = 0
                for _ in range(max_pages_per_playlist):
                    tr_page = self.client.api_get(
                        f"/playlists/{pid}/items",
                        params={
                            "limit": SPOTIFY_DEV_MAX_PAGE,
                            "offset": track_offset,
                            "fields": "items(item(id,uri)),next",
                        },
                    )
                    pages_fetched += 1
                    if not isinstance(tr_page, dict):
                        break
                    rows = tr_page.get("items") if isinstance(tr_page.get("items"), list) else []
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        item = row.get("item") if isinstance(row.get("item"), dict) else row.get("track")
                        if not isinstance(item, dict):
                            continue
                        iid = item.get("id")
                        uri = item.get("uri")
                        if iid == track_id or uri == want_uri:
                            found = True
                            break
                    if found:
                        break
                    if not tr_page.get("next"):
                        break
                    track_offset += 100
                if found:
                    matches.append(
                        {
                            "playlist_id": pid,
                            "name": name,
                            "owner_id": owner.get("id") if isinstance(owner, dict) else None,
                        }
                    )
            if truncated:
                break
            if not page.get("next"):
                break
            offset += 50
        if isinstance(page, dict) and page.get("next") and scanned >= max_playlists:
            truncated = True
        out: dict[str, Any] = {
            "track_id": track_id,
            "playlists": matches,
            "scanned": scanned,
            "truncated": truncated,
            "pages_fetched": pages_fetched,
        }
        if truncated:
            out["note"] = (
                f"Stopped after scanning {scanned} playlists (max_playlists={max_playlists}). "
                "Increase max_playlists or narrow with spotify_user_playlists if you need full coverage."
            )
        return json.dumps(out, ensure_ascii=False)

    @staticmethod
    def _coerce_time_range(raw: Any) -> str:
        s = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "short": "short_term",
            "month": "short_term",
            "monthly": "short_term",
            "4weeks": "short_term",
            "4_weeks": "short_term",
            "medium": "medium_term",
            "halfyear": "medium_term",
            "half_year": "medium_term",
            "6months": "medium_term",
            "6_months": "medium_term",
            "long": "long_term",
            "year": "long_term",
            "yearly": "long_term",
            "alltime": "long_term",
            "all_time": "long_term",
        }
        s = aliases.get(s, s)
        if s in ("short_term", "medium_term", "long_term"):
            return s
        return "medium_term"

    def _top_artists(self, arguments: dict[str, Any]) -> str:
        time_range = self._coerce_time_range(arguments.get("time_range"))
        limit = clamp_spotify_page_limit(arguments.get("limit"), default=SPOTIFY_DEV_MAX_PAGE)
        offset = _safe_int(arguments.get("offset"), 0, lo=0, hi=49)
        data = self.client.api_get(
            "/me/top/artists",
            params={"time_range": time_range, "limit": limit, "offset": offset},
        )
        if not isinstance(data, dict):
            return _compact(data, limit=8000)
        raw_items = data.get("items") if isinstance(data.get("items"), list) else []
        items: list[dict[str, Any]] = []
        for i, a in enumerate(raw_items):
            if not isinstance(a, dict):
                continue
            images = a.get("images") if isinstance(a.get("images"), list) else []
            first_img = images[0] if images and isinstance(images[0], dict) else None
            followers = a.get("followers") if isinstance(a.get("followers"), dict) else {}
            ext = a.get("external_urls") if isinstance(a.get("external_urls"), dict) else {}
            items.append(
                {
                    "rank": offset + i + 1,
                    "id": a.get("id"),
                    "name": a.get("name"),
                    "uri": a.get("uri"),
                    "popularity": a.get("popularity"),
                    "followers": followers.get("total") if isinstance(followers, dict) else None,
                    "genres": a.get("genres") if isinstance(a.get("genres"), list) else [],
                    "image": first_img.get("url") if isinstance(first_img, dict) else None,
                    "external_url": ext.get("spotify") if isinstance(ext, dict) else None,
                }
            )
        shrunk = {
            "time_range": time_range,
            "limit": limit,
            "offset": offset,
            "total": data.get("total"),
            "items": items,
            "hint": (
                "rank is 1-based within the current time_range (short_term ~last 4 weeks, "
                "medium_term ~last 6 months, long_term = calculated from ~the user's all-time "
                "history). Spotify refreshes these daily and does not return per-artist play counts."
            ),
        }
        return _compact(shrunk, limit=8000)

    def _top_tracks(self, arguments: dict[str, Any]) -> str:
        time_range = self._coerce_time_range(arguments.get("time_range"))
        limit = clamp_spotify_page_limit(arguments.get("limit"), default=SPOTIFY_DEV_MAX_PAGE)
        offset = _safe_int(arguments.get("offset"), 0, lo=0, hi=49)
        data = self.client.api_get(
            "/me/top/tracks",
            params={"time_range": time_range, "limit": limit, "offset": offset},
        )
        if not isinstance(data, dict):
            return _compact(data, limit=8000)
        raw_items = data.get("items") if isinstance(data.get("items"), list) else []
        items: list[dict[str, Any]] = []
        for i, t in enumerate(raw_items):
            if not isinstance(t, dict):
                continue
            album = t.get("album") if isinstance(t.get("album"), dict) else {}
            artists_raw = t.get("artists") if isinstance(t.get("artists"), list) else []
            artists = [
                {"id": a.get("id"), "name": a.get("name")}
                for a in artists_raw
                if isinstance(a, dict)
            ]
            ext = t.get("external_urls") if isinstance(t.get("external_urls"), dict) else {}
            items.append(
                {
                    "rank": offset + i + 1,
                    "id": t.get("id"),
                    "name": t.get("name"),
                    "uri": t.get("uri"),
                    "popularity": t.get("popularity"),
                    "duration_ms": t.get("duration_ms"),
                    "explicit": t.get("explicit"),
                    "is_playable": t.get("is_playable"),
                    "artists": artists,
                    "album": {
                        "id": album.get("id"),
                        "name": album.get("name"),
                        "release_date": album.get("release_date"),
                    } if album else None,
                    "external_url": ext.get("spotify") if isinstance(ext, dict) else None,
                }
            )
        summary_lines = _format_ranked_track_lines(items)
        user_message = "\n".join(summary_lines) if summary_lines else "No top tracks returned."
        shrunk = {
            "time_range": time_range,
            "limit": limit,
            "offset": offset,
            "total": data.get("total"),
            "items": items,
            "summary_lines": summary_lines,
            "user_message": user_message,
            "hint": (
                "rank is 1-based within the current time_range (short_term ~last 4 weeks, "
                "medium_term ~last 6 months, long_term = calculated from ~the user's all-time "
                "history). Spotify refreshes these daily and does not return per-track play counts."
            ),
        }
        return _compact(shrunk, limit=8000)

    def _followed_artists(self, arguments: dict[str, Any]) -> str:
        """List artists the signed-in user follows. Web API does NOT expose followed *users*."""
        limit = clamp_spotify_page_limit(arguments.get("limit"), default=SPOTIFY_DEV_MAX_PAGE)
        after = _coerce_str(_pick_arg(arguments, "after", "cursor", "next_cursor"), "")
        params: dict[str, Any] = {"type": "artist", "limit": limit}
        if after:
            params["after"] = after
        data = self.client.api_get("/me/following", params=params)
        if not isinstance(data, dict):
            return _compact(data, limit=8000)
        block = data.get("artists") if isinstance(data.get("artists"), dict) else {}
        raw_items = block.get("items") if isinstance(block.get("items"), list) else []
        items: list[dict[str, Any]] = []
        for a in raw_items:
            if not isinstance(a, dict):
                continue
            images = a.get("images") if isinstance(a.get("images"), list) else []
            first_img = images[0] if images and isinstance(images[0], dict) else None
            followers = a.get("followers") if isinstance(a.get("followers"), dict) else {}
            ext = a.get("external_urls") if isinstance(a.get("external_urls"), dict) else {}
            items.append(
                {
                    "id": a.get("id"),
                    "name": a.get("name"),
                    "uri": a.get("uri"),
                    "popularity": a.get("popularity"),
                    "followers": followers.get("total") if isinstance(followers, dict) else None,
                    "genres": a.get("genres") if isinstance(a.get("genres"), list) else [],
                    "image": first_img.get("url") if isinstance(first_img, dict) else None,
                    "external_url": ext.get("spotify") if isinstance(ext, dict) else None,
                }
            )
        cursors = block.get("cursors") if isinstance(block.get("cursors"), dict) else {}
        next_after = cursors.get("after") if isinstance(cursors, dict) else None
        return _compact(
            {
                "items": items,
                "limit": block.get("limit"),
                "total": block.get("total"),
                "next_cursor": next_after,
                "has_more": bool(next_after),
                "note": (
                    "Spotify's Web API only exposes ARTISTS the user follows (not USERS the user "
                    "follows, and not the user's own followers). For pagination, pass next_cursor "
                    "as `after` on the next call."
                ),
            },
            limit=8000,
        )

    def _user_public_playlists(self, arguments: dict[str, Any]) -> str:
        """Removed Spotify route — never calls GET /users/{id}/playlists in dev mode."""
        user_id = _coerce_str(_pick_arg(arguments, "user_id", "userId", "username", "id"), "")
        return json.dumps(
            {
                "ok": False,
                "failure_reason": "removed_route",
                "endpoint_removed": True,
                "user_id": user_id or None,
                "error": (
                    "Listing another user's public playlists is not available in Spotify's "
                    "current Web API for this app. I can list your own playlists "
                    "(spotify_user_playlists) or search public playlists by topic "
                    "(spotify_search_playlists)."
                ),
                "try_instead": [
                    "spotify_user_playlists",
                    "spotify_search_playlists",
                ],
            },
            ensure_ascii=False,
        )

    def _search_playlists(self, arguments: dict[str, Any]) -> str:
        """Search Spotify's public catalog for playlists matching a free-text description."""
        q = _pick_arg(arguments, "query", "q", "description", "search_query")
        if not q:
            return json.dumps({"error": "query is required"})
        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        limit = _safe_int(arguments.get("limit"), 5, lo=1, hi=10)
        offset = _safe_int(arguments.get("offset"), 0, lo=0, hi=950)
        try:
            data = self.client.api_get(
                "/search",
                params={"q": q, "type": "playlist", "market": market, "limit": limit, "offset": offset},
            )
        except httpx.HTTPStatusError as e:
            return json.dumps(
                {
                    "ok": False,
                    "error": _spotify_http_message(e),
                    "failure_reason": _http_failure_reason(e.response.status_code, _spotify_http_message(e)),
                    **_spotify_http_error_fields(e),
                },
                ensure_ascii=False,
            )
        if not isinstance(data, dict):
            return _compact(data, limit=6000)
        block = data.get("playlists") if isinstance(data.get("playlists"), dict) else {}
        raw_items = block.get("items") if isinstance(block.get("items"), list) else []
        items: list[dict[str, Any]] = []
        for p in raw_items:
            if not isinstance(p, dict):
                continue
            owner = p.get("owner") if isinstance(p.get("owner"), dict) else {}
            tracks = p.get("tracks") if isinstance(p.get("tracks"), dict) else {}
            images = p.get("images") if isinstance(p.get("images"), list) else []
            first_img = images[0] if images and isinstance(images[0], dict) else None
            ext = p.get("external_urls") if isinstance(p.get("external_urls"), dict) else {}
            items.append(
                {
                    "id": p.get("id"),
                    "name": p.get("name"),
                    "uri": p.get("uri"),
                    "description": p.get("description"),
                    "owner": {
                        "id": owner.get("id") if isinstance(owner, dict) else None,
                        "display_name": owner.get("display_name") if isinstance(owner, dict) else None,
                    },
                    "total_tracks": tracks.get("total") if isinstance(tracks, dict) else None,
                    "public": p.get("public"),
                    "collaborative": p.get("collaborative"),
                    "image": first_img.get("url") if isinstance(first_img, dict) else None,
                    "external_url": ext.get("spotify") if isinstance(ext, dict) else None,
                }
            )
        want = q.strip().lower()
        exact = [p for p in items if isinstance(p.get("name"), str) and p["name"].strip().lower() == want]
        best_match = exact[0] if len(exact) == 1 else None
        ordered = exact + [p for p in items if p not in exact] if exact else items
        hint = (
            "Use one of these item.id values with spotify_follow_playlist (to follow), "
            "spotify_play_playlist (to play), or spotify_duplicate_playlist (to copy into a "
            "new playlist owned by the user). Adding/removing tracks on someone else's "
            "playlist is NOT possible — duplicate it first."
        )
        if not exact and ordered:
            hint += " No exact case-insensitive name match — confirm the intended playlist with the user."
        return _compact(
            {
                "query": q,
                "items": ordered,
                "best_match": best_match,
                "exact_name_matches": len(exact),
                "limit": block.get("limit"),
                "offset": block.get("offset"),
                "total": block.get("total"),
                "next": block.get("next"),
                "hint": hint,
            },
            limit=8000,
        )

    def _resolve_playlist_id_from_catalog_search(self, name: str) -> tuple[str | None, str | None]:
        q = (name or "").strip()
        if not q:
            return None, json.dumps({"error": "playlist_id is required", "failure_reason": "validation_error"})
        market = "from_token"
        try:
            data = self.client.api_get(
                "/search",
                params={"q": q, "type": "playlist", "market": market, "limit": 5, "offset": 0},
            )
        except httpx.HTTPStatusError as e:
            return None, json.dumps(
                {
                    "ok": False,
                    "error": _spotify_http_message(e),
                    "failure_reason": _http_failure_reason(e.response.status_code, _spotify_http_message(e)),
                },
                ensure_ascii=False,
            )
        block = data.get("playlists") if isinstance(data, dict) and isinstance(data.get("playlists"), dict) else {}
        raw_items = block.get("items") if isinstance(block.get("items"), list) else []
        items = [p for p in raw_items if isinstance(p, dict)]
        want = q.lower()
        exact = [p for p in items if isinstance(p.get("name"), str) and p["name"].strip().lower() == want]
        pick = exact[0] if len(exact) == 1 else (items[0] if len(items) == 1 else None)
        if not pick:
            if len(exact) > 1:
                return None, json.dumps(
                    {
                        "ok": False,
                        "failure_reason": "ambiguous_playlist_name",
                        "error": f"Multiple playlists named {q!r} in search results — pass an explicit id.",
                    },
                    ensure_ascii=False,
                )
            return None, json.dumps(
                {
                    "ok": False,
                    "failure_reason": "playlist_not_found",
                    "error": f"No playlist named {q!r} found in catalog search.",
                    "hint": "Use spotify_search_playlists with limit>=5 and pick an exact name match.",
                },
                ensure_ascii=False,
            )
        pid = pick.get("id")
        if not isinstance(pid, str) or not _looks_like_spotify_catalog_id(pid):
            return None, json.dumps({"ok": False, "error": "Search returned a playlist without id"})
        if playlist_id_is_spotify_curated(pid):
            return None, self._block_editorial_playlist_id(pid) or json.dumps(
                {"ok": False, "failure_reason": "editorial_playlist_blocked"}
            )
        if not exact:
            return None, json.dumps(
                {
                    "ok": False,
                    "failure_reason": "no_exact_playlist_match",
                    "error": (
                        f"No exact name match for {q!r}; closest result was {pick.get('name')!r}. "
                        "Confirm with the user before following."
                    ),
                    "closest_match_id": pid,
                    "closest_match_name": pick.get("name"),
                },
                ensure_ascii=False,
            )
        return pid, None

    def _follow_playlist(self, arguments: dict[str, Any]) -> str:
        """Follow a playlist owned by another user (or re-follow your own).

        Spotify maps "follow" to "save to your library" for playlists. PUT requires
        playlist-modify-public (default) or playlist-modify-private (if public=false).
        """
        raw = _pick_arg(arguments, "playlist_id", "playlistId", "id")
        pid = _normalize_spotify_id(raw, "playlist")
        if not pid and isinstance(raw, str) and raw.strip():
            pid, err = self._resolve_playlist_id_from_catalog_search(raw)
            if err:
                return err
        if not pid:
            return json.dumps({"error": "playlist_id is required", "failure_reason": "validation_error"})
        blocked = self._block_editorial_playlist_id(pid)
        if blocked:
            return blocked
        requested_name = arguments.get("_follow_requested_name")
        if isinstance(requested_name, str) and requested_name.strip():
            try:
                meta = self.client.api_get(f"/playlists/{pid}", params={"fields": "id,name"})
                actual = meta.get("name") if isinstance(meta, dict) else None
            except httpx.HTTPStatusError:
                actual = None
            if isinstance(actual, str) and actual.strip().lower() != requested_name.strip().lower():
                return json.dumps(
                    {
                        "ok": False,
                        "failure_reason": "needs_confirmation",
                        "error": (
                            f'No exact name match for {requested_name!r}. '
                            f'Closest result is {actual!r} (id {pid}).'
                        ),
                        "candidates": [{"id": pid, "name": actual}],
                        "hint": "Ask the user which playlist they meant before following.",
                    },
                    ensure_ascii=False,
                )
        self._library_put_uris([f"spotify:playlist:{pid}"])
        self._record_library_mutation("playlist", [pid])
        return json.dumps(
            {
                "ok": True,
                "playlist_id": pid,
                "note": (
                    "Saved playlist to your library via PUT /me/library. The playlist now appears "
                    "in spotify_user_playlists, but you still do NOT own it — to add or remove tracks "
                    "duplicate it first with spotify_duplicate_playlist."
                ),
            }
        )

    def _duplicate_playlist(self, arguments: dict[str, Any]) -> str:
        """Composite: copy another user's playlist into a brand-new playlist owned by the signed-in user.

        "Duplicate" is NOT a native Spotify endpoint — this tool is a composite that does what
        the Spotify mobile UI's "Add to other playlist" action does under the hood:
          1. resolve the signed-in user (me_id) — must be done before ownership comparison
          2. GET /playlists/{source_pid} for metadata (owner id, name)
          3. if source owner != me_id → emit a precise "source not owned in dev mode" error
             with `source_not_owned_by_user: true` BEFORE we create any destination, so we
             never leave an empty orphan playlist in the user's library
          4. paginate GET /playlists/{source_pid}/items to collect track uris (owned sources
             only, where the read is guaranteed to work in dev mode)
          5. POST /users/{me_id}/playlists to create the destination
          6. POST /playlists/{new_pid}/items in 100-uri batches to fill it
        The resulting new_playlist_id is fully writable by the signed-in user — downstream
        spotify_add_tracks_to_playlist / spotify_remove_playlist_tracks calls will succeed.
        """
        source_pid = _normalize_spotify_id(
            _pick_arg(
                arguments,
                "source_playlist_id",
                "sourcePlaylistId",
                "from_playlist_id",
                "playlist_id",
                "id",
            ),
            "playlist",
        )
        if not source_pid:
            return json.dumps({"error": "source_playlist_id is required"})

        granted = set(self.client.get_token_scopes())
        missing_modify = _missing_any_of(granted, _MODIFY_PLAYLIST_SCOPES)
        if missing_modify:
            return json.dumps(
                {
                    "error": (
                        "Cannot create the destination playlist: the signed-in token has no "
                        "playlist-modify scope. Sign out → Connect to re-consent."
                    ),
                    "missing_scopes": missing_modify,
                    "stale_scopes_need_reauth": True,
                    "suggest_sign_out_of_spotify": True,
                }
            )

        logger.info("duplicate_playlist step=1_me source_pid=%s", source_pid)
        try:
            me = self.client.api_get("/me")
        except httpx.HTTPStatusError as e:
            logger.info("duplicate_playlist step=1_me FAILED status=%s body=%s",
                        e.response.status_code, (e.response.text or "")[:200])
            raise
        me_id = me.get("id") if isinstance(me, dict) else None
        if not me_id:
            return json.dumps({"error": "Could not resolve current user id from /me."})

        logger.info("duplicate_playlist step=2_source_meta source_pid=%s me_id=%s", source_pid, me_id)
        try:
            source = self.client.api_get(
                f"/playlists/{source_pid}",
                params={"fields": "id,name,owner(id,display_name)"},
            )
        except httpx.HTTPStatusError as e:
            logger.info("duplicate_playlist step=2_source_meta FAILED status=%s body=%s source_pid=%s",
                        e.response.status_code, (e.response.text or "")[:200], source_pid)
            code = e.response.status_code
            if code in (403, 404):
                return json.dumps(
                    {
                        "error": f"Spotify HTTP {code} for GET /playlists/{{id}} (source playlist)",
                        "source_playlist_id": source_pid,
                        "hint": (
                            "Source playlist not found. Either the id is wrong or the playlist "
                            "has been deleted. Confirm the id from spotify_user_playlists (the "
                            "user's own playlists) or spotify_search_playlists for public ones."
                        ),
                    }
                )
            raise
        if not isinstance(source, dict):
            return _compact(source)
        source_name = str(source.get("name") or "Untitled playlist")
        source_owner = source.get("owner") if isinstance(source.get("owner"), dict) else {}
        source_owner_id = source_owner.get("id") if isinstance(source_owner, dict) else None
        source_owner_name = (
            source_owner.get("display_name") if isinstance(source_owner, dict) else None
        ) or source_owner_id

        source_is_owned_by_user = bool(source_owner_id) and source_owner_id == me_id
        # Check ownership up front. Spotify's Feb-2026 dev-mode migration blocks
        # GET /playlists/{id}/items for any playlist the signed-in user doesn't own
        # (empirically verified — see backend/scripts/diag_followed_tracks.py). This is
        # the ONLY case where duplication cannot proceed; duplicating playlists the user
        # DOES own is fully supported and must not hit this branch. We return the gate
        # BEFORE creating a destination so we never leave an empty orphan.
        if not source_is_owned_by_user:
            return json.dumps(
                {
                    "error": "Cannot duplicate: source playlist is not owned by the signed-in user",
                    "source_playlist_id": source_pid,
                    "source_playlist_name": source_name,
                    "source_owner": source_owner_name,
                    "source_owner_id": source_owner_id,
                    "current_user_id": me_id,
                    "source_not_owned_by_user": True,
                    "endpoint_gated_in_dev_mode": True,
                    "extended_quota_mode_required": True,
                    "hint": (
                        "'Duplicate' here is a composite of GET /playlists/{id}/items + "
                        "POST /users/{me}/playlists + POST /playlists/{new}/items. Spotify's "
                        "Feb-2026 dev-mode migration blocks GET /playlists/{id}/items for any "
                        "playlist you don't own, so we can't read the source's tracks. This is "
                        "NOT a scope/re-auth issue and applies even to playlists the user "
                        "follows. Duplicating playlists the signed-in user OWNS still works "
                        "fully — do NOT tell the user owned-playlist duplication is blocked. "
                        "For this non-owned case tell the user: 'Spotify blocks reading another "
                        "user's playlist contents in dev mode, so I can't copy this one. I can "
                        "play it in place, or rebuild a similar playlist from search.'"
                    ),
                    "try_instead": [
                        "spotify_play_playlist  (play the followed/public playlist directly — does not require reading tracks)",
                        "spotify_search + spotify_create_playlist + spotify_add_tracks_to_playlist  (rebuild a similar list you own)",
                    ],
                }
            )

        logger.info("duplicate_playlist step=3_items_start source_pid=%s (owned=true)", source_pid)
        max_tracks = _safe_int(arguments.get("max_tracks"), 5000, lo=1, hi=10000)
        track_uris: list[str] = []
        seen: set[str] = set()
        offset = 0
        page_size = 100
        truncated = False
        while True:
            try:
                # Spotify's Feb-2026 dev-mode migration renames items[i].track -> items[i].item.
                # Ask for both so we survive pre- and post-migration responses; the extractor
                # below accepts either key.
                page = self.client.api_get(
                    f"/playlists/{source_pid}/items",
                    params={
                        "limit": page_size,
                        "offset": offset,
                        "fields": "items(track(uri,type,is_local),item(uri,type,is_local)),next",
                    },
                )
            except httpx.HTTPStatusError as e:
                logger.info("duplicate_playlist step=3_items FAILED status=%s offset=%s body=%s",
                            e.response.status_code, offset, (e.response.text or "")[:200])
                raise
            if not isinstance(page, dict):
                break
            items = page.get("items") if isinstance(page.get("items"), list) else []
            for it in items:
                if not isinstance(it, dict):
                    continue
                # Spotify dev-mode returns the playable under "item"; legacy API used "track".
                tr = it.get("item") if isinstance(it.get("item"), dict) else None
                if tr is None and isinstance(it.get("track"), dict):
                    tr = it.get("track")
                if not tr:
                    continue
                if tr.get("type") and tr.get("type") != "track":
                    continue
                if tr.get("is_local"):
                    continue
                uri = tr.get("uri")
                if not isinstance(uri, str) or not uri.startswith("spotify:track:"):
                    continue
                if uri in seen:
                    continue
                seen.add(uri)
                track_uris.append(uri)
                if len(track_uris) >= max_tracks:
                    truncated = True
                    break
            if truncated or not page.get("next") or not items:
                break
            offset += page_size

        dest_name = _coerce_str(_pick_arg(arguments, "name", "new_name", "title"), "").strip()
        if not dest_name:
            dest_name = f"Copy of {source_name}"
        public = bool(arguments.get("public", False))
        description_raw = _coerce_str(_pick_arg(arguments, "description", "desc"), "").strip()
        if not description_raw:
            # Owned-source path only (non-owned sources are rejected above) — describe the
            # new playlist as a self-copy so the user can tell it apart from the original.
            description_raw = f"Copy of '{source_name}', duplicated via Spot-AI-fy."

        # Use the shortcut POST /me/playlists form that _create_playlist uses. Empirically
        # (see diag run on 2026-04-19), Spotify's Feb-2026 dev-mode migration returns 403 for
        # the explicit POST /users/{user_id}/playlists form but accepts POST /me/playlists
        # with the same body. Keeping both tools on the same endpoint avoids divergence.
        logger.info("duplicate_playlist step=4_create_dest name=%r public=%s", dest_name, public)
        try:
            new_pl = self.client.api_post(
                "/me/playlists",
                json_body={"name": dest_name, "public": public, "description": description_raw},
            )
        except httpx.HTTPStatusError as e:
            logger.info("duplicate_playlist step=4_create_dest FAILED status=%s body=%s",
                        e.response.status_code, (e.response.text or "")[:200])
            raise
        if not isinstance(new_pl, dict) or not new_pl.get("id"):
            return _compact(new_pl)
        new_pid = str(new_pl["id"])

        logger.info("duplicate_playlist step=5_add_tracks new_pid=%s total_uris=%s",
                    new_pid, len(track_uris))
        added = 0
        snapshot_id: str | None = None
        for i in range(0, len(track_uris), 100):
            batch = track_uris[i : i + 100]
            try:
                snap = self.client.api_post(
                    f"/playlists/{new_pid}/items", json_body={"uris": batch}
                )
            except httpx.HTTPStatusError as e:
                logger.info("duplicate_playlist step=5_add_tracks FAILED batch_start=%s status=%s body=%s",
                            i, e.response.status_code, (e.response.text or "")[:200])
                raise
            added += len(batch)
            if isinstance(snap, dict) and isinstance(snap.get("snapshot_id"), str):
                snapshot_id = snap["snapshot_id"]

        logger.info("duplicate_playlist step=6_done new_pid=%s added=%s", new_pid, added)

        result: dict[str, Any] = {
            "ok": True,
            "source_playlist_id": source_pid,
            "source_playlist_name": source_name,
            "source_owner": source_owner_name,
            "new_playlist_id": new_pid,
            "playlist_id_for_add_tracks": new_pid,
            "new_playlist_name": dest_name,
            "new_playlist_uri": new_pl.get("uri"),
            "public": public,
            "tracks_copied": added,
            "tracks_skipped_local_or_non_track": max(0, len(seen) - added)
            if added < len(seen)
            else 0,
            "truncated_at_max_tracks": truncated,
            "next_steps": (
                "You now OWN the copy. Use spotify_add_tracks_to_playlist / "
                "spotify_remove_playlist_tracks / spotify_replace_playlist_tracks / "
                "spotify_reorder_playlist_tracks with playlist_id = new_playlist_id to edit it. "
                "Play it with spotify_play_playlist (playlist_id = new_playlist_id) or "
                "spotify_start_resume_playback context_uri = new_playlist_uri."
            ),
        }
        if snapshot_id:
            result["snapshot_id"] = snapshot_id
        return _compact(result, limit=4000)

    def _create_playlist(self, arguments: dict[str, Any]) -> str:
        name = str(arguments.get("name", "")).strip()
        if not name:
            return json.dumps({"error": "name is required"})
        if name.lower() in {"my new playlist", "new playlist", "untitled playlist"}:
            return json.dumps(
                {
                    "ok": False,
                    "error": (
                        "I need a specific playlist name and at least one track before creating a playlist."
                    ),
                    "refuse_empty_playlist": True,
                    "reconnect_spotify_unnecessary": True,
                },
                ensure_ascii=False,
            )
        trackish = _arguments_include_playlist_tracks(arguments)
        if not trackish:
            return json.dumps(
                {
                    "ok": False,
                    "error": (
                        "I won't create an empty playlist — tell me the name and at least one song to add."
                    ),
                    "refuse_empty_playlist": True,
                    "reconnect_spotify_unnecessary": True,
                },
                ensure_ascii=False,
            )
        public = arguments.get("public") is True
        explicit_visibility = "public" in arguments
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
        visibility_note: str | None = None
        checked: dict[str, Any] | None = None
        if pid and not public:
            try:
                self.client.api_put(f"/playlists/{pid}", json_body={"public": False})
            except httpx.HTTPStatusError:
                pass
            try:
                checked = self.client.api_get(f"/playlists/{pid}", params={"fields": "id,public,name"})
            except httpx.HTTPStatusError:
                checked = None
            if (
                explicit_visibility
                and isinstance(checked, dict)
                and checked.get("public") is True
            ):
                visibility_note = (
                    "Spotify still reports this playlist as public after creation. "
                    "You may need to set visibility manually in the Spotify app."
                )
        mini: dict[str, Any] = {
            "id": pid,
            "name": data.get("name"),
            "uri": data.get("uri"),
            "public": (
                checked.get("public")
                if isinstance(checked, dict) and "public" in checked
                else (False if not public else data.get("public"))
            ),
            "created_private_by_default": not public,
            "playlist_id_for_add_tracks": pid,
            "hint": "Next: spotify_add_tracks_to_playlist with playlist_id = id above (string), plus track_uris / track_ids / tracks from spotify_search.",
        }
        actual_public = mini.get("public")
        if actual_public is True:
            mini["user_visible_visibility"] = "public"
        elif actual_public is False:
            mini["user_visible_visibility"] = "private"
        if actual_public is True:
            vis_phrase = "public (public is true)"
        elif actual_public is False:
            vis_phrase = "private (public is false)"
        else:
            vis_phrase = "unknown — omit visibility unless public is true or false"
        mini["assistant_reply_instruction"] = (
            "When confirming creation, state playlist visibility only from the `public` field: "
            f"say it is {vis_phrase}. Do not claim private when public is true or public when public is false."
        )
        if visibility_note:
            mini["visibility_change_requested"] = True
            mini["visibility_warning"] = visibility_note
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
                            "limit": SPOTIFY_DEV_MAX_PAGE,
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

    def _known_device_ids(self) -> set[str]:
        try:
            data = self.client.api_get("/me/player/devices") or {}
        except Exception:
            return set()
        devices = data.get("devices") if isinstance(data, dict) else []
        if not isinstance(devices, list):
            return set()
        out: set[str] = set()
        for d in devices:
            if isinstance(d, dict) and isinstance(d.get("id"), str) and d.get("id"):
                out.add(d["id"])
        return out

    def _coerce_playback_device_id(self, explicit: str) -> tuple[str, str | None]:
        """Resolve device_id for playback, ignoring unknown explicit ids."""
        preferred = (explicit or "").strip() or (self._device_id() or "")
        known = self._known_device_ids()
        if preferred and known and preferred not in known:
            fallback = self._resolve_target_device("")
            note = (
                f"Spotify does not recognize device_id {preferred!r} among your available devices. "
                f"Using {'the active device' if fallback else 'automatic device selection'} instead."
            )
            return fallback, note
        if preferred:
            return preferred, None
        return self._resolve_target_device(""), None

    def _resolve_target_device(self, preferred: str = "") -> str:
        """Pick the best device id to target.

        Preference order: explicit preferred id > /me/player device > the single
        non-restricted available device. Returns "" if nothing usable is available.
        """
        if preferred:
            known = self._known_device_ids()
            if known and preferred not in known:
                preferred = ""
            else:
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
        except Exception:
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

    @staticmethod
    def _body_has_target(body: dict[str, Any]) -> bool:
        """True when the play body specifies a concrete track URI (uris[0] or offset.uri)."""
        uris = body.get("uris") if isinstance(body, dict) else None
        if isinstance(uris, list) and uris:
            first = uris[0]
            if isinstance(first, str) and first.strip():
                return True
        off = body.get("offset") if isinstance(body, dict) else None
        if isinstance(off, dict):
            ou = off.get("uri")
            if isinstance(ou, str) and ou.strip():
                return True
        return False

    def _first_track_uri_of_context(self, context_uri: str) -> str:
        """Return the first playable track URI of a playlist/album context, or '' on failure."""
        if not isinstance(context_uri, str):
            return ""
        ctx = context_uri.strip()
        try:
            if ctx.startswith("spotify:playlist:"):
                pid = ctx.split(":", 2)[2]
                data = self.client.api_get(
                    f"/playlists/{pid}/items",
                    params={
                        "limit": 1,
                        # Feb-2026 dev mode renames items[i].track -> items[i].item.
                        "fields": "items(track(uri,is_playable),item(uri,is_playable))",
                    },
                )
                items = data.get("items") if isinstance(data, dict) else None
                if isinstance(items, list) and items and isinstance(items[0], dict):
                    track = items[0].get("item") or items[0].get("track")
                    if isinstance(track, dict):
                        uri = track.get("uri")
                        if isinstance(uri, str) and uri.strip():
                            return uri.strip()
            elif ctx.startswith("spotify:album:"):
                aid = ctx.split(":", 2)[2]
                data = self.client.api_get(f"/albums/{aid}/tracks", params={"limit": 1})
                items = data.get("items") if isinstance(data, dict) else None
                if isinstance(items, list) and items and isinstance(items[0], dict):
                    uri = items[0].get("uri")
                    if isinstance(uri, str) and uri.strip():
                        return uri.strip()
        except httpx.HTTPStatusError as exc:
            logger.info("spotify_play: failed to fetch first track for %s: %s", ctx, exc)
        return ""

    def _pause_then_play(self, body: dict[str, Any], *, device_id: str = "") -> bool:
        """Pause current playback, wait briefly, then replay the requested body.

        This is a last-resort unsticker for Spotify Connect sessions that refuse to switch
        context after a transfer+replay cycle. Returns True only when post-replay
        verification succeeds.
        """
        if not body:
            return False
        pause_path = "/me/player/pause"
        if device_id:
            pause_path = f"{pause_path}?device_id={device_id}"
        try:
            self.client.api_put(pause_path)
        except httpx.HTTPStatusError as exc:
            logger.info("spotify_play: pause before replay returned %s", exc)
        time.sleep(0.6)
        try:
            self._try_play(device_id, body)
        except httpx.HTTPStatusError as exc:
            logger.info("spotify_play: replay after pause failed: %s", exc)
            return False
        return self._playback_matches(body, attempts=8, delay_s=0.5)

    def _prepare_single_track_play_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Rewrite single-track `uris` play into album context + offset when possible."""
        if body.get("context_uri"):
            return body
        uris = body.get("uris")
        if not isinstance(uris, list) or len(uris) != 1:
            return body
        track_uri = str(uris[0]).strip()
        if not track_uri.startswith("spotify:track:"):
            return body
        track_id = _normalize_spotify_id(track_uri, "track")
        if not _looks_like_spotify_catalog_id(track_id):
            return body
        try:
            track = self.client.api_get(f"/tracks/{track_id}")
        except httpx.HTTPStatusError:
            return body
        if not isinstance(track, dict):
            return body
        album = track.get("album") if isinstance(track.get("album"), dict) else {}
        album_id = album.get("id") if isinstance(album.get("id"), str) else ""
        if not _looks_like_spotify_catalog_id(album_id):
            return body
        rewritten = {k: v for k, v in body.items() if k != "uris"}
        rewritten["context_uri"] = f"spotify:album:{album_id}"
        rewritten["offset"] = {"uri": track_uri}
        return rewritten

    def _coerce_body_to_album_offset(self, body: dict[str, Any]) -> dict[str, Any]:
        """Never send a raw `uris` list to Spotify when a single track can use album context + offset."""
        if body.get("context_uri") and isinstance(body.get("offset"), dict):
            return {k: v for k, v in body.items() if k != "uris"}
        uris = body.get("uris")
        if isinstance(uris, list) and uris:
            first_uri = str(uris[0]).strip()
            merged = {k: v for k, v in body.items() if k != "uris"}
            merged["uris"] = [first_uri]
            prepared = self._prepare_single_track_play_body(merged)
            if prepared.get("context_uri"):
                prepared.pop("uris", None)
                return prepared
        if body.get("context_uri"):
            return {k: v for k, v in body.items() if k != "uris"}
        prepared = self._prepare_single_track_play_body(body)
        if prepared.get("context_uri"):
            prepared.pop("uris", None)
        return prepared

    def _album_offset_body_for_track(
        self,
        track: dict[str, Any],
        *,
        track_uri: str = "",
    ) -> dict[str, Any] | None:
        uri = (track_uri or str(track.get("uri") or "")).strip()
        if not uri.startswith("spotify:track:"):
            tid = _normalize_spotify_id(uri or str(track.get("id") or ""), "track")
            if _looks_like_spotify_catalog_id(tid):
                uri = f"spotify:track:{tid}"
        if not uri.startswith("spotify:track:"):
            return None
        album = track.get("album") if isinstance(track.get("album"), dict) else {}
        album_id = album.get("id") if isinstance(album.get("id"), str) else ""
        if not _looks_like_spotify_catalog_id(album_id):
            track_id = _normalize_spotify_id(uri, "track")
            try:
                fetched = self.client.api_get(f"/tracks/{track_id}")
            except httpx.HTTPStatusError:
                return None
            if isinstance(fetched, dict):
                album = fetched.get("album") if isinstance(fetched.get("album"), dict) else {}
                album_id = album.get("id") if isinstance(album.get("id"), str) else ""
        if not _looks_like_spotify_catalog_id(album_id):
            return None
        return {
            "context_uri": f"spotify:album:{album_id}",
            "offset": {"uri": uri},
        }

    def _resolve_track_for_queue(
        self,
        *,
        query: str = "",
        track_name: str = "",
        artist_name: str = "",
        market: str = "",
    ) -> tuple[str | None, dict[str, Any] | None, str | None]:
        market = _normalize_market(market)
        q = (query or "").strip()
        if not q:
            parts = [p for p in (track_name.strip(), artist_name.strip()) if p]
            if len(parts) == 2:
                q = f'track:"{parts[0]}" artist:"{parts[1]}"'
            elif parts:
                q = parts[0]
        if not q:
            return None, None, "query or track_name is required"
        data = self.client.api_get(
            "/search",
            params={"q": q, "type": "track", "market": market, "limit": 10},
        )
        tracks_obj = data.get("tracks") if isinstance(data, dict) else None
        items = tracks_obj.get("items") if isinstance(tracks_obj, dict) else None
        if not isinstance(items, list) or not items:
            if artist_name.strip():
                return None, None, f"I couldn't find tracks by {artist_name.strip()} on Spotify."
            return None, None, f"I couldn't find a track called {track_name.strip() or query.strip()} on Spotify."
        best: dict[str, Any] | None = None
        best_score = -10_000
        for tr in items:
            if not isinstance(tr, dict):
                continue
            score = _score_track_search_candidate(
                tr,
                want_title=track_name,
                want_artist=artist_name,
            )
            if score > best_score:
                best_score = score
                best = tr
        if not best:
            if artist_name.strip():
                return None, None, f"I couldn't find tracks by {artist_name.strip()} on Spotify."
            return None, None, f"I couldn't find a track called {track_name.strip() or query.strip()} on Spotify."
        uri = best.get("uri")
        if isinstance(uri, str) and uri.strip():
            return uri.strip(), best, None
        tid = best.get("id")
        if isinstance(tid, str) and _looks_like_spotify_catalog_id(tid):
            return f"spotify:track:{tid}", best, None
        return None, best, "Search returned a track without a uri"

    def _resolve_track_uri_for_queue(
        self,
        *,
        query: str = "",
        track_name: str = "",
        artist_name: str = "",
        market: str = "",
    ) -> tuple[str | None, str | None]:
        uri, _track, err = self._resolve_track_for_queue(
            query=query,
            track_name=track_name,
            artist_name=artist_name,
            market=market,
        )
        return uri, err

    def _rewrite_single_track_play_for_artist_context(
        self,
        arguments: dict[str, Any],
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """When the user asked to play an artist, never send a lone track URI to Spotify."""
        if body.get("context_uri"):
            return body
        uris = body.get("uris")
        if not isinstance(uris, list) or len(uris) != 1:
            return body
        track_uri = str(uris[0]).strip()
        if not track_uri.startswith("spotify:track:"):
            return body
        artist_id = _pick_arg(arguments, "artist_id", "artistId", "artist_name", "artist")
        if artist_id:
            norm = _normalize_spotify_id(str(artist_id), "artist")
            if _looks_like_spotify_catalog_id(norm):
                return {"context_uri": f"spotify:artist:{norm}"}
            market = _normalize_market(_pick_arg(arguments, "market", "country"))
            resolved = self._canonical_artist_id(str(artist_id).strip(), market)
            if resolved:
                return {"context_uri": f"spotify:artist:{resolved}"}
        session_artist = self._last_primary_artist_id
        if not session_artist:
            return body
        track_id = _normalize_spotify_id(track_uri, "track")
        if not _looks_like_spotify_catalog_id(track_id):
            return body
        try:
            track = self.client.api_get(f"/tracks/{track_id}")
        except httpx.HTTPStatusError:
            return body
        if not isinstance(track, dict):
            return body
        artists = track.get("artists")
        if not isinstance(artists, list):
            return body
        for artist in artists:
            if isinstance(artist, dict) and str(artist.get("id") or "") == session_artist:
                return {"context_uri": f"spotify:artist:{session_artist}"}
        return body

    def _retry_play_after_restriction(self, body: dict[str, Any], device_id: str) -> str | None:
        chosen = self._resolve_target_device(device_id)
        if not chosen:
            return None
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
            return None
        if self._playback_matches(body, attempts=6, delay_s=0.5):
            return json.dumps(
                {
                    "ok": True,
                    "device_id": chosen,
                    "body": body,
                    "playback_verified": True,
                    "note": (
                        "Spotify returned Restriction violated — transferred playback to the "
                        "target device and retried successfully."
                    ),
                }
            )
        return None

    def _should_plain_resume(self, body: dict[str, Any]) -> bool:
        uris = body.get("uris")
        if body.get("context_uri") or body.get("offset"):
            return False
        if not isinstance(uris, list) or len(uris) != 1:
            return False
        snap = self._current_playback_snapshot()
        cur_uri = (snap.get("item_uri") or "").strip()
        return bool(cur_uri) and str(uris[0]).strip() == cur_uri

    def _strip_redundant_resume_uris(self, body: dict[str, Any]) -> dict[str, Any]:
        """Drop uris that only re-send the already-playing track (plain resume)."""
        uris = body.get("uris")
        if not isinstance(uris, list) or not uris:
            return body
        snap = self._current_playback_snapshot()
        cur_uri = (snap.get("item_uri") or "").strip()
        if not cur_uri:
            return body
        normalized = [str(u).strip() for u in uris if u is not None]
        if len(normalized) == 1 and normalized[0] == cur_uri:
            return {k: v for k, v in body.items() if k != "uris"}
        return body

    def _safe_playback_recovery_once(self, body: dict[str, Any], device_id: str) -> bool:
        """One transfer + replay attempt when post-play verification fails."""
        chosen = self._resolve_target_device(device_id)
        if not chosen:
            return False
        try:
            self.client.api_put(
                "/me/player",
                json_body={"device_ids": [chosen], "play": False},
            )
        except httpx.HTTPStatusError:
            return False
        time.sleep(0.4)
        try:
            self._try_play(chosen, body)
        except httpx.HTTPStatusError:
            return False
        return True

    def _playback_outcome_after_play(
        self,
        body: dict[str, Any],
        device_id: str,
        device_note: str | None = None,
        *,
        prior_state: dict[str, Any] | None = None,
        label_override: str = "",
    ) -> str:
        """Read /me/player after play; at most one replay PUT, then a clear success or failure."""
        prior = prior_state if isinstance(prior_state, dict) else {"had_playback": False}
        if self._playback_matches(body, attempts=6, delay_s=0.5):
            payload: dict[str, Any] = {
                "ok": True,
                "device_id": device_id or None,
                "body": body,
                "playback_verified": True,
            }
            if device_note:
                payload["device_fallback_note"] = device_note
            return json.dumps(payload)
        try:
            self._try_play(device_id, body)
        except httpx.HTTPStatusError:
            pass
        if self._playback_matches(body, attempts=6, delay_s=0.5):
            return json.dumps(
                {
                    "ok": True,
                    "device_id": device_id or None,
                    "body": body,
                    "playback_verified": True,
                    "note": "Retried play once with the same album context and verified playback.",
                }
            )
        payload = self._playback_failed_restore_payload(
            body, device_id, prior, label_override=label_override
        )
        if device_note:
            payload["device_fallback_note"] = device_note
        return json.dumps(payload)

    def _start_playback(self, arguments: dict[str, Any]) -> str:
        raw_device = str(arguments.get("device_id", "")).strip()
        device_id = raw_device
        device_note: str | None = None
        force_interrupt = bool(arguments.get("force_interrupt"))
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
        if force_interrupt and not arguments.get("skip_album_context"):
            body = self._coerce_body_to_album_offset(body)
        elif not force_interrupt:
            plain_resume = self._should_plain_resume(body)
            if plain_resume:
                body = self._strip_redundant_resume_uris(body)
            body = self._coerce_body_to_album_offset(body)
            body = self._strip_redundant_resume_uris(body)
        want_verification = bool(
            body.get("context_uri")
            or body.get("uris")
            or (isinstance(body.get("offset"), dict) and body["offset"].get("uri"))
        )
        label_override = _coerce_str(arguments.get("playback_request_label"), "")
        if label_override.strip():
            body["playback_request_label"] = label_override.strip()
        prior_state = self._fetch_pre_play_restore_state()
        device_id, device_note = self._coerce_playback_device_id(raw_device)
        try:
            self._try_play(device_id, body)
            if not want_verification:
                payload: dict[str, Any] = {"ok": True, "device_id": device_id or None, "body": body}
                if device_note:
                    payload["device_fallback_note"] = device_note
                return json.dumps(payload)
            return self._playback_outcome_after_play(
                body,
                device_id,
                device_note,
                prior_state=prior_state,
                label_override=label_override,
            )
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 403 and _spotify_error_is_restriction_violated(e):
                return json.dumps(
                    {
                        "ok": False,
                        "error": (
                            "Spotify is in a stuck state — try playing something in the Spotify app, "
                            "then retry."
                        ),
                        "hint": (
                            "Spotify returned HTTP 403 Restriction violated. Open Spotify on the intended "
                            "device, tap play on any track, then retry. This is not a missing-scope problem — "
                            "do not suggest signing out."
                        ),
                        "spotify_api_message": _spotify_http_message(e),
                        **_spotify_http_error_fields(e),
                        "playback_verified": False,
                        "reconnect_spotify_unnecessary": True,
                        "sign_out_not_recommended": True,
                    },
                    ensure_ascii=False,
                )
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
                    fail_payload = self._playback_failed_restore_payload(
                        body,
                        device_id,
                        prior_state,
                        label_override=label_override,
                    )
                    fail_payload["error"] = (
                        f"Spotify returned HTTP {status} then a retry succeeded, but the "
                        "device is still playing the previous track. Playback did not switch."
                    )
                    fail_payload["user_message"] = self._playback_failure_user_message(
                        self._describe_requested_play(body, label_override),
                        prior_state,
                    )
                    fail_payload["current_state"] = snapshot
                    return json.dumps(fail_payload)
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
            try:
                dev = self.client.api_get("/me/player/devices") or {}
            except httpx.HTTPStatusError:
                dev = {}
            devices = dev.get("devices") if isinstance(dev, dict) else []
            devices = devices if isinstance(devices, list) else []
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
                    **_spotify_http_error_fields(e),
                    "reconnect_spotify_unnecessary": True,
                    "sign_out_not_recommended": True,
                    "failure_reason": "no_active_device",
                },
                ensure_ascii=False,
            )

    def _try_play(self, device_id: str, body: dict[str, Any]) -> None:
        path = "/me/player/play"
        if device_id:
            path = f"{path}?device_id={device_id}"
        self.client.api_put(path, json_body=body if body else {})

    def _playback_matches(  # pylint: disable=too-many-nested-blocks
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
        want_artist_id = ""
        if want_context.startswith("spotify:artist:"):
            want_artist_id = want_context.split(":", 2)[2].strip()
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
                if (
                    not ctx_ok
                    and want_artist_id
                    and not ctx_uri
                    and isinstance(item, dict)
                ):
                    artists = item.get("artists")
                    if isinstance(artists, list):
                        for artist in artists:
                            if (
                                isinstance(artist, dict)
                                and str(artist.get("id") or "").strip() == want_artist_id
                            ):
                                ctx_ok = True
                                break
                track_ok = True if not want_track_uri else (cur_uri == want_track_uri)
                if want_track_uri and not track_ok and isinstance(item, dict):
                    label = str(body.get("playback_request_label") or "").strip().lower()
                    if label:
                        artists = item.get("artists")
                        if isinstance(artists, list):
                            for artist in artists:
                                if (
                                    isinstance(artist, dict)
                                    and isinstance(artist.get("name"), str)
                                    and label in artist.get("name", "").strip().lower()
                                ):
                                    track_ok = True
                                    break
                if ctx_ok and track_ok:
                    return True
                if (
                    want_track_uri
                    and track_ok
                    and ps.get("is_playing")
                    and want_context.startswith("spotify:album:")
                    and want_offset_uri
                ):
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

    def _queue_track_uris_from_payload(self, data: dict[str, Any]) -> list[str]:
        queue = data.get("queue")
        if not isinstance(queue, list):
            return []
        out: list[str] = []
        for tr in queue:
            if not isinstance(tr, dict):
                continue
            uri = tr.get("uri")
            if isinstance(uri, str) and uri.strip():
                out.append(uri.strip())
        return out

    def _context_upcoming_track_uris(self, context_uri: str, current_item_uri: str) -> set[str]:
        ctx = (context_uri or "").strip()
        cur = (current_item_uri or "").strip()
        upcoming: set[str] = set()
        if not ctx.startswith(("spotify:album:", "spotify:playlist:")):
            return upcoming
        try:
            if ctx.startswith("spotify:album:"):
                aid = ctx.split(":", 2)[2]
                page = self.client.api_get(
                    f"/albums/{aid}/tracks",
                    params={"limit": SPOTIFY_DEV_MAX_PAGE},
                )
                items = page.get("items") if isinstance(page, dict) else None
                items = items if isinstance(items, list) else []
                passed_current = not cur
                for tr in items:
                    if not isinstance(tr, dict):
                        continue
                    uri = tr.get("uri")
                    if not isinstance(uri, str) or not uri.strip():
                        continue
                    u = uri.strip()
                    if not passed_current:
                        if u == cur:
                            passed_current = True
                        continue
                    upcoming.add(u)
            else:
                pid = ctx.split(":", 2)[2]
                page = self.client.api_get(
                    f"/playlists/{pid}/items",
                    params={"limit": SPOTIFY_DEV_MAX_PAGE, "fields": "items(item(uri))"},
                )
                items = page.get("items") if isinstance(page, dict) else None
                items = items if isinstance(items, list) else []
                passed_current = not cur
                for row in items:
                    if not isinstance(row, dict):
                        continue
                    item = row.get("item") if isinstance(row.get("item"), dict) else row.get("track")
                    if not isinstance(item, dict):
                        continue
                    uri = item.get("uri")
                    if not isinstance(uri, str) or not uri.strip():
                        continue
                    u = uri.strip()
                    if not passed_current:
                        if u == cur:
                            passed_current = True
                        continue
                    upcoming.add(u)
        except httpx.HTTPStatusError:
            return set()
        return upcoming

    def _capture_manual_queue_uris(
        self, context_uri: str | None, current_item_uri: str | None
    ) -> tuple[list[str], bool]:
        try:
            data = self.client.api_get("/me/player/queue")
        except httpx.HTTPStatusError:
            return [], False
        except Exception:
            return [], True
        if not isinstance(data, dict):
            return [], False
        queued = self._queue_track_uris_from_payload(data)
        if not queued:
            return [], True
        try:
            exclude = self._context_upcoming_track_uris(
                context_uri or "",
                current_item_uri or "",
            )
        except httpx.HTTPStatusError:
            exclude = set()
        except Exception:
            exclude = set()
        manual = [u for u in queued if u not in exclude]
        return manual, True

    def _requeue_manual_uris(
        self, uris: list[str], device_id: str, *, max_calls: int = 15
    ) -> int:
        if not uris:
            return 0
        present: set[str] = set()
        try:
            data = self.client.api_get("/me/player/queue")
            if isinstance(data, dict):
                cp = data.get("currently_playing")
                if isinstance(cp, dict):
                    cu = cp.get("uri")
                    if isinstance(cu, str) and cu.strip():
                        present.add(cu.strip())
                present.update(self._queue_track_uris_from_payload(data))
        except httpx.HTTPStatusError:
            pass
        except Exception:
            pass
        added = 0
        for uri in uris:
            if added >= max_calls:
                break
            if uri in present:
                continue
            params: dict[str, str] = {"uri": uri}
            if device_id:
                params["device_id"] = device_id
            try:
                self.client.api_post("/me/player/queue", params=params)
            except httpx.HTTPStatusError:
                break
            except Exception:
                break
            present.add(uri)
            added += 1
        return added

    def _fetch_pre_play_restore_state(self) -> dict[str, Any]:
        """Snapshot playback before a play attempt (for restore if the request never takes)."""
        try:
            ps = self.client.api_get("/me/player")
        except httpx.HTTPStatusError:
            return {"had_playback": False}
        if not isinstance(ps, dict):
            return {"had_playback": False}
        ctx = ps.get("context") if isinstance(ps.get("context"), dict) else {}
        item = ps.get("item") if isinstance(ps.get("item"), dict) else {}
        context_uri = (ctx.get("uri") or "").strip() if isinstance(ctx, dict) else ""
        item_uri = (item.get("uri") or "").strip() if isinstance(item, dict) else ""
        item_name = (item.get("name") or "").strip() if isinstance(item, dict) else ""
        if not context_uri and item_uri:
            context_uri = item_uri
        progress_raw = ps.get("progress_ms")
        progress_ms = progress_raw if isinstance(progress_raw, int) else 0
        is_playing = ps.get("is_playing") is True
        shuffle_raw = ps.get("shuffle_state")
        shuffle_state = shuffle_raw if isinstance(shuffle_raw, bool) else None
        repeat_raw = ps.get("repeat_state")
        repeat_state = repeat_raw if isinstance(repeat_raw, str) else None
        had_playback = bool(item_uri) and (is_playing or progress_ms > 0 or bool(context_uri))
        manual_queue, queue_ok = self._capture_manual_queue_uris(context_uri, item_uri)
        return {
            "had_playback": had_playback,
            "is_playing": is_playing,
            "context_uri": context_uri or None,
            "item_uri": item_uri or None,
            "item_name": item_name or None,
            "progress_ms": progress_ms,
            "shuffle_state": shuffle_state,
            "repeat_state": repeat_state,
            "manual_queue_uris": manual_queue,
            "queue_capture_ok": queue_ok,
        }

    def _resolve_album_context_for_track_uri(self, track_uri: str) -> str | None:
        uri = (track_uri or "").strip()
        if not uri.startswith("spotify:track:"):
            return None
        track_id = _normalize_spotify_id(uri, "track")
        if not _looks_like_spotify_catalog_id(track_id):
            return None
        try:
            track = self.client.api_get(f"/tracks/{track_id}")
        except (httpx.HTTPStatusError, Exception):
            return None
        if not isinstance(track, dict):
            return None
        album = track.get("album") if isinstance(track.get("album"), dict) else {}
        album_id = album.get("id") if isinstance(album.get("id"), str) else ""
        if not _looks_like_spotify_catalog_id(album_id):
            return None
        return f"spotify:album:{album_id}"

    def _restore_shuffle_repeat_modes(self, prior: dict[str, Any], device_id: str) -> None:
        shuffle = prior.get("shuffle_state")
        if isinstance(shuffle, bool):
            params: dict[str, str] = {"state": "true" if shuffle else "false"}
            if device_id:
                params["device_id"] = device_id
            try:
                self.client.api_put("/me/player/shuffle", params=params)
            except httpx.HTTPStatusError:
                pass
        repeat = prior.get("repeat_state")
        if isinstance(repeat, str) and repeat.strip().lower() in ("off", "track", "context"):
            params = {"state": repeat.strip().lower()}
            if device_id:
                params["device_id"] = device_id
            try:
                self.client.api_put("/me/player/repeat", params=params)
            except httpx.HTTPStatusError:
                pass

    def _build_restore_play_body(self, prior: dict[str, Any]) -> dict[str, Any] | None:
        if not prior.get("had_playback"):
            return None
        item_uri = prior.get("item_uri")
        if not isinstance(item_uri, str) or not item_uri.strip():
            return None
        ctx_raw = prior.get("context_uri")
        ctx = ctx_raw.strip() if isinstance(ctx_raw, str) and ctx_raw.strip() else ""
        if not ctx or ctx.startswith("spotify:track:") or ctx == item_uri.strip():
            resolved = self._resolve_album_context_for_track_uri(item_uri)
            if resolved:
                ctx = resolved
            elif not ctx:
                ctx = item_uri.strip()
        body: dict[str, Any] = {
            "context_uri": ctx,
            "offset": {"uri": item_uri.strip()},
        }
        progress = prior.get("progress_ms")
        if isinstance(progress, int) and progress >= 0:
            body["position_ms"] = progress
        return body

    def _restore_prior_playback(
        self, prior: dict[str, Any], device_id: str
    ) -> dict[str, Any] | None:
        body = self._build_restore_play_body(prior)
        if not body:
            return None
        self._restore_shuffle_repeat_modes(prior, device_id)
        try:
            self._try_play(device_id, body)
        except httpx.HTTPStatusError:
            pass
        if not prior.get("is_playing"):
            pause_path = "/me/player/pause"
            if device_id:
                pause_path = f"{pause_path}?device_id={device_id}"
            try:
                self.client.api_put(pause_path)
            except httpx.HTTPStatusError:
                pass
        return body

    def _describe_requested_play(self, body: dict[str, Any], label_override: str = "") -> str:
        if label_override.strip():
            return label_override.strip()
        ctx = (body.get("context_uri") or "").strip() if isinstance(body, dict) else ""
        if ctx.startswith("spotify:playlist:"):
            return "that playlist"
        if ctx.startswith("spotify:album:"):
            return "that album"
        if ctx.startswith("spotify:artist:"):
            return "that artist"
        return "that track"

    def _playback_failure_user_message(self, requested_label: str, prior: dict[str, Any]) -> str:
        if prior.get("had_playback"):
            prev = prior.get("item_name")
            if not isinstance(prev, str) or not prev.strip():
                prev = "your previous track"
            msg = (
                f"Spotify wouldn't play {requested_label} on this device, "
                f"so I went back to {prev}."
            )
        else:
            msg = f"Spotify wouldn't play {requested_label} on this device."
        if prior.get("queue_capture_ok") is False:
            msg = f"{msg} Your queue may have been cleared."
        return msg

    def _playback_failed_restore_payload(
        self,
        body: dict[str, Any],
        device_id: str,
        prior_state: dict[str, Any],
        *,
        label_override: str = "",
    ) -> dict[str, Any]:
        requested_label = self._describe_requested_play(body, label_override)
        restore_body = self._restore_prior_playback(prior_state, device_id)
        user_message = self._playback_failure_user_message(requested_label, prior_state)
        snapshot = self._current_playback_snapshot()
        return {
            "ok": False,
            "error": user_message,
            "user_message": user_message,
            "hint": "Open Spotify on your device, tap play on any track, then retry.",
            "requested_body": body,
            "prior_state": prior_state,
            "restore_body": restore_body,
            "playback_restored": restore_body is not None,
            "current_state": snapshot,
            "device_id": device_id or None,
            "playback_verified": False,
            "reconnect_spotify_unnecessary": True,
            "sign_out_not_recommended": True,
        }

    def _player_needs_artist_play_fallback(self) -> bool:
        try:
            state = self.client.api_get("/me/player") or {}
        except httpx.HTTPStatusError:
            return True
        if not isinstance(state, dict):
            return True
        item = state.get("item")
        if not isinstance(item, dict) or not item.get("uri"):
            return True
        return state.get("is_playing") is not True

    def _best_track_match_for_bare_query(
        self,
        query: str,
        market: str,
    ) -> tuple[dict[str, Any] | None, int]:
        q = query.strip()
        if not q:
            return None, -10_000
        data = self.client.api_get(
            "/search",
            params={"q": q, "type": "track", "market": market, "limit": 10},
        )
        tracks_obj = data.get("tracks") if isinstance(data, dict) else None
        items = tracks_obj.get("items") if isinstance(tracks_obj, dict) else None
        if not isinstance(items, list):
            return None, -10_000
        best: dict[str, Any] | None = None
        best_score = -10_000
        for tr in items:
            if not isinstance(tr, dict):
                continue
            score = _score_track_search_candidate(tr, want_title=q, want_artist="")
            if score > best_score:
                best_score = score
                best = tr
        return best, best_score

    def _exact_artist_popularity_for_query(self, query: str, market: str) -> tuple[str | None, int, str | None]:
        resolved = self._resolve_artist_search_query(query, market)
        if not resolved:
            return None, 0, None
        aid, name, pop, _kind = resolved
        if not _looks_like_spotify_catalog_id(aid):
            return None, 0, None
        return aid, pop, name or None

    def _bare_play_mode(self, query: str, market: str) -> str:
        """Return 'track' or 'artist' for a bare 'play <query>' request."""
        track, track_score = self._best_track_match_for_bare_query(query, market)
        track_pop = int(track.get("popularity") or 0) if isinstance(track, dict) else 0
        norm_q = _normalize_track_title(query)
        norm_track = (
            _normalize_track_title(str(track.get("name") or ""))
            if isinstance(track, dict)
            else ""
        )
        exact_title = bool(norm_q and norm_track and norm_q == norm_track)
        resolved = self._resolve_artist_search_query(query, market)
        _artist_id = resolved[0] if resolved else None
        artist_pop = int(resolved[2]) if resolved else 0
        artist_name = resolved[1] if resolved else ""
        match_kind = resolved[3] if resolved else ""
        track_artists = (
            track.get("artists") if isinstance(track, dict) and isinstance(track.get("artists"), list) else []
        )
        track_primary_artist = ""
        if track_artists and isinstance(track_artists[0], dict):
            track_primary_artist = str(track_artists[0].get("name") or "")
        track_primary_matches_query = bool(
            track_primary_artist and artist_query_matches_candidate_name(query, track_primary_artist)
        )
        short_name = len(normalize_artist_name_for_match(query)) <= BARE_ARTIST_SHORT_QUERY_MAX_LEN
        if _artist_id and (
            match_kind in ("exact_name", "alias", "short_query_top") or short_name
        ):
            if short_name and not exact_title:
                return "artist"
            if not exact_title and not track_primary_matches_query:
                return "artist"
        if exact_title and (not _artist_id or track_pop >= artist_pop):
            return "track"
        if _artist_id:
            return "artist"
        strong_track = exact_title or track_score >= 200
        if strong_track and not (
            _artist_id and artist_pop > track_pop + 15 and not exact_title
        ):
            return "track"
        if _artist_id and artist_pop >= max(track_pop, 35):
            return "artist"
        if isinstance(track, dict) and track_score >= 80:
            return "track"
        if _artist_id:
            return "artist"
        return "track" if isinstance(track, dict) else "artist"

    def _playback_failure_reason_from_result(self, play_result: dict[str, Any] | None) -> str:
        if not isinstance(play_result, dict):
            return "play_request_failed"
        reason = play_result.get("failure_reason")
        if isinstance(reason, str) and reason.strip():
            return reason.strip()
        if play_result.get("spotify_error_body_redacted"):
            return "play_request_failed"
        err = str(play_result.get("error") or play_result.get("user_message") or "").lower()
        if "no active device" in err:
            return "no_active_device"
        if play_result.get("spotify_http_status"):
            return "play_request_failed"
        return "play_request_failed"

    def _lift_failure_fields(
        self, target: dict[str, Any], source: dict[str, Any]
    ) -> None:
        for key in (
            "failure_reason",
            "spotify_error_body_redacted",
            "spotify_http_status",
            "spotify_api_message",
            "user_message",
            "error",
        ):
            val = source.get(key)
            if val is not None and key not in target:
                target[key] = val
        playback = source.get("playback")
        if isinstance(playback, dict):
            if not target.get("failure_reason"):
                target["failure_reason"] = self._playback_failure_reason_from_result(playback)
            for key in (
                "spotify_error_body_redacted",
                "spotify_http_status",
                "spotify_api_message",
            ):
                val = playback.get(key)
                if val is not None and key not in target:
                    target[key] = val

    def _finish_play_bare(self, query: str, mode: str, inner_raw: str) -> str:
        label = query.strip()
        payload: dict[str, Any] = {
            "mode": mode,
            "bare_play_mode": mode,
            "query": label,
            "playback_result": inner_raw,
        }
        if mode == "track":
            payload["track_name"] = label
            payload["artist_name"] = ""
        else:
            payload["artist_name"] = label
        try:
            inner = json.loads(inner_raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            inner = None
        if isinstance(inner, dict):
            ok = inner.get("ok")
            verified = inner.get("playback_verified")
            payload["ok"] = ok is True or verified is True
            if payload.get("ok") is not True:
                self._lift_failure_fields(payload, inner)
            if payload.get("ok") is not True:
                if not payload.get("failure_reason"):
                    if inner.get("artist_id") and "playable tracks" in str(inner.get("error") or ""):
                        payload["failure_reason"] = "no_tracks_for_artist"
                    elif "couldn't find an artist" in str(inner.get("error") or "").lower():
                        payload["failure_reason"] = "artist_not_found"
                    elif verified is False and ok is not True:
                        payload["failure_reason"] = "playback_not_verified"
                    else:
                        payload["failure_reason"] = self._playback_failure_reason_from_result(
                            inner.get("playback") if isinstance(inner.get("playback"), dict) else inner
                        )
        else:
            payload["ok"] = False
            payload["failure_reason"] = "play_request_failed"
        if payload.get("ok") is not True:
            logger.warning(
                "spotify_play_bare_failed query=%r mode=%s reason=%s",
                label,
                mode,
                payload.get("failure_reason"),
            )
        return _compact(payload)

    def _play_bare(self, arguments: dict[str, Any]) -> str:
        query = _pick_arg(arguments, "query", "q", "name", "track_name", "artist_name")
        if not query.strip():
            return json.dumps(
                {"ok": False, "error": "query is required", "failure_reason": "invalid_query"}
            )
        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        label = query.strip()
        mode = self._bare_play_mode(label, market)
        if mode == "track":
            inner = self._play_track({"track_name": label, "artist_name": ""})
            return self._finish_play_bare(label, "track", inner)
        resolved = self._resolve_artist_search_query(label, market)
        if not resolved:
            payload = {
                "ok": False,
                "mode": "artist",
                "bare_play_mode": "artist",
                "query": label,
                "artist_name": label,
                "error": f"I couldn't find an artist called {label} on Spotify.",
                "user_message": f"I couldn't find an artist called {label} on Spotify.",
                "failure_reason": "artist_not_found",
                "reconnect_spotify_unnecessary": True,
            }
            logger.warning(
                "spotify_play_bare_failed query=%r mode=artist reason=artist_not_found",
                label,
            )
            return _compact(payload)
        cid, aname, _, _match_kind = resolved
        inner = self._play_artist(
            {
                "artist_id": cid,
                "artist_name": aname or label,
                "market": market,
                "_skip_bare_redirect": True,
                "_bare_play_query": label,
            }
        )
        return self._finish_play_bare(label, "artist", inner)

    def _play_artist(self, arguments: dict[str, Any]) -> str:
        """Play an artist by starting their top track via album context + offset (never raw uris)."""
        raw_ref = _pick_arg(arguments, "artist_name", "name", "artist_id", "id", "artist")
        if not raw_ref or not str(raw_ref).strip():
            return json.dumps(
                {
                    "error": "artist_name is required",
                    "hint": "Pass the artist's name from spotify_search (e.g. Radiohead).",
                }
            )
        market = _normalize_market(_pick_arg(arguments, "market", "country"))
        skip_redirect = bool(arguments.get("_skip_bare_redirect"))
        parsed = _parse_spotify_context_ref(str(raw_ref))
        if (
            not skip_redirect
            and (parsed is None or parsed[0] != "artist")
            and not _looks_like_spotify_catalog_id(str(raw_ref).strip())
        ):
            if self._bare_play_mode(str(raw_ref).strip(), market) == "track":
                return self._play_track(
                    {"track_name": str(raw_ref).strip(), "artist_name": ""}
                )
        artist_id_arg = _pick_arg(arguments, "artist_id", "id")
        parsed = _parse_spotify_context_ref(str(raw_ref))
        if (
            isinstance(artist_id_arg, str)
            and artist_id_arg.strip()
            and _looks_like_spotify_catalog_id(artist_id_arg.strip())
        ):
            cid = artist_id_arg.strip()
        elif parsed and parsed[0] == "artist":
            cid = parsed[1]
        else:
            cid = self._canonical_artist_id(str(raw_ref).strip(), market)
        if not cid:
            label = str(raw_ref).strip()
            return json.dumps(
                {
                    "ok": False,
                    "failure_reason": "artist_not_found",
                    "error": f"I couldn't find an artist called {label} on Spotify.",
                    "user_message": f"I couldn't find an artist called {label} on Spotify.",
                    "reconnect_spotify_unnecessary": True,
                },
                ensure_ascii=False,
            )
        requested_label = str(raw_ref).strip()
        bare_play_query = str(arguments.get("_bare_play_query") or "").strip()
        name_was_free_text = not _looks_like_spotify_catalog_id(
            requested_label
        ) and not (parsed and parsed[0] == "artist")
        resolve_name_ref = requested_label
        if _looks_like_spotify_catalog_id(resolve_name_ref):
            resolve_name_ref = _pick_arg(arguments, "artist_name", "name") or cid
        _, artist_name_resolved = self._resolve_artist_id_and_name(resolve_name_ref, market)
        name_ok = artist_query_matches_candidate_name(
            requested_label, artist_name_resolved or ""
        )
        resolved_search_name = ""
        if bare_play_query:
            resolved_bare = self._resolve_artist_search_query(bare_play_query, market)
            if resolved_bare and resolved_bare[0] == cid:
                name_ok = True
                resolved_search_name = str(resolved_bare[1] or "").strip()
        if name_was_free_text and not name_ok:
            return json.dumps(
                {
                    "ok": False,
                    "failure_reason": "artist_not_found",
                    "error": f"I couldn't find an artist called {requested_label} on Spotify.",
                    "user_message": f"I couldn't find an artist called {requested_label} on Spotify.",
                    "reconnect_spotify_unnecessary": True,
                },
                ensure_ascii=False,
            )
        passed_artist_name = _pick_arg(arguments, "artist_name", "name").strip()
        track_search_name = self._artist_name_for_track_search(
            cid,
            market,
            resolved_search_name,
            passed_artist_name,
            requested_label,
            bare_play_query,
        )
        artist_name_resolved = track_search_name or artist_name_resolved or requested_label
        name_hints = tuple(
            h
            for h in (
                track_search_name,
                resolved_search_name,
                passed_artist_name,
                requested_label,
                bare_play_query,
            )
            if isinstance(h, str) and h.strip() and not _looks_like_spotify_catalog_id(h.strip())
        )
        track_dicts = self._search_artist_tracks_for_popularity(
            artist_id=cid,
            artist_name=track_search_name,
            market=market,
            name_hints=name_hints,
        )
        track_dicts = [tr for tr in track_dicts if isinstance(tr, dict) and tr.get("uri")]
        if not track_dicts:
            track_dicts = self._search_tracks_by_artist_id(
                artist_id=cid,
                market=market,
                name_hints=name_hints,
            )
            track_dicts = [tr for tr in track_dicts if isinstance(tr, dict) and tr.get("uri")]
        if not track_dicts:
            device_id = _coerce_str(arguments.get("device_id"))
            summary = self._play_artist_via_context_uri(
                artist_id=cid,
                artist_name=artist_name_resolved,
                device_id=device_id,
            )
            if summary.get("ok") is True or summary.get("playback_verified") is True:
                return _compact(summary)
            summary["failure_reason"] = "no_tracks_for_artist"
            if not summary.get("error"):
                summary["error"] = "I couldn't find playable tracks for that artist right now."
            return _compact(summary)
        first = _pick_track_by_popularity(track_dicts) or track_dicts[0]
        play_body = self._album_offset_body_for_track(first)
        if not play_body:
            return json.dumps(
                {
                    "ok": False,
                    "failure_reason": "no_tracks_for_artist",
                    "artist_id": cid,
                    "error": "I couldn't resolve an album context for that artist's top track.",
                    "reconnect_spotify_unnecessary": True,
                },
                ensure_ascii=False,
            )
        artist_name = artist_name_resolved or str(raw_ref).strip()
        play_args: dict[str, Any] = dict(play_body)
        device_id = _coerce_str(arguments.get("device_id"))
        if device_id:
            play_args["device_id"] = device_id
        play_args["playback_request_label"] = artist_name
        play_raw = self._start_playback(play_args)
        try:
            play_result = json.loads(play_raw)
        except (json.JSONDecodeError, ValueError):
            play_result = {"ok": False, "raw": play_raw}
        play_ok = isinstance(play_result, dict) and play_result.get("ok") is True
        player = self._poll_player_state(attempts=6, delay_s=0.5)
        verified = bool(
            isinstance(play_result, dict) and play_result.get("playback_verified") is True
        )
        if player and isinstance(first, dict):
            track_uri = str(first.get("uri") or "")
            if track_uri and self._track_uri_from_player(player) == track_uri:
                verified = True
            elif self._player_item_features_artist(player, artist_name, artist_id=cid):
                verified = True
        summary: dict[str, Any] = {
            "artist_id": cid,
            "artist_name": artist_name,
            "play_body": play_body,
            "playback": play_result,
            "player_after": player,
            "playback_verified": verified,
            "ok": play_ok or verified,
        }
        if not play_ok and not verified:
            user_msg = play_result.get("user_message") if isinstance(play_result, dict) else None
            if isinstance(user_msg, str) and user_msg.strip():
                summary["user_message"] = user_msg.strip()
                summary["error"] = user_msg.strip()
            else:
                summary["error"] = _PLAYBACK_START_FAILED_USER_MESSAGE
            summary["failure_reason"] = (
                self._playback_failure_reason_from_result(play_result)
                if isinstance(play_result, dict)
                else "playback_not_verified"
            )
            if not verified:
                summary["failure_reason"] = "playback_not_verified"
            self._lift_failure_fields(summary, play_result if isinstance(play_result, dict) else {})
        return _compact(summary)

    def _play_playlist(self, arguments: dict[str, Any]) -> str:
        """Composite tool: start a playlist (optionally at a specific track) and set repeat/shuffle in one call.

        Prefer this over chaining spotify_start_resume_playback + spotify_set_repeat when the user
        says something like "play RNB2025 starting at Kill Bill with repeat on". It plays the
        context, then (if the play call returned ok) applies repeat and shuffle.
        """
        raw_ref = _pick_arg(arguments, "playlist_id", "playlistId", "id", "context_uri", "artist_name")
        parsed = _parse_spotify_context_ref(raw_ref) if raw_ref else None
        if parsed is None and raw_ref:
            bare = _normalize_spotify_id(raw_ref, "playlist")
            if _looks_like_spotify_catalog_id(bare):
                parsed = ("playlist", bare)
            else:
                market = _normalize_market(_pick_arg(arguments, "market", "country"))
                artist_cid = self._canonical_artist_id(raw_ref.strip(), market)
                if artist_cid:
                    return self._play_artist(
                        {
                            "artist_id": artist_cid,
                            "market": market,
                            **{
                                k: v
                                for k, v in arguments.items()
                                if k not in ("playlist_id", "playlistId", "id", "context_uri", "artist_name")
                            },
                        }
                    )
                else:
                    return json.dumps(
                        {
                            "error": "playlist_id must be a 22-character Spotify id or spotify:playlist:/album:/artist: URI from search tools.",
                            "hint": "Call spotify_search or spotify_user_playlists first — do not invent ids.",
                            "reconnect_spotify_unnecessary": True,
                        },
                        ensure_ascii=False,
                    )
        if parsed is None:
            return json.dumps(
                {
                    "error": "playlist_id is required",
                    "hint": "Pass playlist_id (bare id or spotify:playlist:<id> from spotify_user_playlists or search).",
                }
            )

        kind, pid = parsed
        if kind == "album":
            context_uri = f"spotify:album:{pid}"
        elif kind == "artist":
            return self._play_artist(
                {
                    "artist_id": pid,
                    **{
                        k: v
                        for k, v in arguments.items()
                        if k not in ("playlist_id", "playlistId", "id", "context_uri")
                    },
                }
            )
        elif kind == "track":
            play_args = {"uris": [f"spotify:track:{pid}"]}
            device_id = _coerce_str(arguments.get("device_id"))
            if device_id:
                play_args["device_id"] = device_id
            play_raw = self._start_playback(play_args)
            try:
                play_result = json.loads(play_raw)
            except (json.JSONDecodeError, ValueError):
                play_result = {"raw": play_raw}
            return _compact(
                {
                    "track_id": pid,
                    "playback": play_result,
                    "ok": isinstance(play_result, dict) and play_result.get("ok") is True,
                }
            )
        else:
            blocked = self._block_editorial_playlist_id(pid)
            if blocked:
                return blocked
            try:
                snap = self._playlist_owner_snapshot(pid)
            except (httpx.HTTPStatusError, AssertionError, RuntimeError, OSError, ValueError):
                snap = {}
            if snap.get("is_owned") is False:
                return json.dumps(
                    {
                        "ok": False,
                        "failure_reason": "not_owned",
                        "user_message": _PLAYLIST_PLAY_UNAVAILABLE_USER_MESSAGE,
                        "error": _PLAYLIST_PLAY_UNAVAILABLE_USER_MESSAGE,
                        "playlist_not_owned_by_user": True,
                        "reconnect_spotify_unnecessary": True,
                    },
                    ensure_ascii=False,
                )
            context_uri = f"spotify:playlist:{pid}"

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

        play_args: dict[str, Any] = {"context_uri": context_uri}
        if start_at_uri:
            play_args["offset"] = {"uri": start_at_uri}
        elif start_at_position >= 0:
            play_args["offset"] = {"position": start_at_position}
        device_id = _coerce_str(arguments.get("device_id"))
        if device_id:
            play_args["device_id"] = device_id
        label = _coerce_str(arguments.get("playback_request_label"), "")
        if label:
            play_args["playback_request_label"] = label

        play_raw = self._start_playback(play_args)
        try:
            play_result = json.loads(play_raw)
        except (json.JSONDecodeError, ValueError):
            play_result = {"raw": play_raw}

        summary: dict[str, Any] = {
            "context_type": kind,
            "playlist_id": pid if kind == "playlist" else None,
            "context_id": pid,
            "context_uri": context_uri,
            "start_at_uri": start_at_uri or None,
            "start_at_position": start_at_position if start_at_position >= 0 else None,
            "playback": play_result,
        }

        play_ok = isinstance(play_result, dict) and play_result.get("ok") is True
        player = self._poll_player_state(attempts=6, delay_s=0.5)
        verified = bool(
            isinstance(play_result, dict) and play_result.get("playback_verified") is True
        )
        if player and isinstance(play_result, dict) and play_result.get("body"):
            body_play = play_result.get("body")
            if isinstance(body_play, dict) and self._playback_matches(body_play, attempts=1, delay_s=0):
                verified = True
        summary["player_after"] = player
        summary["playback_verified"] = verified
        if not play_ok or not verified:
            summary["ok"] = False
            user_msg = play_result.get("user_message") if isinstance(play_result, dict) else None
            if isinstance(user_msg, str) and user_msg.strip():
                summary["user_message"] = user_msg.strip()
                summary["error"] = user_msg.strip()
            else:
                summary["error"] = _PLAYBACK_START_FAILED_USER_MESSAGE
            if isinstance(play_result, dict):
                for key in (
                    "spotify_error_body_redacted",
                    "spotify_http_status",
                    "spotify_api_message",
                    "failure_reason",
                ):
                    val = play_result.get(key)
                    if val is not None:
                        summary[key] = val
            if not summary.get("failure_reason"):
                if isinstance(play_result, dict) and play_result.get("spotify_error_body_redacted"):
                    summary.setdefault("failure_reason", "spotify_http_error")
                elif not verified:
                    summary["failure_reason"] = "playback_not_verified"
                elif not play_ok:
                    summary["failure_reason"] = "playback_start_failed"
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
        summary["playback_verified"] = verified
        return _compact(summary)

    def _track_uri_from_player(self, player: dict[str, Any] | None) -> str:
        if not player or not isinstance(player, dict):
            return ""
        item = player.get("item") if isinstance(player.get("item"), dict) else None
        if not item:
            return ""
        uri = item.get("uri")
        return uri.strip() if isinstance(uri, str) else ""

    def _player_item_features_artist(
        self,
        player: dict[str, Any] | None,
        artist_name: str,
        *,
        artist_id: str | None = None,
    ) -> bool:
        if not player or not isinstance(player, dict):
            return False
        req = (artist_name or "").strip()
        if not req:
            return False
        item = player.get("item") if isinstance(player.get("item"), dict) else None
        if not item:
            return False
        artists = item.get("artists")
        if not isinstance(artists, list):
            return False
        aid = (artist_id or "").strip()
        for artist in artists:
            if not isinstance(artist, dict):
                continue
            if aid:
                credited_id = artist.get("id")
                if isinstance(credited_id, str) and credited_id.strip() == aid:
                    return True
            name = artist.get("name")
            if isinstance(name, str):
                credited = name.strip()
                if artist_names_match(req, credited):
                    return True
                if artist_query_matches_candidate_name(req, credited):
                    return True
        return False

    def _poll_player_state(self, *, attempts: int = 8, delay_s: float = 0.35) -> dict[str, Any] | None:
        for _ in range(max(1, attempts)):
            try:
                ps = self.client.api_get("/me/player")
            except httpx.HTTPStatusError:
                ps = None
            if isinstance(ps, dict) and isinstance(ps.get("item"), dict):
                return ps
            time.sleep(delay_s)
        return None

    def _play_track(self, arguments: dict[str, Any]) -> str:
        track_name = _pick_arg(arguments, "track_name", "title", "name")
        artist_name = _pick_arg(arguments, "artist_name", "artist")
        if track_name.strip().lower() in _GENERIC_PLAYBY_TRACK_TITLES and artist_name.strip():
            return self._play_artist(
                {
                    "artist_name": artist_name.strip(),
                    "device_id": arguments.get("device_id"),
                    "_skip_bare_redirect": True,
                }
            )
        if not track_name.strip():
            return json.dumps({"ok": False, "error": "track_name is required"})
        uri, track_match, err = self._resolve_track_for_queue(
            track_name=track_name,
            artist_name=artist_name,
        )
        if err or not uri:
            msg = err or "Could not resolve track"
            if "No tracks found for" in msg:
                label = track_name.strip() or "that"
                msg = f"I couldn't find a track called {label!s} on Spotify."
            return json.dumps({"ok": False, "error": msg, "user_message": msg})
        device_id = _coerce_str(arguments.get("device_id"))
        play_args_base: dict[str, Any] = {"force_interrupt": True}
        if device_id:
            play_args_base["device_id"] = device_id
        album_body = (
            self._album_offset_body_for_track(track_match, track_uri=uri)
            if isinstance(track_match, dict)
            else None
        )
        play_result: dict[str, Any] = {"ok": False}
        if album_body:
            play_raw = self._start_playback({**play_args_base, **album_body})
            try:
                play_result = json.loads(play_raw)
            except (json.JSONDecodeError, ValueError):
                play_result = {"ok": False}
        verified = bool(
            isinstance(play_result, dict) and play_result.get("playback_verified") is True
        )
        player = self._poll_player_state()
        if player and self._track_uri_from_player(player) == uri:
            verified = True
        if not verified:
            uris_args: dict[str, Any] = {
                **play_args_base,
                "uris": [uri],
                "skip_album_context": True,
            }
            play_raw = self._start_playback(uris_args)
            try:
                play_result = json.loads(play_raw)
            except (json.JSONDecodeError, ValueError):
                play_result = {"ok": False}
            player = self._poll_player_state()
            verified = bool(
                isinstance(play_result, dict) and play_result.get("playback_verified") is True
            )
            if player and self._track_uri_from_player(player) == uri:
                verified = True
        summary: dict[str, Any] = {
            "ok": verified,
            "playback_verified": verified,
            "requested_track": track_name,
            "requested_artist": artist_name,
            "uri": uri,
            "player_after": player,
            "playback": play_result,
        }
        if not verified:
            summary["error"] = "Playback did not switch to the requested track."
        return _compact(summary)

    def _skip_next(self, arguments: dict[str, Any]) -> str:
        return self._skip_direction(arguments, direction="next")

    def _skip_previous(self, arguments: dict[str, Any]) -> str:
        return self._skip_direction(arguments, direction="previous")

    def _skip_direction(self, arguments: dict[str, Any], *, direction: str) -> str:
        before = self._current_playback_snapshot()
        before_uri = (before.get("item_uri") or "").strip()
        device_id = str(arguments.get("device_id", "")).strip() or self._device_id()
        params: dict[str, str] = {}
        if device_id:
            params["device_id"] = device_id
        path = "/me/player/next" if direction == "next" else "/me/player/previous"
        try:
            if params:
                self.client.api_post(path, params=params)
            else:
                self.client.api_post(path)
        except httpx.HTTPStatusError as exc:
            return json.dumps(
                {
                    "ok": False,
                    "skipped": False,
                    "error": _spotify_http_message(exc),
                    "before_uri": before_uri,
                    **_spotify_http_error_fields(exc),
                },
                ensure_ascii=False,
            )
        changed = False
        player: dict[str, Any] | None = None
        for _ in range(10):
            time.sleep(0.35)
            snap = self._current_playback_snapshot()
            after_uri = (snap.get("item_uri") or "").strip()
            if after_uri and after_uri != before_uri:
                changed = True
                player = self._poll_player_state(attempts=1, delay_s=0)
                break
        return json.dumps(
            {
                "ok": changed,
                "skipped": changed,
                "before_uri": before_uri,
                "after_uri": self._track_uri_from_player(player) if player else "",
                "player_after": player,
            },
            ensure_ascii=False,
        )

    def _add_to_queue(self, arguments: dict[str, Any]) -> str:
        uri = str(arguments.get("uri", "")).strip()
        if not uri:
            query = _pick_arg(arguments, "query", "q")
            track_name = _pick_arg(arguments, "track_name", "name", "title")
            artist_name = _pick_arg(arguments, "artist_name", "artist")
            market = _pick_arg(arguments, "market", "country")
            resolved, err = self._resolve_track_uri_for_queue(
                query=query,
                track_name=track_name,
                artist_name=artist_name,
                market=market,
            )
            if err:
                return json.dumps({"error": err})
            uri = resolved or ""
        if not uri:
            return json.dumps({"error": "uri is required (or pass query / track_name + artist_name)"})
        explicit = _sanitize_model_device_id(str(arguments.get("device_id", "")))
        saved = (self._device_id() or "").strip()
        if explicit:
            device_id, _note = self._coerce_playback_device_id(explicit)
        elif saved:
            device_id, _note = self._coerce_playback_device_id(saved)
        else:
            device_id = ""
        params: dict[str, str] = {"uri": uri}
        if device_id:
            params["device_id"] = device_id
        self.client.api_post("/me/player/queue", params=params)
        return json.dumps({"ok": True, "uri": uri})

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
            "description": (
                "Search Spotify (tracks, artists, albums, playlists, shows/podcasts, episodes). "
                "For podcasts use types=show (not spotify_search_playlists). For episodes use types=episode."
            ),
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
            "description": (
                "List playlists already in the signed-in user's library (owned + followed). "
                "Do NOT use for creating playlists — use spotify_create_playlist to make a new one. "
                "Do NOT use for top-artist analytics — use spotify_top_artists instead."
            ),
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
            "name": "spotify_artist_latest_album",
            "description": "Newest album or single for an artist (by release_date). Lookup only — use spotify_play_artist_latest_release to play.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_id": {"type": "string"},
                    "include_groups": {"type": "string"},
                    "prefer": {"type": "string"},
                    "market": {"type": "string"},
                },
                "required": ["artist_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_play_artist_latest_release",
            "description": (
                "Find the newest released single/album for an artist (paginated, release_date <= today) and start playback."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_id": {"type": "string"},
                    "artist_name": {"type": "string"},
                    "kind": {"type": "string"},
                    "market": {"type": "string"},
                    "device_id": {"type": "string"},
                },
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
            "name": "spotify_play_artist_popular_track",
            "description": (
                "PLAY NOW — play an artist's most popular / best-known song. Searches tracks with "
                'artist:"<name>", keeps only that artist, picks highest popularity (or first search hit '
                "when popularity is unavailable in dev mode). Use for "
                "'play their most popular song', 'biggest hit by X', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "artist": {
                        "type": "string",
                        "description": "Artist name or id from spotify_search / chat context.",
                    },
                    "artist_name": {"type": "string"},
                    "artist_id": {"type": "string"},
                    "device_id": {"type": "string"},
                    "market": {"type": "string"},
                },
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
            "name": "spotify_recently_played",
            "description": (
                "Return the user's recently played tracks (up to 50). Requires scope "
                "user-read-recently-played — if missing, tell the user to reconnect Spotify."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "1-50, default 20."},
                    "after": {"type": "string", "description": "Unix ms timestamp cursor."},
                    "before": {"type": "string", "description": "Unix ms timestamp cursor."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_save_tracks",
            "description": "Save (like) one or more tracks to the user's library. Requires user-library-modify.",
            "parameters": {
                "type": "object",
                "properties": {
                    "track_id": {"type": "string"},
                    "track_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_unsave_tracks",
            "description": "Remove saved (liked) tracks from the user's library. Requires user-library-modify.",
            "parameters": {
                "type": "object",
                "properties": {
                    "track_id": {"type": "string"},
                    "track_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_save_albums",
            "description": "Save albums to the user's library. Requires user-library-modify.",
            "parameters": {
                "type": "object",
                "properties": {
                    "album_id": {"type": "string"},
                    "album_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_unsave_albums",
            "description": "Remove saved albums from the user's library. Requires user-library-modify.",
            "parameters": {
                "type": "object",
                "properties": {
                    "album_id": {"type": "string"},
                    "album_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_saved_albums",
            "description": (
                "List albums saved in the user's library (compact summary in user_message — "
                "do not paste raw JSON to the user). For 'is this album saved?' use spotify_library_contains."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                    "offset": {"type": "integer"},
                    "market": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_follow_artist",
            "description": "Follow one or more artists. Requires user-follow-modify.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_id": {"type": "string"},
                    "artist_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_unfollow_artist",
            "description": "Unfollow artists. Requires user-follow-modify.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_id": {"type": "string"},
                    "artist_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_get_queue",
            "description": "Read the current playback queue (now playing + up next). Requires user-read-playback-state.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_remove_from_queue",
            "description": (
                "Explain that Spotify's Web API cannot remove arbitrary queue items — only skip forward. "
                "Use when the user asks to remove/unqueue a song; offer spotify_skip_next instead."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_playlists_containing_track",
            "description": (
                "Composite: scan the user's playlists for a track id (slow). "
                "Do NOT use for 'is this song in my likes?' — use spotify_library_contains instead. "
                "Stops after max_playlists (default 50, max 200) with truncated=true when capped."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "track_id": {"type": "string"},
                    "max_playlists": {"type": "integer"},
                    "max_pages_per_playlist": {"type": "integer"},
                },
                "required": ["track_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_top_artists",
            "description": (
                "Return the signed-in user's top artists for a given time window. "
                "Use this for 'my top artists / favorite artists / who do I listen to most'. "
                "time_range: short_term (~last 4 weeks), medium_term (~last 6 months, default), "
                "long_term (calculated from ~the user's all-time history). "
                "Spotify does NOT expose per-artist play counts; rank is the only ordering signal. "
                "Requires Spotify scope user-top-read — if missing, instruct the user to Sign out → Connect."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "time_range": {
                        "type": "string",
                        "enum": ["short_term", "medium_term", "long_term"],
                        "description": "Time window for the ranking. Default medium_term (~last 6 months).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many artists to return (1-50). Default 20.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Pagination offset (0-49). Spotify caps the total list at 50.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_top_tracks",
            "description": (
                "Return the signed-in user's top tracks for a given time window. "
                "Use this for 'my top songs / most played tracks / favorite songs'. "
                "time_range: short_term (~last 4 weeks), medium_term (~last 6 months, default), "
                "long_term (calculated from ~the user's all-time history). "
                "Spotify does NOT expose per-track play counts; rank is the only ordering signal. "
                "Requires Spotify scope user-top-read — if missing, instruct the user to Sign out → Connect."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "time_range": {
                        "type": "string",
                        "enum": ["short_term", "medium_term", "long_term"],
                        "description": "Time window for the ranking. Default medium_term (~last 6 months).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many tracks to return (1-50). Default 20.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Pagination offset (0-49). Spotify caps the total list at 50.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_followed_artists",
            "description": (
                "List the ARTISTS the signed-in user follows (cursor-paginated). "
                "Spotify's Web API does NOT expose users the user follows, and does NOT expose the "
                "user's own follower list (only a count via spotify_me). If the user asks for either "
                "of those, say so plainly and offer this tool as the closest available alternative. "
                "Requires Spotify scope user-follow-read — if missing, instruct Sign out → Connect."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "1-50. Default 20."},
                    "after": {
                        "type": "string",
                        "description": "Cursor for the next page (echo back next_cursor from a previous call).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_user_public_playlists",
            "description": (
                "List a Spotify user's PUBLIC playlists by their user_id. No scope required (public data). "
                "Use this when the user asks 'what playlists does <person> have' and gives a user_id "
                "(the part after spotify:user: or open.spotify.com/user/<id>). The Web API has NO endpoint "
                "to look up a user by display name, and CANNOT show another user's private playlists — "
                "say that plainly if the user expected either."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Target Spotify user_id."},
                    "limit": {"type": "integer", "description": "1-50. Default 20."},
                    "offset": {"type": "integer", "description": "Pagination offset."},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_search_playlists",
            "description": (
                "Search Spotify's catalog for public PLAYLISTS matching a free-text description "
                "(e.g. 'lo-fi study beats', 'workout 2025 hip hop'). Returns id, name, owner, "
                "image, total_tracks. Use this for 'find me a playlist about <X>'. Then optionally "
                "spotify_follow_playlist (to save it), spotify_play_playlist (to play it), or "
                "spotify_duplicate_playlist (to copy it into a writable playlist you own). "
                "Editing someone else's playlist is NOT possible — duplicate first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-text description / search query."},
                    "limit": {"type": "integer", "description": "1-10. Default 5."},
                    "offset": {"type": "integer", "description": "Pagination offset (0-950)."},
                    "market": {"type": "string", "description": "Optional ISO 3166 country code (e.g. US)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_follow_playlist",
            "description": (
                "Follow a playlist (Spotify treats this as 'save to your library'). Works for any "
                "playlist_id, including ones owned by other users. Following does NOT make you the "
                "owner — you still cannot add/remove tracks; for that, use spotify_duplicate_playlist. "
                "Requires playlist-modify-public (default) or playlist-modify-private if public=false."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_id": {"type": "string"},
                    "public": {
                        "type": "boolean",
                        "description": "Whether the follow is public on your profile. Default true.",
                    },
                },
                "required": ["playlist_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_duplicate_playlist",
            "description": (
                "Composite tool (NOT a native Spotify endpoint): reads the source playlist's "
                "tracks, creates a brand-new playlist OWNED by the signed-in user, and fills it "
                "with those tracks. Same effect as 'Add to other playlist' in the Spotify mobile "
                "UI. Use for 'copy <playlist> and let me edit it' or 'duplicate my playlist X'. "
                "Works fully when the user OWNS the source; returns new_playlist_id (also exposed "
                "as playlist_id_for_add_tracks) which is fully writable. Dev-mode limit: if the "
                "source is owned by another user (including playlists the signed-in user merely "
                "follows), the read is blocked by Spotify and the tool returns "
                "source_not_owned_by_user: true with try_instead alternatives — do NOT suggest "
                "sign-out in that case. After a successful copy, edit with "
                "spotify_add_tracks_to_playlist / spotify_remove_playlist_tracks etc., and play "
                "with spotify_play_playlist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_playlist_id": {
                        "type": "string",
                        "description": "Playlist to copy from (any user's public/collaborative playlist).",
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional name for the new playlist. Defaults to 'Copy of <source>'.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description for the new playlist.",
                    },
                    "public": {
                        "type": "boolean",
                        "description": "Whether the new playlist is public on the user's profile. Default false.",
                    },
                    "max_tracks": {
                        "type": "integer",
                        "description": "Cap on tracks to copy. Default 5000, hard max 10000.",
                    },
                },
                "required": ["source_playlist_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_create_playlist",
            "description": (
                "Create a brand-new empty playlist for the signed-in user (not the same as spotify_user_playlists, "
                "which only lists existing playlists). Defaults to private unless public=true. "
                "Then add tracks with spotify_add_tracks_to_playlist."
            ),
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
            "name": "spotify_play_track",
            "description": (
                "PLAY NOW — start a specific track by title and primary artist using PUT /me/player/play "
                "with a track URI (not the queue). Prefer this for 'play <song> by <artist>'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "track_name": {"type": "string", "description": "Song title (e.g. Blinding Lights)."},
                    "artist_name": {"type": "string", "description": "Primary artist (e.g. The Weeknd)."},
                    "device_id": {"type": "string"},
                    "market": {"type": "string"},
                },
                "required": ["track_name", "artist_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_play_artist",
            "description": (
                "PLAY NOW — play an artist by starting their top track using album context + offset "
                "(never a raw uris list). Use for 'play Radiohead', 'play Taylor Swift', or artist_id "
                "/ spotify:artist: URIs from search. Do NOT use spotify_play_playlist for artist names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_name": {
                        "type": "string",
                        "description": "Artist display name (e.g. Radiohead).",
                    },
                    "artist_id": {
                        "type": "string",
                        "description": "Optional 22-char artist id or spotify:artist: URI.",
                    },
                    "device_id": {"type": "string"},
                    "market": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_play_playlist",
            "description": (
                "PLAY NOW — immediate interruption. COMPOSITE tool: start a playlist or album context "
                "(playlist_id accepts spotify:playlist:, spotify:album:, or bare 22-char ids from search). "
                "For artist names use spotify_play_artist instead. "
                "(optionally at a specific track via start_at_uri) AND set repeat/shuffle in one call. This is the RIGHT "
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
                "properties": {
                    "uri": {"type": "string"},
                    "query": {"type": "string"},
                    "track_name": {"type": "string"},
                    "artist_name": {"type": "string"},
                    "market": {"type": "string"},
                    "device_id": {"type": "string"},
                },
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
                "properties": {
                    "uri": {"type": "string"},
                    "query": {"type": "string"},
                    "track_name": {"type": "string"},
                    "artist_name": {"type": "string"},
                    "market": {"type": "string"},
                    "device_id": {"type": "string"},
                },
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
    {
        "type": "function",
        "function": {
            "name": "spotify_library_contains",
            "description": (
                "Check whether Spotify URIs are saved/liked/in the user's library (tracks, albums, episodes, "
                "shows, playlists, artists). Prefer this for 'is X saved / in my likes / in my library?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uris": {"type": "array", "items": {"type": "string"}},
                    "uri": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_library_save",
            "description": "Save any supported Spotify URIs to the user's library (PUT /me/library).",
            "parameters": {
                "type": "object",
                "properties": {"uris": {"type": "array", "items": {"type": "string"}}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_library_remove",
            "description": "Remove Spotify URIs from the user's library (DELETE /me/library).",
            "parameters": {
                "type": "object",
                "properties": {"uris": {"type": "array", "items": {"type": "string"}}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_get_show",
            "description": "Get podcast show metadata by id.",
            "parameters": {
                "type": "object",
                "properties": {"show_id": {"type": "string"}},
                "required": ["show_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_get_show_episodes",
            "description": "List episodes for a podcast show.",
            "parameters": {
                "type": "object",
                "properties": {
                    "show_id": {"type": "string"},
                    "limit": {"type": "integer"},
                    "offset": {"type": "integer"},
                },
                "required": ["show_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_get_episode",
            "description": "Get podcast episode metadata by id.",
            "parameters": {
                "type": "object",
                "properties": {"episode_id": {"type": "string"}},
                "required": ["episode_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_get_audiobook",
            "description": "Get audiobook metadata by id.",
            "parameters": {
                "type": "object",
                "properties": {"audiobook_id": {"type": "string"}},
                "required": ["audiobook_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_get_audiobook_chapters",
            "description": "List chapters for an audiobook.",
            "parameters": {
                "type": "object",
                "properties": {
                    "audiobook_id": {"type": "string"},
                    "limit": {"type": "integer"},
                    "offset": {"type": "integer"},
                },
                "required": ["audiobook_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_get_chapter",
            "description": "Get audiobook chapter metadata by id.",
            "parameters": {
                "type": "object",
                "properties": {"chapter_id": {"type": "string"}},
                "required": ["chapter_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_user_saved_shows",
            "description": "List podcast shows saved in the user's library.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}, "offset": {"type": "integer"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_user_saved_episodes",
            "description": "List saved podcast episodes in the user's library.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}, "offset": {"type": "integer"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_user_saved_audiobooks",
            "description": "List audiobooks saved in the user's library.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}, "offset": {"type": "integer"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_play_show_latest_episode",
            "description": (
                "Play the most recent episode of a podcast show (search shows first, then pass show_id). "
                "Use for 'play the latest episode of <show>'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "show_id": {"type": "string"},
                    "device_id": {"type": "string"},
                },
                "required": ["show_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_playlist_builder_preview",
            "description": (
                "Propose a new private playlist: resolve track_queries via search, show numbered preview, "
                "and wait for user approval before spotify_playlist_builder_commit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "track_queries": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["track_queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_playlist_builder_edit",
            "description": (
                "Edit the pending playlist builder preview in place (remove by 1-based indices, "
                "add tracks by search query, or replace one slot). Returns a renumbered preview."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "remove_indices": {"type": "array", "items": {"type": "integer"}},
                    "add_queries": {"type": "array", "items": {"type": "string"}},
                    "replace_index": {"type": "integer"},
                    "replace_query": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_playlist_builder_commit",
            "description": "Create the pending preview playlist after the user approves (private, never empty).",
            "parameters": {
                "type": "object",
                "properties": {
                    "approve": {"type": "boolean"},
                    "name": {"type": "string"},
                },
            },
        },
    },
]
