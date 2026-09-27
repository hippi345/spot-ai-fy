"""Aggregate Spotify now-playing + queue for the mini-player API."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from spot_backend.spotify_client import SpotifyAuthError, SpotifyClient, SpotifyRateLimitError
from spot_backend.token_store import load_device

_QUEUE_MAX = 10


@dataclass
class _NowPlayingCache:
    last_good: dict[str, Any] | None = None
    rate_limited_until: float = 0.0


_CACHE = _NowPlayingCache()


def _retry_after_from_error(exc: SpotifyRateLimitError) -> int:
    msg = str(exc)
    for part in msg.split():
        if part.isdigit():
            return int(part)
    return 1


def _mark_rate_limited(seconds: int) -> None:
    _CACHE.rate_limited_until = time.time() + max(0, seconds)


def _rate_limit_active() -> bool:
    return time.time() < _CACHE.rate_limited_until


def _stale_response() -> dict[str, Any]:
    if _CACHE.last_good is not None:
        out = dict(_CACHE.last_good)
        out["stale"] = True
        return out
    return {
        "is_playing": False,
        "track": None,
        "progress_ms": 0,
        "device": None,
        "queue": [],
        "fetched_at": time.time(),
        "stale": True,
    }


def _artist_names(artists: Any) -> list[str]:
    if not isinstance(artists, list):
        return []
    out: list[str] = []
    for a in artists:
        if isinstance(a, dict) and isinstance(a.get("name"), str):
            out.append(a["name"])
    return out


def _image_sort_key(im: dict[str, Any]) -> int:
    w = im.get("width")
    h = im.get("height")
    if isinstance(w, (int, float)) and w > 0:
        return int(w)
    if isinstance(h, (int, float)) and h > 0:
        return int(h)
    return 300


def _pick_image_url(images: Any, *, prefer_largest: bool, target_px: int = 64) -> str | None:
    if not isinstance(images, list) or not images:
        return None
    valid: list[dict[str, Any]] = []
    for im in images:
        if not isinstance(im, dict):
            continue
        url = im.get("url")
        if not isinstance(url, str) or not url.strip().startswith("http"):
            continue
        valid.append(im)
    if not valid:
        return None
    if prefer_largest:
        valid.sort(key=_image_sort_key, reverse=True)
        return valid[0]["url"]
    valid.sort(key=lambda im: abs(_image_sort_key(im) - target_px))
    return valid[0]["url"]


def _images_for_item(item: dict[str, Any]) -> Any:
    album = item.get("album") if isinstance(item.get("album"), dict) else None
    if album and album.get("images"):
        return album.get("images")
    if item.get("images"):
        return item.get("images")
    show = item.get("show") if isinstance(item.get("show"), dict) else None
    if show and show.get("images"):
        return show.get("images")
    return None


def _normalize_track(item: dict[str, Any] | None, *, small_art: bool = False) -> dict[str, Any] | None:
    if not item or not isinstance(item, dict):
        return None
    tid = item.get("id")
    name = item.get("name")
    if not isinstance(tid, str) or not isinstance(name, str):
        return None
    album = item.get("album") if isinstance(item.get("album"), dict) else {}
    art_url = _pick_image_url(
        _images_for_item(item),
        prefer_largest=not small_art,
        target_px=64 if small_art else 300,
    )
    album_name = album.get("name") if isinstance(album, dict) and isinstance(album.get("name"), str) else ""
    duration = item.get("duration_ms")
    uri = item.get("uri")
    track_uri = uri.strip() if isinstance(uri, str) and uri.strip() else f"spotify:track:{tid}"
    return {
        "id": tid,
        "uri": track_uri,
        "name": name,
        "artists": _artist_names(item.get("artists")),
        "album": album_name,
        "art_url": art_url,
        "duration_ms": int(duration) if isinstance(duration, (int, float)) else 0,
    }


def _normalize_queue_item(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    tr = entry.get("track") if isinstance(entry.get("track"), dict) else entry
    if not isinstance(tr, dict):
        return None
    name = tr.get("name")
    if not isinstance(name, str):
        return None
    album = tr.get("album") if isinstance(tr.get("album"), dict) else {}
    art_url = _pick_image_url(
        _images_for_item(tr) if isinstance(tr, dict) else None,
        prefer_largest=False,
        target_px=48,
    )
    tr_uri = tr.get("uri")
    tid = tr.get("id")
    uri_out = ""
    if isinstance(tr_uri, str) and tr_uri.strip():
        uri_out = tr_uri.strip()
    elif isinstance(tid, str) and tid.strip():
        uri_out = f"spotify:track:{tid.strip()}"
    return {
        "uri": uri_out,
        "name": name,
        "artists": _artist_names(tr.get("artists")),
        "art_url": art_url,
    }


def _device_params(client: SpotifyClient) -> dict[str, str]:
    sel = load_device(client.settings.resolved_device_path)
    if sel and sel.device_id:
        return {"device_id": sel.device_id}
    return {}


def _fetch_player_and_queue(client: SpotifyClient) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    player = client.api_get("/me/player")
    queue_raw: list[dict[str, Any]] = []
    qdata = client.api_get("/me/player/queue")
    if isinstance(qdata, dict):
        qlist = qdata.get("queue")
        if isinstance(qlist, list):
            for row in qlist[:_QUEUE_MAX]:
                norm = _normalize_queue_item(row)
                if norm:
                    queue_raw.append(norm)
    return (player if isinstance(player, dict) else None), queue_raw


def _drop_leading_current_from_queue(
    queue: list[dict[str, Any]],
    current_uri: str | None,
) -> list[dict[str, Any]]:
    if not queue or not current_uri:
        return queue
    first_uri = queue[0].get("uri") if isinstance(queue[0], dict) else None
    if isinstance(first_uri, str) and first_uri.strip() == current_uri.strip():
        return queue[1:]
    return queue


def build_now_playing_payload(
    *,
    player: dict[str, Any] | None,
    queue: list[dict[str, Any]],
) -> dict[str, Any]:
    if not player:
        return {
            "is_playing": False,
            "track": None,
            "progress_ms": 0,
            "device": None,
            "queue": queue,
            "fetched_at": time.time(),
        }

    item = player.get("item") if isinstance(player.get("item"), dict) else None
    track = _normalize_track(item) if item else None
    is_playing = bool(player.get("is_playing"))
    progress = player.get("progress_ms")
    progress_ms = int(progress) if isinstance(progress, (int, float)) else 0
    dev = player.get("device") if isinstance(player.get("device"), dict) else None
    device_out = None
    if dev:
        dname = dev.get("name") if isinstance(dev.get("name"), str) else ""
        dtype = dev.get("type") if isinstance(dev.get("type"), str) else ""
        device_out = {"name": dname, "type": dtype}

    if not track:
        return {
            "is_playing": False,
            "track": None,
            "progress_ms": progress_ms,
            "device": device_out,
            "queue": queue,
            "fetched_at": time.time(),
        }

    current_uri = track.get("uri") if isinstance(track.get("uri"), str) else None
    queue_out = _drop_leading_current_from_queue(queue, current_uri)

    return {
        "is_playing": is_playing,
        "track": track,
        "progress_ms": progress_ms,
        "device": device_out,
        "queue": queue_out,
        "fetched_at": time.time(),
    }


def get_now_playing(client: SpotifyClient) -> dict[str, Any]:
    if _rate_limit_active():
        return _stale_response()
    try:
        player, queue = _fetch_player_and_queue(client)
    except SpotifyRateLimitError as e:
        _mark_rate_limited(_retry_after_from_error(e))
        return _stale_response()
    except (SpotifyAuthError, httpx.HTTPError):
        if _CACHE.last_good is not None:
            return _stale_response()
        raise

    payload = build_now_playing_payload(player=player, queue=queue)
    _CACHE.last_good = payload
    return payload


def _player_signature(player: dict[str, Any] | None) -> tuple[str | None, bool | None]:
    if not isinstance(player, dict):
        return (None, None)
    item = player.get("item") if isinstance(player.get("item"), dict) else None
    tid = item.get("id") if isinstance(item, dict) and isinstance(item.get("id"), str) else None
    is_playing = bool(player.get("is_playing")) if "is_playing" in player else None
    return (tid, is_playing)


def _refresh_now_playing_after_player_action(
    client: SpotifyClient,
    *,
    before: tuple[str | None, bool | None],
    expect_track_change: bool,
) -> dict[str, Any]:
    """Re-fetch /me/player until Spotify reflects the control action (bounded wait)."""
    deadline = time.time() + 2.0
    last_payload: dict[str, Any] | None = None
    while time.time() < deadline:
        player, queue = _fetch_player_and_queue(client)
        payload = build_now_playing_payload(player=player, queue=queue)
        last_payload = payload
        sig = _player_signature(player)
        if expect_track_change:
            if sig[0] and sig[0] != before[0]:
                _CACHE.last_good = payload
                return payload
        elif sig[1] is not None and sig[1] != before[1]:
            _CACHE.last_good = payload
            return payload
        time.sleep(0.15)
    if last_payload is not None:
        _CACHE.last_good = last_payload
        return last_payload
    return get_now_playing(client)


def player_toggle(client: SpotifyClient) -> dict[str, Any]:
    params = _device_params(client)
    player = client.api_get("/me/player")
    before = _player_signature(player if isinstance(player, dict) else None)
    is_playing = isinstance(player, dict) and bool(player.get("is_playing"))
    if is_playing:
        path = "/me/player/pause"
        if params:
            client.api_put(path, params=params)
        else:
            client.api_put(path)
    else:
        path = "/me/player/play"
        if params:
            client.api_put(path, params=params)
        else:
            client.api_put(path)
    return _refresh_now_playing_after_player_action(
        client, before=before, expect_track_change=False
    )


def player_next(client: SpotifyClient) -> dict[str, Any]:
    params = _device_params(client)
    player = client.api_get("/me/player")
    before = _player_signature(player if isinstance(player, dict) else None)
    client.api_post("/me/player/next", params=params or None)
    return _refresh_now_playing_after_player_action(
        client, before=before, expect_track_change=True
    )


def player_previous(client: SpotifyClient) -> dict[str, Any]:
    params = _device_params(client)
    player = client.api_get("/me/player")
    before = _player_signature(player if isinstance(player, dict) else None)
    client.api_post("/me/player/previous", params=params or None)
    return _refresh_now_playing_after_player_action(
        client, before=before, expect_track_change=True
    )


def reset_now_playing_cache_for_tests() -> None:
    """Test helper to clear module-level cache."""
    _CACHE.last_good = None
    _CACHE.rate_limited_until = 0.0
