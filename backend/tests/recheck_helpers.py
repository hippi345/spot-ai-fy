"""Shared fixtures/helpers for r2/r3/r4 recheck tests (dedupe for pylint R0801)."""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx
import respx


def mock_artist_name_search(
    artist_id: str,
    name: str = "Radiohead",
    *,
    extra_items: list[dict[str, Any]] | None = None,
) -> None:
    items = extra_items or [{"id": artist_id, "name": name, "type": "artist"}]
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(
            200,
            json={"artists": {"items": items}},
        )
    )


def gemini_candidates_payload(*candidates: dict[str, Any]) -> dict[str, Any]:
    return {"candidates": list(candidates)}


def gemini_stop_candidate(*parts: dict[str, Any]) -> dict[str, Any]:
    return {"finishReason": "STOP", "content": {"parts": list(parts)}}


def gemini_thought_then_pause_then_text_handler(
    bodies: list[dict[str, Any]] | None = None,
) -> Callable[[dict[str, Any], int, httpx.Request], httpx.Response]:
    """Shared Gemini fake POST sequence: thought → pause tool → text (r3 item03 / r4 item1)."""

    def handler(_body: dict[str, Any], n: int, req: httpx.Request) -> httpx.Response:
        if bodies is not None:
            bodies.append(_body)
        if n == 1:
            payload = gemini_candidates_payload(
                gemini_stop_candidate({"text": "thinking…", "thought": True})
            )
        elif n == 2:
            payload = gemini_candidates_payload(
                gemini_stop_candidate({"functionCall": {"name": "spotify_pause", "args": {}}})
            )
        else:
            payload = gemini_candidates_payload(
                gemini_stop_candidate({"text": "Playback is paused."})
            )
        return httpx.Response(200, json=payload, request=req)

    return handler


def gemini_any_pause_then_auto_text_handler(
    pause_calls: dict[str, int],
) -> Callable[[dict[str, Any], int, httpx.Request], httpx.Response]:
    """Return pause on ANY rounds, plain text on AUTO (r4 item1)."""

    def handler(body: dict[str, Any], _n: int, req: httpx.Request) -> httpx.Response:
        fc_mode = (body.get("toolConfig") or {}).get("functionCallingConfig", {}).get("mode")
        if fc_mode == "ANY":
            pause_calls["n"] += 1
            payload = gemini_candidates_payload(
                gemini_stop_candidate({"functionCall": {"name": "spotify_pause", "args": {}}})
            )
        else:
            payload = gemini_candidates_payload(
                gemini_stop_candidate({"text": "Playback paused."})
            )
        return httpx.Response(200, json=payload, request=req)

    return handler


def mock_artist_albums(artist_id: str, items: list[dict[str, Any]]) -> None:
    respx.get(f"https://api.spotify.com/v1/artists/{artist_id}/albums").mock(
        return_value=httpx.Response(200, json={"items": items})
    )


def run_artist_latest_album_tool(artist_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    from spot_backend.config import Settings
    from spot_backend.spotify_tools import SpotifyToolRunner

    mock_artist_albums(artist_id, items)
    runner = SpotifyToolRunner(settings=Settings())
    runner._session_known_ids.add(artist_id)
    raw = runner.run("spotify_artist_latest_album", {"artist_id": artist_id})
    runner.close()
    return json.loads(raw)


def mock_spotify_active_device(device_id: str, *, name: str = "Desk") -> None:
    """Register GET /me/player/devices with one active, unrestricted device."""
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(
            200,
            json={
                "devices": [
                    {
                        "id": device_id,
                        "is_active": True,
                        "is_restricted": False,
                        "name": name,
                    },
                ]
            },
        )
    )


def mock_spotify_idle_player_state(*, player_json: dict[str, Any] | None = None) -> None:
    """Minimal mocks for a successful play + idle player poll."""
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(200, json=player_json or {"is_playing": True})
    )


def run_unknown_device_playback_fallback(
    *,
    saved_device_id: str,
    unknown_device_id: str = "device_123",
    settings: Any | None = None,
) -> tuple[dict[str, Any], list[Any]]:
    """Playback with bogus device_id falls back to saved/active device (r5/r6 device tests)."""
    from spot_backend.config import Settings
    from spot_backend.spotify_tools import SpotifyToolRunner
    from spot_backend.token_store import DeviceSelection, save_device

    cfg = settings or Settings()
    save_device(cfg.resolved_device_path, DeviceSelection(device_id=saved_device_id))
    mock_spotify_active_device(saved_device_id)
    mock_spotify_idle_player_state()
    runner = SpotifyToolRunner(settings=cfg)
    raw = runner.run("spotify_start_resume_playback", {"device_id": unknown_device_id})
    runner.close()
    play_requests = [c.request for c in respx.calls if "player/play" in str(c.request.url)]
    return json.loads(raw), play_requests


def run_playback_restriction_violated(track_id: str, *, album_id: str = "aaaaaaaaaaaaaaaaaaaaaa") -> dict[str, Any]:
    from spot_backend.config import Settings
    from spot_backend.spotify_tools import SpotifyToolRunner

    respx.get(f"https://api.spotify.com/v1/tracks/{track_id}").mock(
        return_value=httpx.Response(200, json={"id": track_id, "album": {"id": album_id}})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(
            403,
            json={"error": {"status": 403, "message": "Restriction violated"}},
        )
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player(\?.*)?$").mock(
        return_value=httpx.Response(204)
    )
    mock_spotify_active_device("retry_device_1")
    respx.get("https://api.spotify.com/v1/me/player").mock(return_value=httpx.Response(200, json={}))
    runner = SpotifyToolRunner(settings=Settings())
    runner._session_known_ids.add(track_id)
    raw = runner.run("spotify_start_resume_playback", {"uris": [f"spotify:track:{track_id}"]})
    runner.close()
    return json.loads(raw)


def make_gemini_post_recorder(
    handler: Callable[[dict[str, Any], int], httpx.Response],
) -> tuple[list[dict[str, Any]], Callable[..., httpx.Response]]:
    """Return (bodies, fake_post) for patching httpx.Client.post."""
    bodies: list[dict[str, Any]] = []

    def fake_post(_self, url, **kwargs):
        json_body = kwargs.get("json") or {}
        bodies.append(json_body)
        req = httpx.Request("POST", str(url))
        return handler(json_body, len(bodies), req)

    return bodies, fake_post


def install_cpu_only_ollama_profile_mocks(monkeypatch) -> None:
    import time

    def fake_get(url, *args, **kwargs):
        req = httpx.Request("GET", str(url))
        if str(url).endswith("/api/ps"):
            return httpx.Response(
                200,
                json={"models": [{"name": "qwen3:4b-instruct", "size_vram": 0}]},
                request=req,
            )
        raise AssertionError(url)

    def fake_post(url, *args, **kwargs):
        req = httpx.Request("POST", str(url))
        return httpx.Response(200, json={"response": "OK"}, request=req)

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(time, "perf_counter", lambda: 0.0)


def install_cpu_profile_http_mocks(
    monkeypatch,
    *,
    size_vram: int = 0,
    perf_steps: list[float] | None = None,
    http_calls: list[str] | None = None,
) -> None:
    import time

    steps = iter(perf_steps or [0.0])
    fallback = perf_steps[-1] if perf_steps else 0.0

    def fake_get(url, *args, **kwargs):
        if http_calls is not None:
            http_calls.append(str(url))
        req = httpx.Request("GET", str(url))
        if str(url).endswith("/api/ps"):
            return httpx.Response(
                200,
                json={"models": [{"name": "qwen3:4b-instruct", "size_vram": size_vram}]},
                request=req,
            )
        raise AssertionError(url)

    def fake_post(url, *args, **kwargs):
        if http_calls is not None:
            http_calls.append(str(url))
        req = httpx.Request("POST", str(url))
        return httpx.Response(200, json={"response": "OK"}, request=req)

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(time, "perf_counter", lambda: next(steps, fallback))


class FakeOllamaStream:
    """Minimal httpx stream stand-in shared by agent loop tests."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    def __enter__(self) -> FakeOllamaStream:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "err",
                request=httpx.Request("POST", "http://x"),
                response=httpx.Response(self.status_code),
            )

    def read(self) -> bytes:
        return b""

    def iter_lines(self):
        for line in self._lines:
            yield line


def run_create_playlist_private_flow(
    pid: str,
    *,
    post_public: bool,
    get_public: bool,
) -> tuple[dict[str, Any], bool]:
    from spot_backend.config import Settings
    from spot_backend.spotify_tools import SpotifyToolRunner

    respx.post("https://api.spotify.com/v1/me/playlists").mock(
        return_value=httpx.Response(200, json={"id": pid, "name": "x", "public": post_public})
    )
    put_route = respx.put(f"https://api.spotify.com/v1/playlists/{pid}").mock(
        return_value=httpx.Response(200)
    )
    respx.get(f"https://api.spotify.com/v1/playlists/{pid}").mock(
        return_value=httpx.Response(200, json={"id": pid, "public": get_public, "name": "x"})
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run(
        "spotify_create_playlist",
        {
            "name": "Secret",
            "tracks": [{"uri": "spotify:track:aaaaaaaaaaaaaaaaaaaaaa"}],
        },
    )
    runner.close()
    return json.loads(raw), put_route.called


def mock_player_album_context_playing(album_id: str) -> None:
    mock_spotify_idle_player_state(
        player_json={
            "is_playing": True,
            "context": {"uri": f"spotify:album:{album_id}"},
            "item": {},
        },
    )


def mock_player_track_playing(track_id: str) -> None:
    mock_spotify_idle_player_state(
        player_json={"is_playing": True, "item": {"uri": f"spotify:track:{track_id}"}},
    )


def run_verified_album_context_playback(album_id: str) -> dict[str, Any]:
    from spot_backend.config import Settings
    from spot_backend.spotify_tools import SpotifyToolRunner

    respx.get(f"https://api.spotify.com/v1/albums/{album_id}").mock(
        return_value=httpx.Response(200, json={"id": album_id, "name": "OK"})
    )
    mock_player_album_context_playing(album_id)
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_start_resume_playback", {"context_uri": f"spotify:album:{album_id}"})
    runner.close()
    return json.loads(raw)


def devices_api_get(
    *,
    spotify_response: httpx.Response | None = None,
    side_effect: BaseException | None = None,
) -> httpx.Response:
    """Call GET /api/devices after mocking Spotify's devices route (r2/pr2 dedupe)."""
    route = respx.get("https://api.spotify.com/v1/me/player/devices")
    if side_effect is not None:
        route.mock(side_effect=side_effect)
    else:
        route.mock(return_value=spotify_response or httpx.Response(500, json={"error": "boom"}))
    from spot_backend.app import app

    from fastapi.testclient import TestClient

    client = TestClient(app)
    return client.get("/api/devices")


def run_artist_null_context_playback(artist_id: str) -> dict[str, Any]:
    from spot_backend.config import Settings
    from spot_backend.spotify_tools import SpotifyToolRunner

    respx.get(f"https://api.spotify.com/v1/artists/{artist_id}").mock(
        return_value=httpx.Response(200, json={"id": artist_id, "name": "Artist"})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={
                "is_playing": True,
                "context": None,
                "item": {
                    "uri": "spotify:track:cccccccccccccccccccccc",
                    "artists": [{"id": artist_id, "name": "Artist"}],
                },
            },
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    runner._session_known_ids.add(artist_id)
    raw = runner.run(
        "spotify_start_resume_playback",
        {"context_uri": f"spotify:artist:{artist_id}"},
    )
    runner.close()
    return json.loads(raw)


def run_album_playlist_play(album_id: str) -> dict[str, Any]:
    from spot_backend.config import Settings
    from spot_backend.spotify_tools import SpotifyToolRunner

    mock_player_album_context_playing(album_id)
    runner = SpotifyToolRunner(settings=Settings())
    runner._session_known_ids.add(album_id)
    raw = runner.run("spotify_play_playlist", {"playlist_id": f"spotify:album:{album_id}"})
    runner.close()
    return json.loads(raw)
