"""Stateful Spotify API mock for docs/SMOKE_TEST.md steps 1–7."""

from __future__ import annotations

from typing import Any

import httpx
import respx

ARTIST_ID = "jjjjjjjjjjjjjjjjjjjjjj"
TRACK_A = "aaaaaaaaaaaaaaaaaaaaa1"
TRACK_B = "aaaaaaaaaaaaaaaaaaaaa2"
GRAVITY_ID = "ggggggggggggggggggggg1"


def _track(
    tid: str,
    name: str,
    album_id: str = "albumcontinuum00000001",
) -> dict[str, Any]:
    return {
        "id": tid,
        "uri": f"spotify:track:{tid}",
        "name": name,
        "duration_ms": 210_000,
        "artists": [{"name": "John Mayer"}],
        "album": {
            "id": album_id,
            "name": "Continuum",
            "images": [
                {"url": "https://i.test/l.jpg", "width": 300, "height": 300},
                {"url": "https://i.test/s.jpg", "width": 64, "height": 64},
            ],
        },
    }


class SpotifyPlaybackState:
    def __init__(self) -> None:
        self.tracks = [
            _track(TRACK_A, "Slow Dancing in a Burning Room"),
            _track(TRACK_B, "Waiting on the World to Change"),
        ]
        self.gravity = _track(GRAVITY_ID, "Gravity")
        self.index = 0
        self.is_playing = False
        self.queue: list[dict[str, Any]] = []
        self.playlists = [
            {"id": "plist000000000000000001", "name": "Evening Acoustic"},
            {"id": "plist000000000000000002", "name": "Road Trip Mix"},
        ]
        self._started = False

    def current(self) -> dict[str, Any]:
        return self.tracks[self.index]

    def player_payload(self) -> dict[str, Any] | None:
        if not self._started:
            return None
        return {
            "is_playing": self.is_playing,
            "progress_ms": 5000,
            "item": self.current(),
            "device": {"id": "dev-smoke", "name": "Smoke Speaker", "type": "Speaker"},
        }

    def start_playback(self) -> None:
        self._started = True
        self.is_playing = True

    def skip_next(self) -> None:
        self.index = min(self.index + 1, len(self.tracks) - 1)
        self.is_playing = True

    def pause(self) -> None:
        self.is_playing = False

    def resume(self) -> None:
        self.is_playing = True

    def add_gravity_to_queue(self) -> None:
        self.queue.append(self.gravity)


def install_stateful_spotify_mock(state: SpotifyPlaybackState) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/search"):
            q = request.url.params.get("q", "")
            if request.url.params.get("type") == "track" or "Gravity" in q:
                items = [state.gravity] if "Gravity" in q else state.tracks
                return httpx.Response(200, json={"tracks": {"items": items}})
            return httpx.Response(
                200,
                json={"artists": {"items": [{"id": ARTIST_ID, "name": "John Mayer"}]}},
            )
        if f"/artists/{ARTIST_ID}" in path and path.endswith(f"/artists/{ARTIST_ID}"):
            return httpx.Response(200, json={"id": ARTIST_ID, "name": "John Mayer"})
        if f"/artists/{ARTIST_ID}/top-tracks" in path:
            return httpx.Response(
                200,
                json={"tracks": state.tracks},
            )
        for tr in state.tracks + [state.gravity]:
            if path.endswith(f"/tracks/{tr['id']}"):
                return httpx.Response(200, json=tr)
            if path.endswith(f"/albums/{tr['album']['id']}/tracks"):
                return httpx.Response(200, json={"items": [{"uri": tr["uri"]}]})
            if path.endswith(f"/albums/{tr['album']['id']}"):
                return httpx.Response(
                    200,
                    json={"id": tr["album"]["id"], "uri": f"spotify:album:{tr['album']['id']}"},
                )

        if path.endswith("/me/player/devices"):
            return httpx.Response(
                200,
                json={
                    "devices": [
                        {
                            "id": "dev-smoke",
                            "is_active": True,
                            "is_restricted": False,
                            "name": "Smoke Speaker",
                        }
                    ]
                },
            )
        if path.endswith("/me/player/play") or path.endswith("/me/player"):
            if request.method == "PUT" and "play" in path:
                state.start_playback()
                return httpx.Response(204)
        if path.endswith("/me/player/pause"):
            state.pause()
            return httpx.Response(204)
        if path.endswith("/me/player/next"):
            state.skip_next()
            return httpx.Response(204)
        if path.endswith("/me/player/queue") and request.method == "POST":
            state.add_gravity_to_queue()
            return httpx.Response(204)
        if path.endswith("/me/player/queue") and request.method == "GET":
            return httpx.Response(
                200,
                json={"queue": [{"track": t} for t in state.queue]},
            )
        if path.endswith("/me/player"):
            payload = state.player_payload()
            if payload is None:
                return httpx.Response(204)
            return httpx.Response(200, json=payload)
        if path.endswith("/me/playlists"):
            return httpx.Response(200, json={"items": state.playlists})
        return httpx.Response(200, json={})

    respx.route(url__regex=r"https://api\.spotify\.com/.*").mock(side_effect=handler)
