"""Shared fixtures/helpers for r2/r3/r4 recheck tests (dedupe for pylint R0801)."""

from __future__ import annotations

from typing import Any, Callable

import httpx
import respx


def mock_artist_name_search(artist_id: str, name: str = "Radiohead") -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search\?.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "artists": {
                    "items": [{"id": artist_id, "name": name, "type": "artist"}],
                }
            },
        )
    )


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
