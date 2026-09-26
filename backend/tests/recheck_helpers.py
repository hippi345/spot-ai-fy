"""Shared fixtures/helpers for r2/r3/r4 recheck tests (dedupe for pylint R0801)."""

from __future__ import annotations

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
