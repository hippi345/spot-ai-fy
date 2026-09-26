from __future__ import annotations

import json

import httpx
import respx

from spot_backend.config import Settings
from spot_backend.spotify_client import SpotifyClient, SpotifyRateLimitError
from spot_backend.spotify_tools import SpotifyToolRunner


@respx.mock
def test_spotify_429_then_200_retries_once_with_capped_sleep(data_dir, signed_in_tokens) -> None:
    settings = Settings()
    client = SpotifyClient(settings=settings)
    slept: list[float] = []
    client._rate_limit_sleep = lambda seconds: slept.append(seconds)
    calls = {"n": 0}

    def search_handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "2"},
                json={"error": {"status": 429, "message": "rate"}},
            )
        return httpx.Response(200, json={"tracks": {"items": []}})

    respx.get(url__regex=r"https://api\.spotify\.com/v1/search.*").mock(side_effect=search_handler)

    data = client.api_get("/search", params={"q": "a", "type": "track", "limit": 1})
    assert data == {"tracks": {"items": []}}
    assert calls["n"] == 2
    assert slept == [2.0]
    client.close()


@respx.mock
def test_spotify_429_twice_raises_rate_limit_message(data_dir, signed_in_tokens) -> None:
    settings = Settings()
    client = SpotifyClient(settings=settings)
    client._rate_limit_sleep = lambda _seconds: None

    respx.get(url__regex=r"https://api\.spotify\.com/v1/search.*").mock(
        return_value=httpx.Response(
            429,
            headers={"Retry-After": "7"},
            json={"error": {"status": 429, "message": "rate"}},
        )
    )

    try:
        try:
            client.api_get("/search", params={"q": "a", "type": "track", "limit": 1})
            raise AssertionError("expected SpotifyRateLimitError")
        except SpotifyRateLimitError as e:
            assert str(e) == "Spotify rate limited, try again in 7 s"
    finally:
        client.close()


@respx.mock
def test_spotify_429_huge_retry_after_sleeps_at_cap(data_dir, signed_in_tokens) -> None:
    settings = Settings()
    client = SpotifyClient(settings=settings)
    slept: list[float] = []
    client._rate_limit_sleep = lambda seconds: slept.append(seconds)
    calls = {"n": 0}

    def search_handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "3600"},
                json={"error": {"status": 429, "message": "rate"}},
            )
        return httpx.Response(200, json={"tracks": {"items": []}})

    respx.get(url__regex=r"https://api\.spotify\.com/v1/search.*").mock(side_effect=search_handler)

    client.api_get("/search", params={"q": "a", "type": "track", "limit": 1})
    assert slept == [5.0]
    client.close()


@respx.mock
def test_spotify_search_double_429_via_tool_runner(data_dir, signed_in_tokens) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search.*").mock(
        return_value=httpx.Response(
            429,
            headers={"Retry-After": "12"},
            json={"error": {"status": 429, "message": "API rate limit"}},
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    try:
        raw = runner.run("spotify_search", {"query": "x", "types": "track"})
        data = json.loads(raw)
        assert data == {"error": "Spotify rate limited, try again in 12 s"}
    finally:
        runner.close()


@respx.mock
def test_spotify_search_429_retry_succeeds(data_dir, signed_in_tokens) -> None:
    calls = {"n": 0}

    def search_handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "1"},
                json={"error": {"status": 429, "message": "rate"}},
            )
        return httpx.Response(
            200,
            json={"tracks": {"items": [{"type": "track", "id": "1111111111111111111111", "name": "Ok"}]}},
        )

    respx.get(url__regex=r"https://api\.spotify\.com/v1/search.*").mock(side_effect=search_handler)
    runner = SpotifyToolRunner(settings=Settings())
    try:
        raw = runner.run("spotify_search", {"query": "ok", "types": "track"})
        data = json.loads(raw)
        assert data["tracks"]["items"][0]["name"] == "Ok"
        assert calls["n"] == 2
    finally:
        runner.close()
