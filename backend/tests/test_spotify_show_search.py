"""Show/episode search param normalization and null-safe parsing."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from spot_backend.config import Settings
from spot_backend.spotify_search_params import build_spotify_search_params, normalize_search_type_tokens
from spot_backend.spotify_tools import SpotifyToolRunner


def test_normalize_search_types_shows_plural_to_show() -> None:
    assert normalize_search_type_tokens("shows") == "show"
    assert normalize_search_type_tokens("show,track") == "show,track"


def test_build_search_params_includes_market_and_include_external_for_show() -> None:
    params = build_spotify_search_params(
        query="astronomy",
        types_raw="shows",
        market="",
        limit=5,
        offset=0,
    )
    assert params["type"] == "show"
    assert params["market"] == "from_token"
    assert params["include_external"] == "audio"
    assert params["limit"] == 5


@respx.mock
def test_spotify_search_show_rejects_plural_type_on_wire(data_dir, signed_in_tokens) -> None:
    """Root cause: models send types=shows → API type=shows → HTTP 400 invalid type."""
    bad_body = {
        "error": {
            "status": 400,
            "message": "Invalid type: shows. Allowed values: album, artist, playlist, track, show, episode, audiobook",
        }
    }
    captured: list[dict] = []

    def _assert_good_request(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        captured.append(params)
        if params.get("type") == "shows":
            return httpx.Response(400, json=bad_body)
        return httpx.Response(
            200,
            json={
                "shows": {
                    "items": [
                        None,
                        {
                            "id": "4rOoJ6Egrf8K2IrywzwOMy",
                            "name": "StarTalk Radio",
                            "publisher": "StarTalk",
                            "uri": "spotify:show:4rOoJ6Egrf8K2IrywzwOMy",
                        },
                    ],
                    "total": 1,
                }
            },
        )

    respx.get(url__regex=r"https://api\.spotify\.com/v1/search.*").mock(side_effect=_assert_good_request)
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run(
        "spotify_search",
        {"query": "astronomy podcast", "types": "shows", "limit": 5},
    )
    runner.close()
    data = json.loads(raw)
    assert captured[-1]["type"] == "show"
    assert "StarTalk Radio" in json.dumps(data)
    shows = data["shows"]["items"]
    assert len(shows) == 1
    assert shows[0]["id"] == "4rOoJ6Egrf8K2IrywzwOMy"
    assert shows[0]["publisher"] == "StarTalk"
    assert data.get("user_message")


@respx.mock
def test_spotify_search_show_400_surfaces_spotify_body(data_dir, signed_in_tokens) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search.*").mock(
        return_value=httpx.Response(
            400,
            json={"error": {"status": 400, "message": "Invalid market"}},
        )
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_search", {"query": "x", "types": "show", "market": "ZZZ"})
    runner.close()
    data = json.loads(raw)
    assert data.get("ok") is False
    assert data.get("failure_reason")
    assert data.get("spotify_error_body_redacted") or "Invalid market" in data.get("error", "")
