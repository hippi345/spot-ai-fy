from __future__ import annotations

import json
import os

import httpx
import pytest
import respx

from spot_backend.config import Settings
from spot_backend.library_index import LibraryIndex, library_index_enabled, search_my_library_tool
from spot_backend.ollama_embed import HashingEmbedder
from spot_backend.spotify_tools import SEARCH_MY_LIBRARY_TOOL, ollama_tools_for_agent


@pytest.fixture
def index_settings(data_dir, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("SPOT_LIBRARY_INDEX", "on")
    return Settings(data_dir=data_dir)


@respx.mock
def test_build_search_and_incremental(index_settings: Settings, signed_in_tokens) -> None:
    respx.post("https://accounts.spotify.com/api/token").mock(
        return_value=httpx.Response(200, json={"access_token": "t", "expires_in": 3600})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/tracks.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "track": {
                            "id": "t1" + "x" * 18,
                            "uri": "spotify:track:t1" + "x" * 18,
                            "name": "Rainy Day",
                            "artists": [{"name": "Adele"}],
                            "album": {"name": "25", "release_date": "2015"},
                        }
                    }
                ],
                "total": 1,
            },
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/playlists.*").mock(
        return_value=httpx.Response(200, json={"items": [], "total": 0})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/following.*").mock(
        return_value=httpx.Response(200, json={"artists": {"items": [], "total": 0}})
    )
    idx = LibraryIndex(settings=index_settings, embedder=HashingEmbedder())
    idx.sync_full()
    out = idx.search("rainy", limit=5)
    assert out.get("available") is True
    assert out.get("results")


@respx.mock
def test_fts_fallback_when_embed_fails(
    index_settings: Settings, signed_in_tokens, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BoomEmbedder(HashingEmbedder):
        def embed_query(self, text: str) -> list[float]:
            raise RuntimeError("embed down")

    respx.post("https://accounts.spotify.com/api/token").mock(
        return_value=httpx.Response(200, json={"access_token": "t", "expires_in": 3600})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/tracks.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "track": {
                            "id": "t2" + "x" * 18,
                            "uri": "spotify:track:t2" + "x" * 18,
                            "name": "Jazz Night",
                            "artists": [{"name": "Coltrane"}],
                            "album": {"name": "Blue", "release_date": "1960"},
                        }
                    }
                ]
            },
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/playlists.*").mock(
        return_value=httpx.Response(200, json={"items": []})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/following.*").mock(
        return_value=httpx.Response(200, json={"artists": {"items": []}})
    )
    idx = LibraryIndex(settings=index_settings, embedder=BoomEmbedder())
    idx.sync_full()
    payload = idx.search("jazz", limit=3)
    assert payload.get("mode") == "keyword"


def test_flag_off_hides_tool(data_dir, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPOT_LIBRARY_INDEX", raising=False)
    s = Settings(data_dir=data_dir)
    assert not library_index_enabled(s)
    names = {t["function"]["name"] for t in ollama_tools_for_agent(s)}
    assert "search_my_library" not in names
    assert SEARCH_MY_LIBRARY_TOOL["function"]["name"] == "search_my_library"


def test_tool_unavailable_when_disabled(data_dir, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPOT_LIBRARY_INDEX", "off")
    out = search_my_library_tool("chill", settings=Settings(data_dir=data_dir))
    assert out["available"] is False
