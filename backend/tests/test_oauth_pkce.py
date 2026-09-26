from __future__ import annotations

import base64
import hashlib
import urllib.parse
from pathlib import Path

import httpx
import respx
from fastapi.testclient import TestClient

from spot_backend.app import _pkce_pending, app
from spot_backend.config import Settings
from spot_backend.spotify_client import SpotifyClient


def test_login_redirect_pkce_params(data_dir) -> None:
    client = TestClient(app)
    r = client.get("/login", follow_redirects=False)
    assert r.status_code == 307
    loc = r.headers["location"]
    assert loc.startswith("https://accounts.spotify.com/authorize?")
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(loc).query)
    assert qs["client_id"] == ["test-spotify-client-id"]
    assert qs["redirect_uri"] == ["http://127.0.0.1:8765/callback"]
    assert qs["response_type"] == ["code"]
    assert qs["code_challenge_method"] == ["S256"]
    assert "user-read-private" in qs["scope"][0]
    state = qs["state"][0]
    challenge = qs["code_challenge"][0]
    assert state in _pkce_pending
    verifier = _pkce_pending[state]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    expected = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    assert challenge == expected


def test_login_missing_spotify_client_id(data_dir, monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "")
    client = TestClient(app)
    r = client.get("/login", follow_redirects=False)
    assert r.status_code == 400
    body = r.json()
    assert "SPOTIFY_CLIENT_ID" in body["detail"]


@respx.mock
def test_callback_exchanges_code_and_stores_tokens(data_dir, token_path: Path) -> None:
    client = TestClient(app)
    login = client.get("/login", follow_redirects=False)
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(login.headers["location"]).query)
    state = qs["state"][0]

    respx.post("https://accounts.spotify.com/api/token").mock(
        return_value=httpx.Response(
            200,
            json={
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
                "scope": "user-read-private",
            },
        )
    )

    r = client.get(
        "/callback",
        params={"code": "auth-code-xyz", "state": state},
        follow_redirects=False,
    )
    assert r.status_code == 307
    assert "spotify=connected" in r.headers["location"]
    assert token_path.is_file()
    raw = token_path.read_text(encoding="utf-8")
    assert "new-access" in raw
    assert "new-refresh" in raw
    assert state not in _pkce_pending


@respx.mock
def test_api_get_refreshes_on_401(data_dir, signed_in_tokens) -> None:
    settings = Settings()
    client = SpotifyClient(settings=settings)
    call_count = {"n": 0}

    def search_handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(401, json={"error": "expired"})
        return httpx.Response(200, json={"tracks": {"items": []}})

    respx.get(url__regex=r"https://api\.spotify\.com/v1/search.*").mock(side_effect=search_handler)
    respx.post("https://accounts.spotify.com/api/token").mock(
        return_value=httpx.Response(
            200,
            json={
                "access_token": "refreshed-access",
                "expires_in": 3600,
                "scope": settings.spotify_client_id and "user-read-private",
            },
        )
    )

    data = client.api_get("/search", params={"q": "test", "type": "track", "limit": 1})
    assert isinstance(data, dict)
    assert call_count["n"] == 2
    bundle = client.load_bundle()
    assert bundle is not None
    assert bundle.access_token == "refreshed-access"
    client.close()


def test_exchange_code_without_client_id_raises(data_dir, monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "")
    settings = Settings()
    sc = SpotifyClient(settings=settings)
    try:
        from spot_backend.spotify_client import SpotifyAuthError

        try:
            sc.exchange_authorization_code("code", redirect_uri="http://127.0.0.1:8765/callback", code_verifier="v")
            raise AssertionError("expected SpotifyAuthError")
        except SpotifyAuthError as e:
            assert "SPOTIFY_CLIENT_ID" in str(e)
    finally:
        sc.close()
