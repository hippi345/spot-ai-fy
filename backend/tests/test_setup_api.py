from __future__ import annotations

import json
import stat

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from spot_backend.app import app
from spot_backend.secrets_store import _secrets_path, read_secret, write_secret


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_setup_status_unsigned(data_dir, client: TestClient) -> None:
    r = client.get("/api/setup/status")
    assert r.status_code == 200
    body = r.json()
    assert body["spotify_configured"] is True  # from conftest SPOTIFY_CLIENT_ID
    assert body["spotify_signed_in"] is False
    assert "redirect_uri" in body
    assert "gemini_api_key" not in body
    assert "AIza" not in r.text


@respx.mock
def test_setup_spotify_app_masks_client_id(
    data_dir, client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "")
    r = client.post("/api/setup/spotify-app", json={"client_id": "my-secret-client-id-12345"})
    assert r.status_code == 200
    body = r.json()
    assert body["spotify_client_id_masked"] == "••••••••"
    assert "my-secret-client-id" not in r.text
    assert "my-secret-client-id" not in caplog.text


@respx.mock
def test_setup_llm_gemini_keyring_path(data_dir, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    stored: dict[str, str] = {}

    def fake_set(key: str, value: str) -> bool:
        stored[key] = value
        return True

    def fake_get(key: str) -> str | None:
        return stored.get(key)

    monkeypatch.setattr("spot_backend.secrets_store._keyring_set", fake_set)
    monkeypatch.setattr("spot_backend.secrets_store._keyring_get", fake_get)

    respx.get(url__regex=r"https://generativelanguage\.googleapis\.com/v1beta/models.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/gemini-2.5-flash",
                        "supportedGenerationMethods": ["generateContent"],
                    }
                ]
            },
        )
    )

    secret = "AIzaSyDUMMY_KEY_FOR_TESTS_ONLY"
    r = client.post(
        "/api/setup/llm",
        json={"provider": "gemini", "gemini_api_key": secret, "test": True},
    )
    assert r.status_code == 200
    assert secret not in r.text
    assert read_secret(data_dir, "gemini_api_key") == secret
    assert not _secrets_path(data_dir).is_file()


@respx.mock
def test_setup_llm_gemini_file_fallback_0600(
    data_dir, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("spot_backend.secrets_store._keyring_set", lambda _k, _v: False)
    monkeypatch.setattr("spot_backend.secrets_store._keyring_get", lambda _k: None)

    respx.get(url__regex=r"https://generativelanguage\.googleapis\.com/v1beta/models.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/gemini-2.5-flash",
                        "supportedGenerationMethods": ["generateContent"],
                    }
                ]
            },
        )
    )

    secret = "AIzaSyFILE_FALLBACK_KEY"
    r = client.post(
        "/api/setup/llm",
        json={"provider": "gemini", "gemini_api_key": secret, "test": True},
    )
    assert r.status_code == 200
    path = _secrets_path(data_dir)
    assert path.is_file()
    assert oct(path.stat().st_mode & 0o777) == oct(stat.S_IRUSR | stat.S_IWUSR)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["gemini_api_key"] == secret
    assert secret not in r.text


@respx.mock
def test_setup_llm_ollama(data_dir, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "")
    respx.get("http://127.0.0.1:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "gemma2:2b"}]})
    )
    r = client.post(
        "/api/setup/llm",
        json={
            "provider": "ollama",
            "ollama_host": "http://127.0.0.1:11434",
            "ollama_model": "gemma2:2b",
            "test": True,
        },
    )
    assert r.status_code == 200
    assert r.json()["reachable"] is True


def test_write_secret_never_logged(
    data_dir, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    monkeypatch.setattr("spot_backend.secrets_store._keyring_set", lambda _k, _v: False)
    monkeypatch.setattr("spot_backend.secrets_store._keyring_get", lambda _k: None)
    logging.getLogger("spot_backend.secrets_store").setLevel(logging.DEBUG)
    secret = "AIzaSyLOG_TEST_SECRET"
    write_secret(data_dir, "gemini_api_key", secret)
    assert secret not in caplog.text
