from __future__ import annotations

import json
import logging
import os
import stat

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from spot_backend.app import app
from spot_backend.llm_prefs import prefs_path_exists, read_effective_llm_provider
from spot_backend.secrets_store import _secrets_path, read_secret, read_setup_fields, write_secret


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
    r = client.post("/api/setup/spotify-app", json={"client_id": "b" * 32})
    assert r.status_code == 200
    body = r.json()
    assert body["spotify_client_id_masked"] == "••••••••"
    assert "bbbbbbbb" not in r.text
    assert "bbbbbbbb" not in caplog.text


def _assert_gemini_success_followups(
    client: TestClient, secret: str, caplog: pytest.LogCaptureFixture
) -> None:
    assert secret not in caplog.text
    status = client.get("/api/setup/status")
    assert status.status_code == 200
    body = status.json()
    assert body["llm_ready"] is True
    assert body["provider"] == "gemini"
    assert secret not in status.text
    assert "AIza" not in status.text


@respx.mock
def test_setup_llm_gemini_keyring_path(
    data_dir,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stored: dict[str, str] = {}

    def fake_set(key: str, value: str) -> bool:
        stored[key] = value
        return True

    def fake_get(key: str) -> str | None:
        return stored.get(key)

    monkeypatch.setattr("spot_backend.secrets_store._keyring_set", fake_set)
    monkeypatch.setattr("spot_backend.secrets_store._keyring_get", fake_get)
    caplog.set_level(logging.DEBUG, logger="spot_backend")

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
    _assert_gemini_success_followups(client, secret, caplog)


@respx.mock
def test_setup_llm_gemini_file_fallback_0600(
    data_dir,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr("spot_backend.secrets_store._keyring_set", lambda _k, _v: False)
    monkeypatch.setattr("spot_backend.secrets_store._keyring_get", lambda _k: None)
    caplog.set_level(logging.DEBUG, logger="spot_backend")

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
    if os.name != "nt":
        assert oct(path.stat().st_mode & 0o777) == oct(stat.S_IRUSR | stat.S_IWUSR)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["gemini_api_key"] == secret
    assert secret not in r.text
    _assert_gemini_success_followups(client, secret, caplog)


@respx.mock
def test_setup_llm_gemini_rejected_does_not_persist_key(
    data_dir, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored: dict[str, str] = {}

    monkeypatch.setattr(
        "spot_backend.secrets_store._keyring_set",
        lambda key, value: stored.update({key: value}) or True,
    )
    monkeypatch.setattr("spot_backend.secrets_store._keyring_get", lambda key: stored.get(key))

    respx.get(url__regex=r"https://generativelanguage\.googleapis\.com/v1beta/models.*").mock(
        return_value=httpx.Response(403, json={"error": {"message": "API key not valid"}})
    )

    secret = "AIzaSyREJECTED_KEY_TEST"
    r = client.post(
        "/api/setup/llm",
        json={"provider": "gemini", "gemini_api_key": secret, "test": True},
    )
    assert r.status_code == 400
    assert secret not in r.text
    assert read_secret(data_dir, "gemini_api_key") == ""
    assert secret not in stored
    assert not _secrets_path(data_dir).is_file()
    assert read_effective_llm_provider(data_dir, "ollama") == "ollama"


@respx.mock
def test_setup_llm_ollama_unreachable_does_not_persist(
    data_dir, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "")
    respx.get("http://127.0.0.1:19999/api/tags").mock(side_effect=httpx.ConnectError("connection refused"))

    r = client.post(
        "/api/setup/llm",
        json={
            "provider": "ollama",
            "ollama_host": "http://127.0.0.1:19999",
            "ollama_model": "gemma2:2b",
            "test": True,
        },
    )
    assert r.status_code == 400
    assert "connection refused" in r.text.lower() or "Connect" in r.text
    setup = read_setup_fields(data_dir)
    assert setup.get("ollama_host") is None
    assert not prefs_path_exists(data_dir)


@respx.mock
def test_setup_llm_ollama(data_dir, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "")
    respx.get("http://127.0.0.1:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "gemma2:2b"}]})
    )
    respx.get("http://127.0.0.1:11434/api/ps").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "gemma2:2b", "size_vram": 0}]})
    )
    respx.post("http://127.0.0.1:11434/api/generate").mock(
        return_value=httpx.Response(200, json={"response": "OK"})
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


@respx.mock
def test_setup_ollama_probe_reachable(data_dir, client: TestClient) -> None:
    respx.get("http://127.0.0.1:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "gemma2:2b"}]})
    )
    r = client.get("/api/setup/ollama/probe", params={"host": "http://127.0.0.1:11434"})
    assert r.status_code == 200
    body = r.json()
    assert body["reachable"] is True
    assert body["models"] == ["gemma2:2b"]
    assert body["error"] is None


@respx.mock
def test_setup_ollama_probe_unreachable(data_dir, client: TestClient) -> None:
    respx.get("http://127.0.0.1:19999/api/tags").mock(side_effect=httpx.ConnectError("down"))
    r = client.get("/api/setup/ollama/probe", params={"host": "http://127.0.0.1:19999"})
    assert r.status_code == 200
    body = r.json()
    assert body["reachable"] is False
    assert body["models"] == []
    assert body["error"]


@respx.mock
def test_setup_probe_blocks_metadata_host(data_dir, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import ipaddress

    monkeypatch.setattr(
        "spot_backend.url_safety.resolve_host_ips",
        lambda _h: [ipaddress.ip_address("169.254.169.254")],
    )
    r = client.get("/api/setup/ollama/probe", params={"host": "http://169.254.169.254"})
    assert r.status_code == 200
    body = r.json()
    assert body["reachable"] is False
    assert "blocked" in (body["error"] or "").lower()


def test_write_secret_never_logged(
    data_dir, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("spot_backend.secrets_store._keyring_set", lambda _k, _v: False)
    monkeypatch.setattr("spot_backend.secrets_store._keyring_get", lambda _k: None)
    logging.getLogger("spot_backend.secrets_store").setLevel(logging.DEBUG)
    secret = "AIzaSyLOG_TEST_SECRET"
    write_secret(data_dir, "gemini_api_key", secret)
    assert secret not in caplog.text
