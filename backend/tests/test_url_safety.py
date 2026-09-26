from __future__ import annotations

import ipaddress

import pytest

from spot_backend.url_safety import OllamaUrlNotAllowedError, validate_ollama_base_url


def test_ollama_url_loopback_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "spot_backend.url_safety.resolve_host_ips",
        lambda _h: [ipaddress.ip_address("127.0.0.1")],
    )
    assert validate_ollama_base_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"


def test_ollama_url_rfc1918_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "spot_backend.url_safety.resolve_host_ips",
        lambda _h: [ipaddress.ip_address("192.168.1.50")],
    )
    assert validate_ollama_base_url("http://192.168.1.50:11434") == "http://192.168.1.50:11434"


def test_ollama_url_metadata_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "spot_backend.url_safety.resolve_host_ips",
        lambda _h: [ipaddress.ip_address("169.254.169.254")],
    )
    with pytest.raises(OllamaUrlNotAllowedError, match="blocked"):
        validate_ollama_base_url("http://169.254.169.254")


def test_ollama_url_link_local_hostname_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "spot_backend.url_safety.resolve_host_ips",
        lambda _h: [ipaddress.ip_address("169.254.1.2")],
    )
    with pytest.raises(OllamaUrlNotAllowedError, match="blocked"):
        validate_ollama_base_url("http://metadata.local")


def test_ollama_url_public_requires_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "spot_backend.url_safety.resolve_host_ips",
        lambda _h: [ipaddress.ip_address("8.8.8.8")],
    )
    with pytest.raises(OllamaUrlNotAllowedError, match="Confirm external"):
        validate_ollama_base_url("http://ollama.example.com", allow_public=False)
    assert (
        validate_ollama_base_url("http://ollama.example.com", allow_public=True)
        == "http://ollama.example.com"
    )
