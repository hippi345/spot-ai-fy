from __future__ import annotations

from fastapi.testclient import TestClient

from spot_backend.app import app
from spot_backend.token_store import DeviceSelection, save_device


def test_health() -> None:
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_session_unsigned(data_dir) -> None:
    client = TestClient(app)
    r = client.get("/api/session")
    assert r.status_code == 200
    body = r.json()
    assert body["signed_in"] is False
    assert body["device_id"] is None


def test_session_signed_with_device(data_dir, signed_in_tokens) -> None:
    save_device(data_dir / "device.json", DeviceSelection(device_id="dev-1"))
    client = TestClient(app)
    r = client.get("/api/session")
    assert r.status_code == 200
    body = r.json()
    assert body["signed_in"] is True
    assert body["device_id"] == "dev-1"
    assert body["spotify_playlist_write_ok"] is True
