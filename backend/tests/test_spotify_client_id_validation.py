from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from spot_backend.app import app
from spot_backend.spotify_setup import validate_spotify_client_id


def test_validate_spotify_client_id_ok() -> None:
    cid = "a" * 32
    assert validate_spotify_client_id(cid) == cid


def test_validate_spotify_client_id_rejects_bad_chars() -> None:
    with pytest.raises(ValueError, match="32 alphanumeric"):
        validate_spotify_client_id("not-valid/file:..")


def test_empty_client_id_returns_400_not_422(data_dir) -> None:
    client = TestClient(app)
    r = client.post("/api/setup/spotify-app", json={"client_id": ""})
    assert r.status_code == 400
    assert r.json()["detail"] == "client_id is required"
