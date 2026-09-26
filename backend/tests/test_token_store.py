from __future__ import annotations

import time
from pathlib import Path

from spot_backend.token_store import (
    DeviceSelection,
    TokenBundle,
    is_expired,
    load_device,
    load_tokens,
    save_device,
    save_tokens,
)


def test_token_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "tokens.json"
    bundle = TokenBundle(
        access_token="at",
        refresh_token="rt",
        expires_at=time.time() + 120,
        scope="user-read-private",
    )
    save_tokens(path, bundle)
    loaded = load_tokens(path)
    assert loaded is not None
    assert loaded.access_token == "at"
    assert loaded.refresh_token == "rt"
    assert loaded.scope == "user-read-private"


def test_load_tokens_missing_file(tmp_path: Path) -> None:
    assert load_tokens(tmp_path / "nope.json") is None


def test_is_expired_with_skew() -> None:
    future = TokenBundle(access_token="a", expires_at=time.time() + 120)
    assert not is_expired(future, skew_seconds=60)
    past = TokenBundle(access_token="a", expires_at=time.time() - 10)
    assert is_expired(past, skew_seconds=60)


def test_device_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "device.json"
    save_device(path, DeviceSelection(device_id="abc123device"))
    loaded = load_device(path)
    assert loaded is not None
    assert loaded.device_id == "abc123device"
