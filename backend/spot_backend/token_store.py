from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TokenBundle(BaseModel):
    access_token: str
    refresh_token: str = ""
    expires_at: float = 0.0
    # Space-separated scopes Spotify returned at token exchange / refresh (may be empty for old files).
    scope: str = ""


class DeviceSelection(BaseModel):
    device_id: str = Field(..., description="Spotify device id for playback commands")


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_tokens(path: Path) -> TokenBundle | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TokenBundle.model_validate(raw)


def save_tokens(path: Path, bundle: TokenBundle) -> None:
    _atomic_write(path, bundle.model_dump())


def load_device(path: Path) -> DeviceSelection | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        raw = json.loads(text)
        return DeviceSelection.model_validate(raw)
    except (json.JSONDecodeError, ValueError, OSError):
        return None


def save_device(path: Path, selection: DeviceSelection) -> None:
    _atomic_write(path, selection.model_dump())


def is_expired(bundle: TokenBundle, skew_seconds: int = 60) -> bool:
    if bundle.expires_at <= 0:
        return False
    return time.time() >= bundle.expires_at - skew_seconds
