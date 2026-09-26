from __future__ import annotations

import time
from pathlib import Path

import pytest

from spot_backend.spotify_client import DEFAULT_SCOPES
from spot_backend.token_store import TokenBundle, save_tokens


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated DATA_DIR; never touches ~/.spot_ai_fy."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "a" * 32)
    monkeypatch.setenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8765/callback")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    return tmp_path


@pytest.fixture
def token_path(data_dir: Path) -> Path:
    return data_dir / "tokens.json"


@pytest.fixture
def signed_in_tokens(token_path: Path) -> Path:
    save_tokens(
        token_path,
        TokenBundle(
            access_token="test-access-token",
            refresh_token="test-refresh-token",
            expires_at=time.time() + 3600,
            scope=DEFAULT_SCOPES,
        ),
    )
    return token_path
