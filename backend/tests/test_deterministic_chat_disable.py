"""Tests for deterministic chat shortcut gating."""

from __future__ import annotations

import pytest

from spot_backend.deterministic_chat import deterministic_chat_shortcuts_disabled
from spot_backend.deterministic_chat import resolve_deterministic_chat_outcome
from spot_backend.spotify_tools import SpotifyToolRunner
from spot_backend.config import Settings


def test_deterministic_shortcuts_disabled_by_env(
    monkeypatch: pytest.MonkeyPatch, data_dir,
) -> None:
    monkeypatch.setenv("SPOT_AI_FY_DISABLE_DETERMINISTIC_CHAT", "1")
    assert deterministic_chat_shortcuts_disabled() is True
    settings = Settings(data_dir=data_dir)
    runner = SpotifyToolRunner(settings=settings)
    try:
        assert (
            resolve_deterministic_chat_outcome(
                "Play John Mayer", runner, conversation_id=None
            )
            is None
        )
    finally:
        runner.close()
