"""E2E: SAMPLE library index → search_my_library → agent pick from planted set (mock LLM + Spotify)."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
import respx

from evals.eval_library_fixture import planted_sunset_uris
from evals.fixture_embedder import CommittedFixtureEmbedder
from evals.mock_spotify import install_profile
from evals.task import run_live_case
from evals.types import EvalCaseInput
from spot_backend.config import Settings
from spot_backend.library_index import LibraryIndex, search_my_library_tool


def _fixture_docs() -> list[dict]:
    import math

    from evals.eval_library_fixture import tracks
    from evals.fixture_index_text import library_index_document_text

    docs: list[dict] = []
    for t in tracks():
        text = library_index_document_text(t, added_at=t.get("liked_at"))
        play = int(t.get("eval_play_count") or t.get("play_count") or 1)
        boost = 1.0 + math.log1p(max(0, play)) * 0.06
        docs.append(
            {
                "id": f"track:{t['id']}",
                "kind": "track",
                "uri": t["uri"],
                "text": text,
                "boost": boost,
            }
        )
    return docs


@pytest.fixture
def sunset_env(data_dir, signed_in_tokens, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SPOT_LIBRARY_INDEX", "on")
    monkeypatch.setattr(
        "spot_backend.library_index.schedule_incremental_sync",
        lambda *a, **k: None,
    )
    settings = Settings(data_dir=data_dir, spot_library_index="on", llm_provider="gemini")
    idx = LibraryIndex(settings=settings, embedder=CommittedFixtureEmbedder())
    idx.sync_from_documents(_fixture_docs())
    return settings


@respx.mock
def test_search_my_library_returns_planted_candidates(sunset_env: Settings) -> None:
    from evals.mock_spotify import allow_eval_llm_passthrough

    allow_eval_llm_passthrough()
    out = search_my_library_tool("sunset vibes I've loved", settings=sunset_env, limit=10)
    assert out.get("available") is True
    results = out.get("results") or []
    assert results
    planted = planted_sunset_uris()
    assert any(r.get("uri") in planted for r in results)


@respx.mock
def test_agent_pick_from_planted_set_mock_llm(sunset_env: Settings) -> None:
    from evals.mock_spotify import allow_eval_llm_passthrough

    allow_eval_llm_passthrough()
    planted = sorted(planted_sunset_uris())
    pick_uri = planted[0]
    pick_name = "Sunset Memory 1"

    tool_json = json.dumps(
        {
            "available": True,
            "mode": "semantic",
            "results": [{"kind": "track", "name": pick_name, "artists": "Sample Artist", "uri": pick_uri, "score": 0.9}],
        }
    )

    events = [
        {"type": "llm_delta", "text": "Searching your library."},
        {"type": "tool_done", "name": "search_my_library", "preview": tool_json},
        {"type": "tool_done", "name": "spotify_play_track", "preview": json.dumps({"uri": pick_uri})},
        {"type": "final", "text": f"Playing {pick_name} from your loved sunset picks."},
    ]

    def fake_iter(*_a, **_k):
        yield from events

    with patch("evals.task.iter_chat_events", fake_iter):
        result = run_live_case(
            EvalCaseInput(
                prompt="play sunset vibes from songs I've loved over the years",
                mode="live",
                mock_spotify_profile="canonical_library",
            ),
            settings=sunset_env,
            provider="gemini",
        )
    assert any(tc.name == "search_my_library" for tc in result.tool_calls)
    assert result.acted_uris
    assert result.acted_uris[0] in planted_sunset_uris()
