from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from evals.play_history_fixture import load_fixture_metadata, seed_play_log_db
from spot_backend.config import Settings
from spot_backend.ollama_embed import HashingEmbedder
from spot_backend.play_log import PlayLogStore
from spot_backend.spotify_tools import ollama_tools_for_agent


@pytest.fixture
def play_log_db(tmp_path) -> PlayLogStore:
    s = Settings(data_dir=tmp_path / "data", spot_play_log="on", user_timezone="UTC")
    store = PlayLogStore(settings=s, embedder=HashingEmbedder(dims=64))
    seed_play_log_db(store)
    return store


def test_sunset_this_week_recall_precision(play_log_db: PlayLogStore) -> None:
    meta = load_fixture_metadata()
    ref = datetime.fromisoformat(meta["reference_now"].replace("Z", "+00:00"))
    planted = set(meta["planted"]["sunset_this_week"])
    out = play_log_db.search("sunset", window="this_week", limit=20, now=ref)
    uris = [r["track_uri"] for r in out["results"]]
    hits = [u for u in uris if u in planted]
    recall = len(hits) / max(1, len(planted))
    precision = len(hits) / max(1, len(uris))
    assert recall >= 0.9
    assert precision >= 0.7
    old = set(meta["planted"]["sunset_old"])
    skipped = set(meta["planted"]["sunset_skipped"])
    assert not set(uris) & old
    assert not set(uris) & skipped


def test_vector_beats_keyword_paraphrase(play_log_db: PlayLogStore) -> None:
    ref = datetime.fromisoformat(load_fixture_metadata()["reference_now"].replace("Z", "+00:00"))
    sem = play_log_db.search("golden hour", window="this_week", limit=20, now=ref)
    # Force keyword-only by using store with failing embedder on query path
    store_kw = PlayLogStore(
        settings=play_log_db.settings,
        path=play_log_db.path,
        embedder=type("Boom", (), {"embed_query": lambda *_: (_ for _ in ()).throw(RuntimeError()), "embed_documents": lambda *_: [], "model_name": "x", "dimensions": 64})(),
    )
    kw = store_kw.search("golden hour", window="this_week", limit=20, now=ref)
    sem_uris = {r["track_uri"] for r in sem["results"]}
    kw_uris = {r["track_uri"] for r in kw["results"]}
    meta = load_fixture_metadata()
    planted = set(meta["planted"]["sunset_this_week"])
    assert len(sem_uris & planted) >= len(kw_uris & planted)


def test_pruning_removes_expired_rows(play_log_db: PlayLogStore) -> None:
    meta = load_fixture_metadata()
    ref = datetime.fromisoformat(meta["reference_now"].replace("Z", "+00:00"))
    stats = play_log_db.prune(now=ref)
    assert stats["deleted"] >= len(meta["planted"]["prune_90_row_ids"]) + len(
        meta["planted"]["prune_skip_row_ids"]
    )


def test_capture_writes_request_text(tmp_path) -> None:
    from spot_backend.play_log_capture import bind_turn_context, record_playback_started
    from spot_backend.spotify_tools import SpotifyToolRunner

    s = Settings(data_dir=tmp_path / "d", spot_play_log="on")
    bind_turn_context("play golden hour sunset mix", "conv-1")
    runner = SpotifyToolRunner(settings=s, conversation_id="conv-1")
    record_playback_started(
        runner,
        track_uri="spotify:track:abc123",
        title="Sunset Song",
        artist="A",
        album="B",
    )
    store = PlayLogStore(settings=s, embedder=HashingEmbedder(dims=64))
    with store._ensure() as conn:
        row = conn.execute("SELECT request_text, skipped FROM play_log LIMIT 1").fetchone()
        assert row["request_text"] == "play golden hour sunset mix"
        assert row["skipped"] == 0
    store.mark_latest_open_skipped("spotify:track:abc123")
    with store._ensure() as conn:
        row2 = conn.execute("SELECT skipped FROM play_log LIMIT 1").fetchone()
    assert row2["skipped"] == 1


def test_retrieval_p95_under_100ms(play_log_db: PlayLogStore) -> None:
    samples = []
    for _ in range(40):
        t0 = time.perf_counter()
        play_log_db.search("dusk drive", window="this_week", limit=20)
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    p95 = samples[int(len(samples) * 0.95) - 1]
    assert p95 < 100.0


def test_flag_off_tool_list_unchanged() -> None:
    off = Settings(spot_play_log="off")
    names = {t["function"]["name"] for t in ollama_tools_for_agent(off)}
    assert "search_play_history" not in names
    assert "clear_play_history" not in names
