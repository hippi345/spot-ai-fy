"""SAMPLE library seed / reset (no Spotify)."""

from __future__ import annotations

from pathlib import Path

from evals.sample_library_seed import reset_sample_library, sample_marker_path, seed_sample_library
from spot_backend.library_index import index_path


def test_seed_and_reset_sample_library(tmp_path: Path) -> None:
    data_dir = tmp_path / "sample-data"
    out = seed_sample_library(data_dir, write_eval_tokens=True)
    assert out["ok"]
    idx = index_path(__import__("spot_backend.config", fromlist=["Settings"]).Settings(data_dir=data_dir))
    assert idx.is_file()
    assert sample_marker_path(data_dir).is_file()
    assert "SAMPLE" in sample_marker_path(data_dir).read_text(encoding="utf-8")
    reset = reset_sample_library(data_dir)
    assert reset["ok"]
    assert not idx.is_file()
    assert not sample_marker_path(data_dir).is_file()
