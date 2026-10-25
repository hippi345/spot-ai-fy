"""Seed / reset local SAMPLE library index from committed eval fixture (no Spotify account)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from evals.eval_library_fixture import FIXTURE_PATH, PRIMARY_USER, load_fixture, tracks
from evals.fixture_embedder import CommittedFixtureEmbedder
from evals.fixture_index_text import library_index_document_text
from spot_backend.config import Settings
from spot_backend.library_index import LibraryIndex, index_path
from spot_backend.spotify_client import DEFAULT_SCOPES
from spot_backend.token_store import TokenBundle, save_tokens

SAMPLE_MARKER = "sample_library_catalog.json"
SAMPLE_LABEL = "SAMPLE_DATA — fictional catalog for local dev and evals only; not your Spotify library."


def sample_marker_path(data_dir: Path) -> Path:
    return data_dir / SAMPLE_MARKER


def _fixture_docs(user_id: str = PRIMARY_USER) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for t in tracks(user_id):
        text = library_index_document_text(t, added_at=t.get("liked_at"))
        play = int(t.get("eval_play_count") or t.get("play_count") or 1)
        boost = 1.0 + math.log1p(max(0, play)) * 0.06
        tid = t["id"]
        docs.append(
            {
                "id": f"track:{tid}",
                "kind": "track",
                "uri": t["uri"],
                "text": text,
                "boost": boost,
            }
        )
    return docs


def seed_sample_library(
    data_dir: Path | None = None,
    *,
    write_eval_tokens: bool = True,
) -> dict[str, Any]:
    """Build library_index.sqlite from committed SAMPLE fixture; no network."""
    settings = Settings(data_dir=data_dir) if data_dir else Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    embedder = CommittedFixtureEmbedder()
    idx = LibraryIndex(settings=settings, embedder=embedder)
    docs = _fixture_docs()
    idx.sync_from_documents(docs)
    meta = load_fixture().get("meta") or {}
    planted = (meta.get("planted_sets") or {}).get("sunset_loved") or []
    marker = {
        "label": SAMPLE_LABEL,
        "sample": True,
        "fixture_path": str(FIXTURE_PATH),
        "track_count": len(docs),
        "planted_sunset_loved_count": len(planted),
        "primary_user_id": PRIMARY_USER,
        "seeded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_model": embedder.model_name,
    }
    sample_marker_path(settings.data_dir).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    if write_eval_tokens:
        tok_path = settings.resolved_token_path
        if not tok_path.is_file():
            save_tokens(
                tok_path,
                TokenBundle(
                    access_token="sample-library-access-token",
                    refresh_token="sample-library-refresh-token",
                    expires_at=time.time() + 86400 * 30,
                    scope=DEFAULT_SCOPES,
                ),
            )
    status = idx.status()
    return {
        "ok": True,
        "data_dir": str(settings.data_dir),
        "index_path": str(index_path(settings)),
        "items": status.get("items"),
        "mode": status.get("mode"),
        "marker": str(sample_marker_path(settings.data_dir)),
    }


def reset_sample_library(data_dir: Path | None = None, *, remove_tokens: bool = False) -> dict[str, Any]:
    settings = Settings(data_dir=data_dir) if data_dir else Settings()
    removed: list[str] = []
    idx_path = index_path(settings)
    if idx_path.is_file():
        idx_path.unlink()
        removed.append(str(idx_path))
    marker = sample_marker_path(settings.data_dir)
    if marker.is_file():
        marker.unlink()
        removed.append(str(marker))
    if remove_tokens:
        tok = settings.resolved_token_path
        if tok.is_file() and tok.read_text(encoding="utf-8").find("sample-library-access-token") >= 0:
            tok.unlink()
            removed.append(str(tok))
    return {"ok": True, "data_dir": str(settings.data_dir), "removed": removed}
