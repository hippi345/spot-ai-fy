#!/usr/bin/env python3
"""Capture a short API-level e2e session log for PR handoff (mocked Spotify)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import respx
from fastapi.testclient import TestClient

from spot_backend.app import app
from spot_backend.config import Settings
from spot_backend.token_store import TokenBundle, save_tokens
from spot_backend.spotify_client import DEFAULT_SCOPES
from evals.mock_spotify import install_profile

OUT = Path(__file__).resolve().parents[1] / "e2e_session.txt"


def main() -> None:
    import os

    lines: list[str] = []
    lines.append(f"Spot-AI-fy e2e session (mocked Spotify) — {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append("")

    data_dir = Path("/tmp/spot-e2e-session")
    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["SPOTIFY_CLIENT_ID"] = "a" * 32
    data_dir.mkdir(parents=True, exist_ok=True)
    save_tokens(
        data_dir / "tokens.json",
        TokenBundle(
            access_token="e2e-access",
            refresh_token="e2e-refresh",
            expires_at=time.time() + 3600,
            scope=DEFAULT_SCOPES,
        ),
    )

    with respx.mock:
        install_profile("default")
        from evals.mock_spotify import TRACK_A

        respx.get(url__regex=rf"https://api\.spotify\.com/v1/tracks/{TRACK_A}").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": TRACK_A,
                    "uri": f"spotify:track:{TRACK_A}",
                    "name": "Down",
                    "artists": [{"name": "Jay Sean"}],
                    "album": {"id": "alb1" + "x" * 18, "name": "All or Nothing"},
                },
            )
        )
        client = TestClient(app)
        flows = [
            ("GET /api/session", lambda: client.get("/api/session")),
            (
                "POST /api/chat — play Down by Jay Sean (mocked Gemini tool loop)",
                lambda: _chat_gemini_mock(client, "play Down by Jay Sean", tool="spotify_play_track"),
            ),
            (
                "POST /api/chat — add to playlist flow",
                lambda: _chat_gemini_mock(
                    client,
                    "add Kill Bill to RNB2025",
                    tool="spotify_add_tracks_by_query",
                ),
            ),
            (
                "POST /api/chat — pause (deterministic shortcut path when applicable)",
                lambda: client.post("/api/chat", json={"message": "skip"}),
            ),
        ]
        for title, fn in flows:
            lines.append(f"## {title}")
            try:
                r = fn()
                body = r.json() if "application/json" in (r.headers.get("content-type") or "") else r.text
                lines.append(f"status={r.status_code}")
                lines.append(json.dumps(body, indent=2)[:4000])
            except Exception as e:
                lines.append(f"ERROR: {e}")
            lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


def _chat_gemini_mock(client: TestClient, message: str, tool: str) -> httpx.Response:
    import os
    from unittest.mock import patch

    from tests.recheck_helpers import (
        gemini_candidates_payload,
        gemini_stop_candidate,
        make_gemini_post_recorder,
    )

    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["GEMINI_API_KEY"] = "e2e-test-key"

    def handler(_body: dict, n: int, req: httpx.Request) -> httpx.Response:
        if n == 1:
            args = {}
            if tool == "spotify_play_track":
                args = {"track_name": "Down", "artist_name": "Jay Sean"}
            part = {"functionCall": {"name": tool, "args": args}}
        else:
            if tool == "spotify_play_track":
                part = {"text": "Playing Down by Jay Sean."}
            else:
                part = {"text": "Done."}
        return httpx.Response(200, json=gemini_candidates_payload(gemini_stop_candidate(part)), request=req)

    _, fake_post = make_gemini_post_recorder(handler)
    with patch("httpx.Client.post", fake_post):
        return client.post("/api/chat", json={"message": message})


if __name__ == "__main__":
    main()
