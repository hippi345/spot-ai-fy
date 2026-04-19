"""Figure out which playlist-creation endpoint actually works in dev mode."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
BACKEND_DIR = HERE.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import httpx

from spot_backend.config import get_settings
from spot_backend.spotify_client import SpotifyClient

API = "https://api.spotify.com/v1"


def cleanup(headers: dict, pid: str | None):
    if not pid:
        return
    r = httpx.delete(f"{API}/playlists/{pid}/followers", headers=headers, timeout=30.0)
    print(f"    cleanup: DELETE /playlists/{pid}/followers -> {r.status_code}")


def main() -> int:
    settings = get_settings()
    client = SpotifyClient(settings=settings)
    if not client.load_bundle():
        return 2
    token = client.ensure_fresh_access_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    me = httpx.get(f"{API}/me", headers=headers, timeout=30.0).json()
    me_id = me.get("id")
    print(f"signed in as: {me_id}\n")

    for label, path in [
        ("POST /me/playlists", f"{API}/me/playlists"),
        (f"POST /users/{me_id}/playlists", f"{API}/users/{me_id}/playlists"),
    ]:
        body = {"name": f"spot-ai-fy diag {label[:10]}", "public": False}
        r = httpx.post(path, headers=headers, json=body, timeout=30.0)
        print(f"{label:45s} -> {r.status_code}")
        if r.status_code in (200, 201):
            try:
                new_id = r.json().get("id")
            except Exception:
                new_id = None
            print(f"   new_id={new_id}")
            cleanup(headers, new_id)
        else:
            print(f"   body: {r.text[:200]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
