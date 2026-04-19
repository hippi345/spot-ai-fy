"""Test /users/{user_id}/playlists against several known user ids."""

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


def main() -> int:
    settings = get_settings()
    client = SpotifyClient(settings=settings)
    if not client.load_bundle():
        return 2
    token = client.ensure_fresh_access_token()
    headers = {"Authorization": f"Bearer {token}"}

    me = httpx.get(f"{API}/me", headers=headers, timeout=30.0).json()
    me_id = me.get("id")
    print(f"signed in as: {me_id}\n")

    user_ids = [
        me_id,                   # you — should always work
        "spotify",               # official Spotify editorial account
        "glennpmcdonald",        # Spotify data alchemist (public profile)
        "bbc",                   # BBC
        "22xkphhjzbf2pjozluqvfvjbi",  # user tried earlier
        "not_a_real_user_xyz_zzz",    # definitely fake
    ]
    for uid in user_ids:
        r = httpx.get(f"{API}/users/{uid}/playlists", params={"limit": 5}, headers=headers, timeout=30.0)
        msg = "(200 OK)" if r.status_code == 200 else r.text[:200]
        count = None
        if r.status_code == 200:
            try:
                count = len(((r.json() or {}).get("items") or []))
            except Exception:
                pass
        print(f"  user={uid:45s}  status={r.status_code}  items={count}  {msg if r.status_code != 200 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
