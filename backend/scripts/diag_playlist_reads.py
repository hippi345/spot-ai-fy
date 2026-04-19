"""Find out which playlist read endpoints work in dev mode.

Walks /me/playlists and probes /playlists/{id} (metadata) and /playlists/{id}/tracks
for each, classifying as OWNED vs FOLLOWED. Reports counts and a sample of failures.
"""

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
        print("Not signed in. Sign in via the UI first.")
        return 2
    token = client.ensure_fresh_access_token()
    headers = {"Authorization": f"Bearer {token}"}

    me = httpx.get(f"{API}/me", headers=headers, timeout=30.0).json()
    me_id = me.get("id")
    print(f"Signed in as: {me_id}\n")

    # Walk first 50 of /me/playlists.
    all_pls = httpx.get(f"{API}/me/playlists", params={"limit": 50}, headers=headers, timeout=30.0).json()
    items = all_pls.get("items") or []
    print(f"/me/playlists -> {len(items)} items\n")

    by = {"owned_meta": {"ok": 0, "bad": []}, "owned_tracks": {"ok": 0, "bad": []},
          "followed_meta": {"ok": 0, "bad": []}, "followed_tracks": {"ok": 0, "bad": []}}

    for p in items[:20]:  # limit to 20 to keep it quick
        pid = p.get("id")
        owner_id = ((p.get("owner") or {}).get("id"))
        owned = owner_id == me_id
        kind = "owned" if owned else "followed"
        meta_r = httpx.get(f"{API}/playlists/{pid}", headers=headers, timeout=30.0)
        tracks_r = httpx.get(f"{API}/playlists/{pid}/tracks", params={"limit": 10}, headers=headers, timeout=30.0)
        meta_key = f"{kind}_meta"
        tracks_key = f"{kind}_tracks"
        if meta_r.status_code == 200:
            by[meta_key]["ok"] += 1
        else:
            by[meta_key]["bad"].append((pid, meta_r.status_code, meta_r.text[:120]))
        if tracks_r.status_code == 200:
            by[tracks_key]["ok"] += 1
        else:
            by[tracks_key]["bad"].append((pid, tracks_r.status_code, tracks_r.text[:120]))
        print(f"  {kind:8s}  {p.get('name')[:40]:40s}  meta={meta_r.status_code}  tracks={tracks_r.status_code}")

    print("\nSummary:")
    for k, v in by.items():
        print(f"  {k:18s}  ok={v['ok']:3d}  bad={len(v['bad']):3d}")
        if v["bad"][:2]:
            for pid, code, body in v["bad"][:2]:
                print(f"    bad sample: pid={pid} code={code} body={body[:120]}")

    print("\nAlso check: /me/player/recently-played (we don't currently expose it)")
    r = httpx.get(f"{API}/me/player/recently-played", params={"limit": 50}, headers=headers, timeout=30.0)
    print(f"  /me/player/recently-played?limit=50 -> {r.status_code}  {r.text[:160] if r.status_code != 200 else '(200 OK)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
