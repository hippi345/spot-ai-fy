"""Verify that /playlists/{source}/items works for a FOLLOWED playlist
(so spotify_duplicate_playlist can paginate it) and print current token scopes
so we can see why user-top-read / user-follow-read are still missing.
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
from spot_backend.spotify_client import SpotifyClient, DEFAULT_SCOPES

API = "https://api.spotify.com/v1"


def main() -> int:
    settings = get_settings()
    client = SpotifyClient(settings=settings)
    if not client.load_bundle():
        print("Not signed in.")
        return 2
    token = client.ensure_fresh_access_token()
    headers = {"Authorization": f"Bearer {token}"}

    granted = sorted(client.get_token_scopes() or [])
    expected = sorted(s.strip() for s in DEFAULT_SCOPES.split() if s.strip())
    missing = sorted(set(expected) - set(granted))
    extra = sorted(set(granted) - set(expected))
    print("Token scope diagnostic")
    print("-" * 60)
    print(f"granted  : {granted}")
    print(f"expected : {expected}")
    print(f"MISSING  : {missing}  <-- these need a fresh Sign out -> Connect")
    print(f"extra    : {extra}")
    print()

    # Find a FOLLOWED (non-owned) playlist.
    me = httpx.get(f"{API}/me", headers=headers, timeout=30.0).json()
    me_id = me.get("id")
    pls = httpx.get(f"{API}/me/playlists", params={"limit": 50}, headers=headers, timeout=30.0).json()
    followed = None
    for p in pls.get("items", []):
        if ((p.get("owner") or {}).get("id")) != me_id:
            followed = p
            break
    if not followed:
        print("No followed (non-owned) playlists in your library. Follow one and re-run.")
        return 0

    fid = followed.get("id")
    fname = followed.get("name")
    fowner = ((followed.get("owner") or {}).get("id"))
    print(f"Testing followed playlist: {fname!r} (id={fid}, owner={fowner})")

    for path in (f"/playlists/{fid}/items", f"/playlists/{fid}/tracks"):
        r = httpx.get(f"{API}{path}", params={"limit": 5, "fields": "items(track(uri))"},
                      headers=headers, timeout=30.0)
        ok = r.status_code == 200
        n = 0
        if ok:
            try:
                n = len((r.json() or {}).get("items") or [])
            except Exception:
                pass
        print(f"  GET {path:40s}  -> {r.status_code}  items_returned={n}  {r.text[:160] if not ok else ''}")

    # Test playlist creation (the second step of duplicate) without actually creating.
    # We'll do a dry-run-ish check by POSTing then DELETEing immediately if it works,
    # but only if user passes --live.
    if "--live" in sys.argv:
        print("\nCreating a throwaway playlist...")
        r = httpx.post(f"{API}/users/{me_id}/playlists", headers=headers,
                       json={"name": "spot-ai-fy diag (delete me)", "public": False},
                       timeout=30.0)
        print(f"  POST /users/{me_id}/playlists -> {r.status_code}")
        if r.status_code in (200, 201):
            new_id = r.json().get("id")
            print(f"  new_id={new_id}. Unfollowing (deleting)...")
            r2 = httpx.delete(f"{API}/playlists/{new_id}/followers", headers=headers, timeout=30.0)
            print(f"  DELETE /playlists/{new_id}/followers -> {r2.status_code}")
        else:
            print(f"  body: {r.text[:240]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
