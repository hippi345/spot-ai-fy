"""Find the real Spotify dev-mode limit cap for /artists/{id}/albums.

Calls the endpoint with limit values 50, 30, 20, 10, 5, 1 and reports which
ones succeed. Also tries /search and a few other endpoints we care about so
we can correctly cap them in spotify_tools.py.
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

# Taylor Swift's Spotify artist id (well-known, public).
TAYLOR_ID = "06HL4z0CvFAxyc27GXpf02"


def probe(client: SpotifyClient, path: str, params: dict) -> tuple[int, str]:
    token = client.ensure_fresh_access_token()
    base = "https://api.spotify.com/v1"
    r = httpx.get(f"{base}{path}", params=params, headers={"Authorization": f"Bearer {token}"}, timeout=30.0)
    body = r.text[:200]
    return r.status_code, body


def main() -> int:
    settings = get_settings()
    client = SpotifyClient(settings=settings)
    if not client.load_bundle():
        print("Not signed in to Spotify (no token bundle). Sign in via the UI first.")
        return 2

    print("=" * 70)
    print(f"/artists/{TAYLOR_ID}/albums  (Taylor Swift)")
    print("=" * 70)
    for lim in (50, 40, 30, 25, 20, 15, 10, 5, 1):
        code, body = probe(client, f"/artists/{TAYLOR_ID}/albums", {"include_groups": "album,single", "limit": lim, "market": "US"})
        ok = code == 200
        marker = "OK " if ok else "BAD"
        print(f"  limit={lim:2d}  {marker} {code}  {body[:160] if not ok else '(200 OK)'}")

    print("\n" + "=" * 70)
    print("/search?type=artist  (sanity check — known cap = 10)")
    print("=" * 70)
    for lim in (50, 20, 10, 5):
        code, body = probe(client, "/search", {"q": "taylor swift", "type": "artist", "limit": lim})
        marker = "OK " if code == 200 else "BAD"
        print(f"  limit={lim:2d}  {marker} {code}  {body[:160] if code != 200 else '(200 OK)'}")

    print("\n" + "=" * 70)
    print("/me/playlists  (we currently cap at 50)")
    print("=" * 70)
    for lim in (50, 20, 10):
        code, body = probe(client, "/me/playlists", {"limit": lim})
        marker = "OK " if code == 200 else "BAD"
        print(f"  limit={lim:2d}  {marker} {code}  {body[:160] if code != 200 else '(200 OK)'}")

    print("\n" + "=" * 70)
    print(f"/artists/{TAYLOR_ID}/top-tracks  (no limit param, sanity)")
    print("=" * 70)
    code, body = probe(client, f"/artists/{TAYLOR_ID}/top-tracks", {"market": "US"})
    print(f"  {'OK ' if code == 200 else 'BAD'} {code}  {body[:160] if code != 200 else '(200 OK)'}")

    print("\n" + "=" * 70)
    print("/users/spotify/playlists  (well-known Spotify-owned account)")
    print("=" * 70)
    for lim in (50, 20, 10, 5):
        code, body = probe(client, "/users/spotify/playlists", {"limit": lim})
        marker = "OK " if code == 200 else "BAD"
        print(f"  limit={lim:2d}  {marker} {code}  {body[:160] if code != 200 else '(200 OK)'}")

    # /playlists/{id}/tracks — pick one of the user's own playlists at runtime
    print("\n" + "=" * 70)
    print("/playlists/<my first>/tracks  (cap usually 100)")
    print("=" * 70)
    me_pl = None
    try:
        token = client.ensure_fresh_access_token()
        r = httpx.get(
            "https://api.spotify.com/v1/me/playlists",
            params={"limit": 1},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )
        items = (r.json() or {}).get("items") or []
        me_pl = (items[0] or {}).get("id") if items else None
    except Exception as e:
        print(f"  (could not fetch a playlist id: {e})")
    if me_pl:
        for lim in (100, 50, 20, 10):
            code, body = probe(client, f"/playlists/{me_pl}/tracks", {"limit": lim, "fields": "items(track(uri))"})
            marker = "OK " if code == 200 else "BAD"
            print(f"  limit={lim:3d}  {marker} {code}  {body[:160] if code != 200 else '(200 OK)'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
