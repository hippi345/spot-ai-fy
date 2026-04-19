"""Can we read a followed playlist's tracks via /playlists/{id} with fields?"""

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

    pls = httpx.get(f"{API}/me/playlists", params={"limit": 50}, headers=headers, timeout=30.0).json()
    followed = [p for p in pls.get("items", []) if ((p.get("owner") or {}).get("id")) != me_id]
    owned = [p for p in pls.get("items", []) if ((p.get("owner") or {}).get("id")) == me_id]

    def probe(pid: str, label: str):
        print(f"\n{label}: id={pid}")
        for variants in [
            {},
            {"fields": "tracks(total,items(track(uri,name)))"},
            {"fields": "tracks.total"},
            {"market": "from_token"},
            {"market": "US"},
        ]:
            r = httpx.get(f"{API}/playlists/{pid}", params=variants, headers=headers, timeout=30.0)
            if r.status_code == 200:
                data = r.json()
                tr = (data.get("tracks") or {})
                total = tr.get("total")
                items = tr.get("items") or []
                first_uri = None
                if items and isinstance(items[0], dict):
                    first_uri = ((items[0].get("track") or {}).get("uri"))
                print(f"  params={variants} -> 200  total={total}  items={len(items)}  first_uri={first_uri}")
            else:
                print(f"  params={variants} -> {r.status_code}  {r.text[:120]}")

    if followed:
        probe(followed[0].get("id"), f"FOLLOWED ({followed[0].get('name')!r}, owner={((followed[0].get('owner') or {}).get('id'))})")
    if owned:
        probe(owned[0].get("id"), f"OWNED ({owned[0].get('name')!r})")

    # Search-for-playlists works in dev mode. Can we play its context_uri?
    print("\nspotify_search_playlists cross-check (search for lofi):")
    r = httpx.get(f"{API}/search", params={"q": "lofi study", "type": "playlist", "limit": 3},
                  headers=headers, timeout=30.0)
    if r.status_code == 200:
        items = (((r.json() or {}).get("playlists") or {}).get("items") or [])
        for p in items[:3]:
            print(f"  id={p.get('id')}  name={p.get('name')!r}  owner={((p.get('owner') or {}).get('id'))}  uri={p.get('uri')}")
    else:
        print(f"  search -> {r.status_code}: {r.text[:200]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
