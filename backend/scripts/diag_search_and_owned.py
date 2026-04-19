"""Probe playlist search AND owned-playlist reads."""

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

# PowerShell cp1252 can't encode emoji; make stdout utf-8 tolerant.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    settings = get_settings()
    client = SpotifyClient(settings=settings)
    if not client.load_bundle():
        print("Not signed in.")
        return 2
    token = client.ensure_fresh_access_token()
    headers = {"Authorization": f"Bearer {token}"}

    me = httpx.get(f"{API}/me", headers=headers, timeout=30.0).json()
    me_id = me.get("id")

    print("=" * 70)
    print("PLAYLIST SEARCH")
    print("=" * 70)
    for q in ["lofi study", "lo-fi study", "chill beats", "workout", "jazz"]:
        r = httpx.get(f"{API}/search", params={"q": q, "type": "playlist", "limit": 5},
                      headers=headers, timeout=30.0)
        if r.status_code != 200:
            print(f"  q={q!r:20s} -> {r.status_code}  {r.text[:160]}")
            continue
        block = ((r.json() or {}).get("playlists") or {})
        raw = block.get("items") or []
        total = block.get("total")
        non_null = [x for x in raw if x is not None]
        dicts = [x for x in non_null if isinstance(x, dict)]
        print(f"  q={q!r:20s} -> 200  total={total}  raw_len={len(raw)}  non_null={len(non_null)}  dicts={len(dicts)}")
        for item in dicts[:2]:
            print(f"     id={item.get('id')}  name={item.get('name')!r}  owner={((item.get('owner') or {}).get('id'))}")

    print("\n" + "=" * 70)
    print("OWNED PLAYLIST READ (first owned from /me/playlists)")
    print("=" * 70)
    pls = httpx.get(f"{API}/me/playlists", params={"limit": 50}, headers=headers, timeout=30.0).json()
    owned = [p for p in pls.get("items", []) if ((p.get("owner") or {}).get("id")) == me_id]
    rnb = [p for p in owned if p.get("name") == "RNB2025"]
    to_test = (rnb[:1] + owned[:2])[:3]
    # de-dup
    seen_ids = set()
    dedup = []
    for p in to_test:
        if p.get("id") in seen_ids:
            continue
        dedup.append(p)
        seen_ids.add(p.get("id"))

    for p in dedup:
        pid = p.get("id")
        print(f"\nOwned playlist: {p.get('name')!r}  id={pid}")
        r_meta = httpx.get(f"{API}/playlists/{pid}", headers=headers, timeout=30.0)
        print(f"  GET /playlists/{pid}                   -> {r_meta.status_code}  {r_meta.text[:160] if r_meta.status_code != 200 else '(200 OK)'}")
        r_items = httpx.get(f"{API}/playlists/{pid}/items", params={"limit": 3, "fields": "items(track(uri)),next,total"}, headers=headers, timeout=30.0)
        ok = r_items.status_code == 200
        n = 0
        if ok:
            try:
                n = len((r_items.json() or {}).get("items") or [])
            except Exception:
                pass
        print(f"  GET /playlists/{pid}/items (limit=3)   -> {r_items.status_code}  items={n}  {r_items.text[:160] if not ok else ''}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
