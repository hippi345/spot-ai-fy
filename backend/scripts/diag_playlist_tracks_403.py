"""Bisect the 403 on GET /playlists/{id}/tracks."""

from __future__ import annotations

import json
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
        print("Not signed in.")
        return 2
    token = client.ensure_fresh_access_token()
    headers = {"Authorization": f"Bearer {token}"}

    pls = httpx.get(f"{API}/me/playlists", params={"limit": 5}, headers=headers, timeout=30.0).json()
    # pick first owned
    me = httpx.get(f"{API}/me", headers=headers, timeout=30.0).json()
    me_id = me.get("id")
    owned_pid = None
    for p in pls.get("items", []):
        if ((p.get("owner") or {}).get("id")) == me_id:
            owned_pid = p.get("id")
            print(f"Using OWNED playlist: {p.get('name')!r} id={owned_pid}")
            break

    cases: list[tuple[str, dict]] = [
        ("bare",                        {}),
        ("limit=10",                    {"limit": 10}),
        ("limit=10,market=from_token",  {"limit": 10, "market": "from_token"}),
        ("limit=10,market=US",          {"limit": 10, "market": "US"}),
        ("limit=1",                     {"limit": 1}),
        ("fields=items(track(uri))",    {"limit": 10, "fields": "items(track(uri))"}),
        ("fields=total",                {"fields": "total"}),
        ("fields=href,total",           {"fields": "href,total"}),
    ]

    print(f"\nGET /playlists/{owned_pid}/tracks with variants:\n")
    for label, params in cases:
        r = httpx.get(f"{API}/playlists/{owned_pid}/tracks", params=params, headers=headers, timeout=30.0)
        print(f"  {label:45s} -> {r.status_code}  {r.text[:160] if r.status_code != 200 else '(200 OK)'}")

    print(f"\nGET /playlists/{owned_pid}/items  (Feb-2026 renamed endpoint) with variants:\n")
    for label, params in cases:
        r = httpx.get(f"{API}/playlists/{owned_pid}/items", params=params, headers=headers, timeout=30.0)
        print(f"  {label:45s} -> {r.status_code}  {r.text[:160] if r.status_code != 200 else '(200 OK)'}")

    print(f"\nGET /playlists/{owned_pid}  (metadata, w/ tracks embedded):")
    r = httpx.get(f"{API}/playlists/{owned_pid}", headers=headers, timeout=30.0, params={"fields": "name,tracks(total,items(track(uri)))"})
    print(f"  -> {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        tracks = (data.get("tracks") or {})
        total = tracks.get("total")
        items = tracks.get("items") or []
        print(f"  name={data.get('name')!r}  total={total}  items_returned={len(items)}")
        if items:
            print(f"  first track uri: {(items[0].get('track') or {}).get('uri')}")
    else:
        print(f"  body: {r.text[:200]}")

    # Introspect token scopes by calling /me and comparing with what Spotify thinks is granted
    print("\nCurrent granted scopes per client bundle:")
    print(f"  {sorted(client.get_token_scopes() or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
