"""Invoke SpotifyToolRunner.run('spotify_duplicate_playlist', ...) for RNB2025
exactly the way the agent would. If this succeeds but the UI still reports
'Spotify API issue', the backend is running stale code (not restarted).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
BACKEND_DIR = HERE.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from spot_backend.spotify_tools import SpotifyToolRunner  # noqa: E402


def main() -> int:
    runner = SpotifyToolRunner()
    try:
        # Step 1: find RNB2025 id directly via the raw client (avoid the shrink/compact path
        # that can emit non-strict JSON when the payload is long).
        print("Looking up RNB2025 in your playlists...")
        raw_pls = runner.client.api_get("/me/playlists", params={"limit": 50})
        items = raw_pls.get("items") if isinstance(raw_pls, dict) else None
        if not isinstance(items, list):
            print(f"Unexpected /me/playlists payload type: {type(raw_pls)}")
            return 2
        data = raw_pls  # keep variable name for downstream code
        rnb = next((p for p in items if isinstance(p, dict) and p.get("name") == "RNB2025"), None)
        if not rnb:
            print("No playlist named 'RNB2025' found in your first 50. Using first owned instead.")
            me_data = runner.client.api_get("/me")
            me_id = me_data.get("id") if isinstance(me_data, dict) else None
            rnb = next((p for p in items if isinstance(p, dict)
                        and ((p.get("owner") or {}).get("id")) == me_id), None)
            if not rnb:
                print("No owned playlist either — abort.")
                return 3

        source_id = rnb.get("id")
        source_name = rnb.get("name")
        owner = ((rnb.get("owner") or {}).get("id"))
        print(f"Source: name={source_name!r}  id={source_id}  owner={owner}")

        # Step 2: call the composite tool with the exact name the user typed.
        args = {"source_playlist_id": source_id, "name": f"{source_name} Copy (diag)"}
        print(f"\nInvoking spotify_duplicate_playlist with args: {args}")
        result_raw = runner.run("spotify_duplicate_playlist", args)
        result = json.loads(result_raw) if isinstance(result_raw, str) else result_raw
        print("\n--- Tool response ---")
        print(json.dumps(result, indent=2)[:4000])
        print("---------------------\n")

        if isinstance(result, dict) and result.get("ok"):
            new_id = result.get("new_playlist_id")
            copied = result.get("tracks_copied")
            print(f"OK  new_playlist_id={new_id}  tracks_copied={copied}")
            print(f"Delete it on Spotify if you don't want it. id={new_id}")
        else:
            print("NOT OK  — see tool response above.")
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
