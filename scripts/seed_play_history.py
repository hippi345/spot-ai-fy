#!/usr/bin/env python3
"""Load SAMPLE play history into DATA_DIR (deterministic fixture)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from spot_backend.config import get_settings
from spot_backend.play_log import PlayLogStore
from evals.play_history_fixture import seed_play_log_db


def main() -> int:
    settings = get_settings()
    settings.spot_play_log = "on"
    store = PlayLogStore(settings=settings)
    store.clear_all()
    planted = seed_play_log_db(store)
    print(f"Seeded play log at {store.path} (sample data).")
    print(f"Planted sunset this week: {len(planted.get('sunset_this_week', []))} tracks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
