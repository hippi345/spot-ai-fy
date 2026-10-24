#!/usr/bin/env python3
"""Remove SAMPLE library index + marker. See README: make reset-library."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from evals.sample_library_seed import SAMPLE_LABEL, reset_sample_library


def main() -> int:
    result = reset_sample_library()
    print(SAMPLE_LABEL)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
