#!/usr/bin/env python3
"""Benefit gate for search_my_library (run locally; not executed in CI).

Gate (Joel's machine, real Ollama + Spotify):
- Hit@1: index_on must beat index_off by >= +25 percentage points
- Task success: no loss vs index_off
- p50 latency: index_on <= index_off + 1.0s
- Qualitative rubrics: not below PR1 baseline (0.05 tolerance)
- pytest: zero regressions

Fill accepted answers in backend/evals/prompts.yaml before running.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[1] / "evals" / "prompts.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Library index A/B benefit gate (manual)")
    parser.add_argument("--dry-run", action="store_true", help="Validate prompts file only")
    args = parser.parse_args()
    if not PROMPTS.is_file():
        print(f"Missing {PROMPTS}", file=sys.stderr)
        return 2
    import yaml

    data = yaml.safe_load(PROMPTS.read_text(encoding="utf-8"))
    prompts = data.get("prompts") or []
    missing = [p.get("id") for p in prompts if not (p.get("accepted_answers"))]
    print(json.dumps({"prompt_count": len(prompts), "missing_accepted_answers": missing}, indent=2))
    if args.dry_run:
        return 0
    print(
        "Full gate requires local Ollama, Spotify tokens, and tagged prompts. "
        "Run index_off vs index_on arms externally; this script is a checklist stub.",
        file=sys.stderr,
    )
    return 0 if not missing else 3


if __name__ == "__main__":
    raise SystemExit(main())
