#!/usr/bin/env python3
"""Write GitHub Actions job summary from OLLAMA_SMOKE_REPORT_PATH JSON."""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    path = os.environ.get("OLLAMA_SMOKE_REPORT_PATH", "").strip()
    if not path or not os.path.isfile(path):
        print(f"No report at {path!r}", file=sys.stderr)
        return 1
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
    md = data.get("markdown_table") or ""
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as out:
            out.write(md)
            out.write("\n")
    else:
        print(md)
    failed = [r for r in data.get("results") or [] if not r.get("passed")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
