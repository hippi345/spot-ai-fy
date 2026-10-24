#!/usr/bin/env python3
"""Redact known secret shapes in-place, then gitleaks + eval key scan. Fail on hit."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from evals.key_leak_scan import find_key_leaks
from spot_backend.llm_secret_safety import redact_known_api_keys

_REDACT_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z\-_]{8,}"),
    re.compile(r"key=[A-Za-z0-9_\-]{16,}", re.I),
)


def _redact_file(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    out = redact_known_api_keys(text)
    for pat in _REDACT_PATTERNS:
        out = pat.sub("REDACTED_SECRET", out)
    if out != text:
        path.write_text(out, encoding="utf-8")


def _scan_eval_patterns(paths: list[Path]) -> list[str]:
    hits: list[str] = []
    for p in paths:
        if not p.is_file():
            continue
        hits.extend(find_key_leaks(p.read_text(encoding="utf-8", errors="replace")))
    return hits


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: ci_scan_artifacts_for_secrets.py <file-or-dir> [...]", file=sys.stderr)
        return 2
    targets: list[Path] = []
    for raw in argv[1:]:
        p = Path(raw)
        if p.is_dir():
            targets.extend([f for f in p.rglob("*") if f.is_file()])
        elif p.is_file():
            targets.append(p)
    if not targets:
        print("No artifact files to scan.")
        return 0
    for p in targets:
        _redact_file(p)
    pattern_hits = _scan_eval_patterns(targets)
    if pattern_hits:
        print("eval key_leak_scan hits (prefixes only):", file=sys.stderr)
        for h in pattern_hits[:20]:
            m = re.search(r"AIza[A-Za-z0-9_-]{0,4}", h)
            prefix = m.group(0)[:4] if m else "key?"
            print(f"  - pattern match prefix={prefix!r}", file=sys.stderr)
        return 1
    for p in targets:
        try:
            proc = subprocess.run(
                ["gitleaks", "detect", "--no-git", "--redact", "-s", str(p)],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            print("gitleaks not installed; skipping gitleaks per-file scan", file=sys.stderr)
            return 0
        if proc.returncode != 0:
            print(f"gitleaks failed for {p}", file=sys.stderr)
            print((proc.stdout or proc.stderr)[:2000], file=sys.stderr)
            return 1
    print(f"Artifact scan clean ({len(targets)} file(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
