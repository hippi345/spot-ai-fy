"""Redact secrets from Gemini eval artifacts and console output."""

from __future__ import annotations

import json
from typing import Any, Iterable

from spot_backend.llm_secret_safety import redact_known_api_keys


def redact_text(text: str, secrets: Iterable[str] = ()) -> str:
    return redact_known_api_keys(text or "", secrets)


def sanitize_eval_turn_row(row: dict[str, Any], secrets: Iterable[str] = ()) -> dict[str, Any]:
    out = dict(row)
    if "prompt" in out and isinstance(out["prompt"], str):
        out["prompt"] = redact_text(out["prompt"], secrets)
    if "reply" in out and isinstance(out["reply"], str):
        out["reply"] = redact_text(out["reply"], secrets)
    if "note" in out and isinstance(out["note"], str):
        out["note"] = redact_text(out["note"], secrets)
    if "tools" in out:
        try:
            blob = json.dumps(out["tools"], ensure_ascii=False)
            out["tools"] = json.loads(redact_text(blob, secrets))
        except (TypeError, ValueError, json.JSONDecodeError):
            out["tools"] = []
    return out


def sanitize_eval_report_dict(data: dict[str, Any], secrets: Iterable[str] = ()) -> dict[str, Any]:
    out = dict(data)
    if isinstance(out.get("crash_detail"), str):
        out["crash_detail"] = redact_text(out["crash_detail"], secrets)
    turns = out.get("turns")
    if isinstance(turns, list):
        out["turns"] = [
            sanitize_eval_turn_row(t, secrets) if isinstance(t, dict) else t for t in turns
        ]
    return out


def write_sanitized_eval_artifact(
    report: Any,
    path: Any,
    secrets: Iterable[str] = (),
) -> None:
    """Write gemini eval JSON with secrets redacted (never store raw API keys)."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = report.to_json() if hasattr(report, "to_json") else dict(report)
    safe = sanitize_eval_report_dict(raw, secrets)
    text = json.dumps(safe, indent=2)
    assert_no_secrets_in_text(text, secrets)
    p.write_text(text, encoding="utf-8")


def assert_no_secrets_in_text(text: str, secrets: Iterable[str] = ()) -> None:
    """Raise AssertionError if any secret or API-key pattern appears."""
    from evals.key_leak_scan import assert_no_key_leaks

    assert_no_key_leaks(text, extra_secrets=secrets)
