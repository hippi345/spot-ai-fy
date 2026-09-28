"""Persist per-reply Spotify tool traces under DATA_DIR (no secrets)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from spot_backend.llm_secret_safety import redact_known_api_keys

_TRACE_FILENAME = "chat_tool_traces.jsonl"

_SENSITIVE_ARG_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "access_token",
        "refresh_token",
        "token",
        "secret",
    }
)


def tool_trace_log_path(data_dir: Path) -> Path:
    return data_dir / _TRACE_FILENAME


def summarize_tool_args(arguments: dict[str, Any] | None, *, max_len: int = 160) -> str:
    if not isinstance(arguments, dict) or not arguments:
        return "{}"
    safe: dict[str, Any] = {}
    for key, val in arguments.items():
        k = str(key).lower()
        if k in _SENSITIVE_ARG_KEYS:
            safe[key] = "<redacted>"
            continue
        if isinstance(val, str) and len(val) > 80:
            safe[key] = val[:77] + "…"
        else:
            safe[key] = val
    text = json.dumps(safe, sort_keys=True, default=str)
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def tool_trace_outcome(raw_result: str) -> str:
    try:
        data = json.loads(raw_result)
    except (json.JSONDecodeError, TypeError, ValueError):
        return "error"
    if not isinstance(data, dict):
        return "error"
    if data.get("informational_refusal"):
        return "refused"
    if data.get("error"):
        return "error"
    if data.get("ok") is False:
        return "error"
    return "ok"


def append_tool_trace_record(
    data_dir: Path,
    *,
    conversation_id: str | None,
    tool_name: str,
    args_summary: str,
    outcome: str,
    duration_ms: int | None = None,
    known_secrets: list[str] | None = None,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = tool_trace_log_path(data_dir)
    row: dict[str, Any] = {
        "ts_ms": int(time.time() * 1000),
        "conversation_id": (conversation_id or "").strip() or None,
        "tool": tool_name,
        "args": args_summary,
        "outcome": outcome,
    }
    if duration_ms is not None:
        row["duration_ms"] = duration_ms
    line = json.dumps(row, ensure_ascii=False)
    secrets = [s for s in (known_secrets or []) if s]
    if secrets:
        line = redact_known_api_keys(line, secrets)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
