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
    if data.get("optional_lookup_failure"):
        return "ok"
    if data.get("ok") is True:
        return "ok"
    if data.get("informational_refusal"):
        return "refused"
    if data.get("error"):
        return "error"
    if data.get("ok") is False:
        return "error"
    inner = data.get("playback_result")
    if isinstance(inner, str):
        inner_outcome = tool_trace_outcome(inner)
        if inner_outcome == "error":
            return "error"
    playback = data.get("playback")
    if isinstance(playback, dict) and playback.get("ok") is False:
        return "error"
    return "ok"


def _walk_trace_dicts(data: dict[str, Any]) -> list[dict[str, Any]]:
    stack: list[dict[str, Any]] = [data]
    playback = data.get("playback")
    if isinstance(playback, dict):
        stack.append(playback)
    inner = data.get("playback_result")
    if isinstance(inner, str):
        try:
            parsed = json.loads(inner)
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            stack.append(parsed)
            pb = parsed.get("playback")
            if isinstance(pb, dict):
                stack.append(pb)
    return stack


def tool_trace_spotify_error_body(raw_result: str) -> str | None:
    """Extract redacted Spotify error body from a tool JSON payload (nested or top-level)."""
    try:
        data = json.loads(raw_result)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    for row in _walk_trace_dicts(data):
        body = row.get("spotify_error_body_redacted")
        if isinstance(body, str) and body.strip():
            return body.strip()
    return None


def tool_trace_failure_reason(raw_result: str) -> str | None:
    try:
        data = json.loads(raw_result)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    for row in _walk_trace_dicts(data):
        reason = row.get("failure_reason")
        if isinstance(reason, str) and reason.strip():
            return reason.strip()
    if data.get("playback_verified") is False and data.get("ok") is False:
        return "playback_not_verified"
    if data.get("playlist_not_owned_by_user"):
        return "not_owned"
    err = data.get("error")
    if isinstance(err, str):
        low = err.lower()
        if "uris is required" in low or "required" in low and "uri" in low:
            return "empty_args"
        if "unknown tool" in low:
            return "validation_error"
    status = data.get("spotify_http_status")
    if isinstance(status, int):
        from spot_backend.spotify_tools import _http_failure_reason

        msg = data.get("spotify_api_message") if isinstance(data.get("spotify_api_message"), str) else None
        return _http_failure_reason(status, msg)
    return None


def persist_shortcut_tool_steps(
    data_dir: Path,
    steps: list[tuple[str, dict[str, Any] | None, str]],
    *,
    conversation_id: str | None = None,
    known_secrets: list[str] | None = None,
) -> None:
    """Append trace rows for deterministic shortcut tool chains."""
    for name, args, raw in steps:
        if not isinstance(name, str) or not name.strip():
            continue
        safe_args = args if isinstance(args, dict) else {}
        append_tool_trace_record(
            data_dir,
            conversation_id=conversation_id,
            tool_name=name,
            args_summary=summarize_tool_args(safe_args),
            outcome=tool_trace_outcome(raw),
            known_secrets=known_secrets,
            raw_result=raw,
        )


def append_tool_trace_record(
    data_dir: Path,
    *,
    conversation_id: str | None,
    tool_name: str,
    args_summary: str,
    outcome: str,
    duration_ms: int | None = None,
    known_secrets: list[str] | None = None,
    raw_result: str | None = None,
    spotify_error_body_redacted: str | None = None,
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
    err_body = spotify_error_body_redacted
    if not err_body and raw_result:
        err_body = tool_trace_spotify_error_body(raw_result)
    failure_reason = tool_trace_failure_reason(raw_result or "") if raw_result else None
    if outcome == "error":
        if err_body:
            row["spotify_error_body_redacted"] = err_body
        if failure_reason:
            row["failure_reason"] = failure_reason
        elif not err_body:
            row["failure_reason"] = "unknown_error"
    line = json.dumps(row, ensure_ascii=False)
    secrets = [s for s in (known_secrets or []) if s]
    if secrets:
        line = redact_known_api_keys(line, secrets)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
