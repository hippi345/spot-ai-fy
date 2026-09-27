"""Opt-in smoke test: docs/SMOKE_TEST.md steps 1–7 via real Ollama tool calling."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from spot_backend.app import app
from spot_backend.now_playing import reset_now_playing_cache_for_tests
from spot_backend.spotify_tools import SpotifyToolRunner
from tests.ollama_smoke_helpers import (
    PromptSmokeResult,
    ToolCallRecord,
    build_reply_validators,
    build_state_validators,
    evaluate_expected_tools,
    ollama_model_tag_from_env,
    ollama_smoke_strict,
    probe_ollama_model,
    results_to_markdown_table,
    write_report,
)
from tests.smoke_spotify_state import SpotifyPlaybackState, install_stateful_spotify_mock

pytestmark = pytest.mark.ollama_smoke

SMOKE_PROMPTS: list[tuple[int, str]] = [
    (1, "Play John Mayer"),
    (2, "What's playing?"),
    (3, "Skip"),
    (4, "Pause"),
    (5, "Resume"),
    (6, "What are my playlists?"),
    (7, "Queue Gravity by John Mayer"),
]


def _collect_sse_events(stream_resp) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in stream_resp.text.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    events.append(json.loads(payload))
    return events


def _run_ollama_smoke_case(
    client: TestClient,
    tool_log: list[ToolCallRecord],
    step: int,
    message: str,
    state_holder: dict[str, Any],
) -> PromptSmokeResult:
    reply_validators = build_reply_validators(state_holder)
    state_validators = build_state_validators(state_holder)
    t0 = time.perf_counter()
    tool_log.clear()
    result = PromptSmokeResult(step=step, prompt=message, latency_s=0.0)
    try:
        resp = client.post("/api/chat/stream", json={"message": message})
        if resp.status_code != 200:
            result.error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            result.latency_s = time.perf_counter() - t0
            return result
        events = _collect_sse_events(resp)
        err = next((e for e in events if e.get("type") == "error"), None)
        if err:
            result.error = str(err.get("message") or err)
        final = next((e for e in events if e.get("type") == "final"), None)
        reply = str(final.get("text") or "") if final else ""
        result.reply_excerpt = reply[:240]
        result.tools = list(tool_log)
        ok_tools, tool_note = evaluate_expected_tools(step, result.tools)
        result.expected_tools_ok = ok_tools
        result.expected_tools_note = tool_note
        np_resp = client.get("/api/now-playing")
        np_payload = np_resp.json() if np_resp.status_code == 200 else {}
        if step in reply_validators:
            ok_reply, reply_note = reply_validators[step](reply, np_payload)
            result.reply_sensible = ok_reply
            result.reply_note = reply_note
        else:
            result.reply_sensible = bool(reply.strip()) and result.error is None
            if not result.reply_sensible:
                result.reply_note = "Empty reply or stream error"
        if step in state_validators:
            ok_state, state_note = state_validators[step](np_payload)
            result.spotify_state_ok = ok_state
            result.spotify_state_note = state_note
        else:
            result.spotify_state_ok = result.error is None
    except Exception as exc:  # noqa: BLE001 — smoke harness must record failures
        result.error = f"{type(exc).__name__}: {exc}"
    result.latency_s = time.perf_counter() - t0
    return result


@pytest.fixture(autouse=True)
def _clear_np_cache() -> None:
    reset_now_playing_cache_for_tests()


@pytest.fixture
def tool_call_log(monkeypatch: pytest.MonkeyPatch) -> list[ToolCallRecord]:
    log: list[ToolCallRecord] = []
    original = SpotifyToolRunner.run

    def logged_run(self, name: str, arguments: dict[str, Any] | None = None, **kwargs: Any) -> str:
        payload = arguments if isinstance(arguments, dict) else {}
        log.append(ToolCallRecord(name=name, arguments=dict(payload)))
        return original(self, name, payload, **kwargs)

    monkeypatch.setattr(SpotifyToolRunner, "run", logged_run)
    return log


@respx.mock
def test_ollama_smoke_steps_1_through_7(
    data_dir,
    signed_in_tokens,
    monkeypatch: pytest.MonkeyPatch,
    tool_call_log: list[ToolCallRecord],
) -> None:
    if not os.environ.get("RUN_OLLAMA_SMOKE", "").strip():
        pytest.skip("Set RUN_OLLAMA_SMOKE=1 to run the real Ollama smoke test")

    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    host_base = host.rstrip("/")
    respx.route(url__regex=rf"{re.escape(host_base)}/.*").pass_through()

    model = ollama_model_tag_from_env()
    ready, detail = probe_ollama_model(host, model)
    if not ready:
        pytest.skip(f"Ollama not ready for smoke test: {detail}")

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", host)
    monkeypatch.setenv("OLLAMA_MODEL", model)
    monkeypatch.setenv("SPOT_AI_FY_DISABLE_DETERMINISTIC_CHAT", "1")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "30m")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    monkeypatch.setenv("OLLAMA_HISTORY_MESSAGES", "10")

    state = SpotifyPlaybackState()
    install_stateful_spotify_mock(state)
    client = TestClient(app)
    state_holder: dict[str, Any] = {}
    results: list[PromptSmokeResult] = []
    total_t0 = time.perf_counter()

    for step, message in SMOKE_PROMPTS:
        row = _run_ollama_smoke_case(client, tool_call_log, step, message, state_holder)
        results.append(row)
        print(f"\n--- Step {step}: {message!r} ({row.latency_s:.1f}s) ---")
        print(results_to_markdown_table([row], model=model))

    total_s = time.perf_counter() - total_t0
    table = results_to_markdown_table(results, model=model)
    print("\n" + table)

    report_path = os.environ.get("OLLAMA_SMOKE_REPORT_PATH", "").strip()
    if report_path:
        write_report(report_path, results, model=model, total_s=total_s)

    failures = [r for r in results if not r.passed]
    if failures and ollama_smoke_strict():
        lines = [f"{r.step}: {r.prompt} — {r.error or r.expected_tools_note or r.reply_note}" for r in failures]
        pytest.fail("Ollama smoke failures (strict mode):\n" + "\n".join(lines))
