"""Run one eval case (mock replay or live provider)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from spot_backend.agent import iter_chat_events
from spot_backend.config import Settings, get_settings
from spot_backend.deterministic_chat import resolve_deterministic_chat_outcome
from spot_backend.spotify_tools import SpotifyToolRunner
from spot_backend.token_store import TokenBundle, save_tokens
from spot_backend.spotify_client import DEFAULT_SCOPES

from evals.mock_spotify import allow_eval_llm_passthrough, install_profile
from evals.types import AgentTurnResult, EvalCaseInput, ToolCallRecord


def _ensure_eval_tokens(settings: Settings) -> None:
    path = settings.resolved_token_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return
    save_tokens(
        path,
        TokenBundle(
            access_token="eval-access-token",
            refresh_token="eval-refresh-token",
            expires_at=time.time() + 3600,
            scope=DEFAULT_SCOPES,
        ),
    )


def _parse_input(raw: Any) -> EvalCaseInput:
    if isinstance(raw, str):
        return EvalCaseInput(prompt=raw, mode="mock")
    if isinstance(raw, dict):
        return EvalCaseInput.model_validate(raw)
    return EvalCaseInput(prompt=str(raw), mode="mock")


def _tool_calls_from_steps(steps: list[tuple[str, dict, str]]) -> list[ToolCallRecord]:
    out: list[ToolCallRecord] = []
    for name, args, raw in steps:
        out.append(ToolCallRecord(name=name, arguments=dict(args), result=raw))
    return out


def _extract_acted_uris(tool_calls: list[ToolCallRecord]) -> list[str]:
    uris: list[str] = []
    for tc in tool_calls:
        args = tc.arguments
        for key in ("uri", "context_uri", "start_at_uri", "playlist_id"):
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                s = val.strip()
                if s.startswith("spotify:"):
                    uris.append(s)
                elif key == "playlist_id" and len(s) == 22:
                    uris.append(f"spotify:playlist:{s}")
        try:
            data = json.loads(tc.result)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(data, dict):
            u = data.get("uri")
            if isinstance(u, str) and u.startswith("spotify:"):
                uris.append(u)
    return uris


def _apply_library_index_arm(arm: str) -> None:
    a = (arm or "").strip().lower()
    if a in ("on", "off"):
        os.environ["SPOT_LIBRARY_INDEX"] = a


def run_mock_case(
    case_input: EvalCaseInput,
    *,
    settings: Settings | None = None,
) -> AgentTurnResult:
    _apply_library_index_arm(case_input.library_index_arm)
    settings = settings or get_settings()
    if case_input.eval_fixture:
        fix = case_input.eval_fixture
        tcs = []
        for row in fix.get("tool_calls") or []:
            if isinstance(row, dict):
                tcs.append(
                    ToolCallRecord(
                        name=str(row.get("name") or ""),
                        arguments=dict(row.get("arguments") or {}),
                        result=str(row.get("result") or ""),
                    )
                )
        return AgentTurnResult(
            reply=str(fix.get("reply") or ""),
            tool_calls=tcs,
            duration_s=float(fix.get("duration_s") or 0.01),
            shortcut=bool(fix.get("shortcut")),
            llm_invoked=bool(fix.get("llm_invoked")),
        )
    _ensure_eval_tokens(settings)
    install_profile(case_input.mock_spotify_profile)
    t0 = time.perf_counter()
    runner = SpotifyToolRunner(settings=settings, conversation_id=case_input.conversation_id)

    if case_input.expect_shortcut or case_input.forbid_llm:
        outcome = resolve_deterministic_chat_outcome(
            case_input.prompt,
            runner,
            conversation_id=case_input.conversation_id,
        )
        runner.close()
        duration = time.perf_counter() - t0
        if outcome is None:
            return AgentTurnResult(
                reply="",
                shortcut=False,
                llm_invoked=False,
                duration_s=duration,
                extra={"error": "expected deterministic shortcut but none matched"},
            )
        steps = outcome.tool_steps
        tool_calls = _tool_calls_from_steps(steps)
        return AgentTurnResult(
            reply=(outcome.reply or "").strip(),
            tool_calls=tool_calls,
            shortcut=True,
            llm_invoked=False,
            duration_s=duration,
            acted_uris=_extract_acted_uris(tool_calls),
        )

    tool_calls: list[ToolCallRecord] = []
    reply = ""
    for step in case_input.mock_script:
        name = str(step.get("name") or "")
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        raw = runner.run(name, dict(args))
        tool_calls.append(ToolCallRecord(name=name, arguments=dict(args), result=raw))
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            data = {}
        if isinstance(data, dict):
            reply = str(
                data.get("user_message")
                or data.get("message")
                or data.get("preview_text")
                or data.get("error")
                or data.get("reconnect_spotify_message")
                or data.get("spotify_api_message")
                or reply
            )
    runner.close()
    duration = time.perf_counter() - t0
    return AgentTurnResult(
        reply=reply.strip(),
        tool_calls=tool_calls,
        shortcut=False,
        llm_invoked=False,
        duration_s=duration,
        acted_uris=_extract_acted_uris(tool_calls),
    )


def run_live_case(
    case_input: EvalCaseInput,
    *,
    settings: Settings | None = None,
    provider: str | None = None,
) -> AgentTurnResult:
    import tempfile

    import respx

    prov = (provider or os.environ.get("LLM_PROVIDER") or "ollama").strip().lower()
    _apply_library_index_arm(case_input.library_index_arm)
    if provider:
        os.environ["LLM_PROVIDER"] = prov
    data_root = Path(tempfile.mkdtemp(prefix="spot-eval-live-"))
    settings = Settings(data_dir=data_root, llm_provider=prov)
    conv_id = f"{case_input.conversation_id}-{os.getpid()}-{time.time_ns()}"
    t0 = time.perf_counter()
    tool_calls: list[ToolCallRecord] = []
    tool_results: list[str] = []
    reply = ""
    saw_llm = False
    with respx.mock:
        _ensure_eval_tokens(settings)
        install_profile(case_input.mock_spotify_profile)
        allow_eval_llm_passthrough()
        arm = (case_input.library_index_arm or "").strip().lower()
        if arm == "on" or (os.environ.get("SPOT_LIBRARY_INDEX") or "").strip().lower() in (
            "on",
            "1",
            "true",
            "yes",
        ):
            from spot_backend.library_index import LibraryIndex
            from spot_backend.ollama_embed import HashingEmbedder

            profile = (case_input.mock_spotify_profile or "").strip().lower()
            if profile in ("canonical_library", "eval_library"):
                from evals.fixture_embedder import CommittedFixtureEmbedder

                embedder = CommittedFixtureEmbedder()
            else:
                embedder = HashingEmbedder()
            try:
                LibraryIndex(settings=settings, embedder=embedder).sync_full()
            except Exception:
                pass
        for ev in iter_chat_events(
            case_input.prompt,
            settings,
            history=case_input.history or None,
            conversation_id=conv_id,
        ):
            et = ev.get("type")
            if et == "llm_delta":
                saw_llm = True
            if et == "tool_done":
                name = str(ev.get("name") or "")
                preview = str(ev.get("preview") or "")
                tool_calls.append(ToolCallRecord(name=name, arguments={}, result=preview))
                tool_results.append(preview)
            if et == "final":
                reply = str(ev.get("text") or "")
    duration = time.perf_counter() - t0
    shortcut = not saw_llm and bool(tool_calls)
    return AgentTurnResult(
        reply=reply.strip(),
        tool_calls=tool_calls,
        shortcut=shortcut,
        llm_invoked=saw_llm,
        duration_s=duration,
        provider=prov,
        acted_uris=_extract_acted_uris(tool_calls),
        extra={"tool_results": tool_results},
    )


def eval_task(raw_input: Any, *, provider: str | None = None, force_mock: bool = False) -> AgentTurnResult:
    """Dataset task entrypoint."""
    case_input = _parse_input(raw_input)
    if force_mock:
        return run_mock_case(case_input)
    return run_live_case(case_input, provider=provider)
