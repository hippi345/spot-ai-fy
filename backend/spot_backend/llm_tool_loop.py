"""Shared Spotify tool-call loop helpers (dedupe, reprompts, device sanitizing hooks)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from spot_backend.action_claim_guard import (
    action_claim_honest_fallback,
    action_claim_reprompt,
    is_failure_boilerplate,
    numeric_factual_claim_honest_fallback,
    record_successful_tool,
    reply_claims_unbacked_action,
    reply_contains_unbacked_numeric_factual_claim,
    tool_summarize_reprompt,
    turn_tool_calls_all_succeeded,
)
from spot_backend.chat_messages import (
    PROMISE_AFTER_ID_ERROR_NUDGE,
    assistant_reply_is_promise_only,
    prepare_user_visible_reply,
    tool_result_is_rejected_or_invalid_id,
)
from spot_backend.prompt_intent import (
    refused_mutating_tool_result,
    spotify_tool_is_mutating,
)
from spot_backend.reply_tool_trace import (
    append_tool_trace_record,
    summarize_tool_args,
    tool_trace_outcome,
)
from spot_backend.spotify_tools import SpotifyToolRunner, _sanitize_model_device_id
from spot_backend.tool_server_enforcement import (
    enforce_tool_arguments_for_turn,
    rewrite_tool_call_for_turn,
)

EmitFn = Callable[[dict[str, Any]], None]


@dataclass
class ToolLoopState:
    successful_tools: set[str] = field(default_factory=set)
    turn_tool_calls: list[tuple[str, str]] = field(default_factory=list)
    deduped_tool_results: dict[tuple[str, str], str] = field(default_factory=dict)
    tool_results: list[str] = field(default_factory=list)
    promise_nudge_used: bool = False
    action_claim_reprompted: bool = False
    tool_summarize_reprompted: bool = False


def sanitize_tool_arguments(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Normalize model-emitted args (device_id placeholders, JSON strings)."""
    out = dict(args) if isinstance(args, dict) else {}
    if "device_id" in out:
        out["device_id"] = _sanitize_model_device_id(str(out.get("device_id") or ""))
    return out


def tool_dedupe_key(name: str, args: dict[str, Any]) -> tuple[str, str]:
    return (name, json.dumps(args, sort_keys=True, default=str))


def run_tool_calls(
    runner: SpotifyToolRunner,
    tool_calls: list[dict[str, Any]],
    state: ToolLoopState,
    *,
    informational_turn: bool,
    user_text: str = "",
    emit: EmitFn | None = None,
    tool_result_cap: int = 12_000,
    trace_data_dir: Any | None = None,
    trace_conversation_id: str | None = None,
    trace_secrets: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Execute tool calls; returns OpenAI-style tool messages to append."""
    tool_messages: list[dict[str, Any]] = []
    for tc in tool_calls:
        fn = tc.get("function") or tc
        name = fn.get("name") if isinstance(fn, dict) else tc.get("name")
        raw_args = fn.get("arguments") if isinstance(fn, dict) else tc.get("arguments")
        if not isinstance(name, str) or not name.strip():
            continue
        args: dict[str, Any] = {}
        if isinstance(raw_args, dict):
            args = sanitize_tool_arguments(name, raw_args)
        elif isinstance(raw_args, str) and raw_args.strip():
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict):
                    args = sanitize_tool_arguments(name, parsed)
            except json.JSONDecodeError:
                args = {}
        dedupe_key = tool_dedupe_key(name, args)
        if dedupe_key in state.deduped_tool_results:
            result = state.deduped_tool_results[dedupe_key]
            if emit:
                emit({"type": "tool_start", "name": name})
                preview = result[:240] + ("…" if len(result) > 240 else "")
                emit({"type": "tool_done", "name": name, "preview": preview})
            content = result[:tool_result_cap] if tool_result_cap > 0 else result
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(tc.get("id") or name),
                    "name": name,
                    "content": content,
                }
            )
            continue
        if emit:
            emit({"type": "tool_start", "name": name})
        if informational_turn and spotify_tool_is_mutating(name):
            result = refused_mutating_tool_result(name)
            if trace_data_dir is not None:
                from pathlib import Path

                append_tool_trace_record(
                    Path(trace_data_dir),
                    conversation_id=trace_conversation_id,
                    tool_name=name,
                    args_summary=summarize_tool_args(args),
                    outcome=tool_trace_outcome(result),
                    known_secrets=trace_secrets,
                    raw_result=result,
                )
        else:
            import time as _time

            name, args = rewrite_tool_call_for_turn(
                name, args, user_text=user_text, runner=runner
            )
            args = enforce_tool_arguments_for_turn(
                name, args, user_text=user_text, runner=runner
            )
            t0 = _time.perf_counter()
            result = runner.run(name, args)
            duration_ms = int((_time.perf_counter() - t0) * 1000)
            if trace_data_dir is not None:
                from pathlib import Path

                append_tool_trace_record(
                    Path(trace_data_dir),
                    conversation_id=trace_conversation_id,
                    tool_name=name,
                    args_summary=summarize_tool_args(args),
                    outcome=tool_trace_outcome(result),
                    duration_ms=duration_ms,
                    known_secrets=trace_secrets,
                    raw_result=result,
                )
        state.deduped_tool_results[dedupe_key] = result
        state.turn_tool_calls.append((name, result))
        state.tool_results.append(result)
        if not (informational_turn and spotify_tool_is_mutating(name)):
            record_successful_tool(state.successful_tools, name, result)
        if emit:
            preview = result[:240] + ("…" if len(result) > 240 else "")
            emit({"type": "tool_done", "name": name, "preview": preview})
        content = result[:tool_result_cap] if tool_result_cap > 0 else result
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": str(tc.get("id") or name),
                "name": name,
                "content": content,
            }
        )
    from spot_backend.turn_reply_intent import (
        intent_needs_library_contains_fallback,
        intent_needs_library_save_fallback,
        library_contains_fallback_args,
        library_save_fallback_args,
    )

    if intent_needs_library_contains_fallback(user_text, [n for n, _ in state.turn_tool_calls]):
        fb_args = enforce_tool_arguments_for_turn(
            "spotify_library_contains",
            library_contains_fallback_args(user_text),
            user_text=user_text,
            runner=runner,
        )
        fb_raw = runner.run("spotify_library_contains", fb_args)
        state.turn_tool_calls.append(("spotify_library_contains", fb_raw))
        state.tool_results.append(fb_raw)
        record_successful_tool(state.successful_tools, "spotify_library_contains", fb_raw)
        if emit:
            emit({"type": "tool_start", "name": "spotify_library_contains"})
            preview = fb_raw[:240] + ("…" if len(fb_raw) > 240 else "")
            emit({"type": "tool_done", "name": "spotify_library_contains", "preview": preview})
        if trace_data_dir is not None:
            from pathlib import Path

            append_tool_trace_record(
                Path(trace_data_dir),
                conversation_id=trace_conversation_id,
                tool_name="spotify_library_contains",
                args_summary=summarize_tool_args(fb_args),
                outcome=tool_trace_outcome(fb_raw),
                known_secrets=trace_secrets,
                raw_result=fb_raw,
            )
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": "spotify_library_contains_fallback",
                "name": "spotify_library_contains",
                "content": fb_raw[:tool_result_cap] if tool_result_cap > 0 else fb_raw,
            }
        )
    run_library_save_fallback_if_needed(
        runner,
        state,
        user_text=user_text,
        emit=emit,
        trace_data_dir=trace_data_dir,
        trace_conversation_id=trace_conversation_id,
        trace_secrets=trace_secrets,
        tool_result_cap=tool_result_cap,
        tool_messages=tool_messages,
    )
    return tool_messages


def run_library_save_fallback_if_needed(
    runner: SpotifyToolRunner,
    state: ToolLoopState,
    *,
    user_text: str,
    emit: EmitFn | None = None,
    trace_data_dir: Any | None = None,
    trace_conversation_id: str | None = None,
    trace_secrets: list[str] | None = None,
    tool_result_cap: int = 12_000,
    tool_messages: list[dict[str, Any]] | None = None,
) -> str | None:
    from spot_backend.turn_reply_intent import (
        intent_needs_library_save_fallback,
        library_save_fallback_args,
    )

    if not intent_needs_library_save_fallback(user_text, [n for n, _ in state.turn_tool_calls]):
        return None
    save_args = enforce_tool_arguments_for_turn(
        "spotify_library_save",
        library_save_fallback_args(user_text),
        user_text=user_text,
        runner=runner,
    )
    save_raw = runner.run("spotify_library_save", save_args)
    state.turn_tool_calls.append(("spotify_library_save", save_raw))
    state.tool_results.append(save_raw)
    record_successful_tool(state.successful_tools, "spotify_library_save", save_raw)
    if emit:
        emit({"type": "tool_start", "name": "spotify_library_save"})
        preview = save_raw[:240] + ("…" if len(save_raw) > 240 else "")
        emit({"type": "tool_done", "name": "spotify_library_save", "preview": preview})
    if trace_data_dir is not None:
        from pathlib import Path

        append_tool_trace_record(
            Path(trace_data_dir),
            conversation_id=trace_conversation_id,
            tool_name="spotify_library_save",
            args_summary=summarize_tool_args(save_args),
            outcome=tool_trace_outcome(save_raw),
            known_secrets=trace_secrets,
            raw_result=save_raw,
        )
    if tool_messages is not None:
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": "spotify_library_save_fallback",
                "name": "spotify_library_save",
                "content": save_raw[:tool_result_cap] if tool_result_cap > 0 else save_raw,
            }
        )
    return save_raw


@dataclass
class TextFinalizeAction:
    kind: str  # return | reprompt | continue
    text: str = ""
    reprompt_user_content: str = ""


def finalize_assistant_text(
    text: str,
    state: ToolLoopState,
    *,
    user_text: str,
) -> TextFinalizeAction:
    joined = (text or "").strip()
    if not joined:
        return TextFinalizeAction(kind="continue")
    if (
        assistant_reply_is_promise_only(joined)
        and state.tool_results
        and tool_result_is_rejected_or_invalid_id(state.tool_results[-1])
        and not state.promise_nudge_used
    ):
        state.promise_nudge_used = True
        return TextFinalizeAction(kind="reprompt", reprompt_user_content=PROMISE_AFTER_ID_ERROR_NUDGE)
    if is_failure_boilerplate(joined):
        from spot_backend.reply_tool_fallback import best_tool_summary_fallback

        tool_names = [n for n, _ in state.turn_tool_calls]
        fallback = best_tool_summary_fallback(
            state.tool_results,
            user_text=user_text,
            tool_names=tool_names,
        )
        if fallback:
            return TextFinalizeAction(
                kind="return",
                text=prepare_user_visible_reply(fallback, state.tool_results),
            )
    if is_failure_boilerplate(joined) and turn_tool_calls_all_succeeded(state.turn_tool_calls):
        if not state.tool_summarize_reprompted:
            state.tool_summarize_reprompted = True
            return TextFinalizeAction(
                kind="reprompt",
                reprompt_user_content=tool_summarize_reprompt(state.tool_results),
            )
    if reply_claims_unbacked_action(
        joined,
        state.successful_tools,
        user_text=user_text,
        turn_tool_calls=state.turn_tool_calls,
    ):
        if not state.action_claim_reprompted:
            state.action_claim_reprompted = True
            return TextFinalizeAction(kind="reprompt", reprompt_user_content=action_claim_reprompt())
        if (
            not state.tool_summarize_reprompted
            and turn_tool_calls_all_succeeded(state.turn_tool_calls)
        ):
            state.tool_summarize_reprompted = True
            return TextFinalizeAction(
                kind="reprompt",
                reprompt_user_content=tool_summarize_reprompt(state.tool_results),
            )
        from spot_backend.reply_tool_fallback import best_tool_summary_fallback

        tool_names = [n for n, _ in state.turn_tool_calls]
        fallback = best_tool_summary_fallback(
            state.tool_results,
            user_text=user_text,
            tool_names=tool_names,
        )
        if fallback:
            return TextFinalizeAction(
                kind="return",
                text=prepare_user_visible_reply(fallback, state.tool_results),
            )
        return TextFinalizeAction(kind="return", text=action_claim_honest_fallback())
    if reply_contains_unbacked_numeric_factual_claim(
        joined, state.turn_tool_calls
    ):
        return TextFinalizeAction(
            kind="return",
            text=numeric_factual_claim_honest_fallback(),
        )
    from spot_backend.reply_tool_fallback import apply_tool_grounded_reply

    tool_names = [n for n, _ in state.turn_tool_calls]
    grounded = apply_tool_grounded_reply(
        joined,
        state.tool_results,
        user_text=user_text,
        tool_names=tool_names,
    )
    return TextFinalizeAction(
        kind="return",
        text=prepare_user_visible_reply(
            grounded,
            state.tool_results,
            tool_names=tool_names,
            user_text=user_text,
        ),
    )
