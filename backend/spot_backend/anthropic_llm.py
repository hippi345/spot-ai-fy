"""Anthropic Messages API tool loop (Claude)."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Iterator

import httpx

from spot_backend.agent import _coerce_chat_history
from spot_backend.chat_tool_state import seed_runner_from_chat_history
from spot_backend.config import Settings
from spot_backend.context_loader import load_optional_agent_context_markdown
from spot_backend.deterministic_chat import gemini_deterministic_shortcut_reply
from spot_backend.gemini_llm import _SYSTEM as _SHARED_SYSTEM
from spot_backend.llm_prefs import read_effective_model_for_provider
from spot_backend.llm_secret_safety import redact_known_api_keys
from spot_backend.llm_tool_loop import ToolLoopState, finalize_assistant_text, run_tool_calls
from spot_backend.prompt_intent import informational_system_suffix, prompt_is_informational
from spot_backend.spotify_tools import OLLAMA_TOOLS, SpotifyToolRunner

logger = logging.getLogger(__name__)

_ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


def _openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str):
            continue
        out.append(
            {
                "name": name,
                "description": str(fn.get("description") or "")[:4000],
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return out


def _history_to_anthropic_messages(hist: list[dict[str, str]], user_text: str) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    for turn in hist:
        role = "user" if turn["role"] == "user" else "assistant"
        msgs.append({"role": role, "content": turn["content"]})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def run_chat_turn_anthropic(
    user_text: str,
    settings: Settings,
    history: list[dict[str, str]] | None = None,
    *,
    emit: Callable[[dict[str, Any]], None] | None = None,
    conversation_id: str | None = None,
) -> str:
    key = (settings.anthropic_api_key or "").strip()
    if not key:
        return "ANTHROPIC_API_KEY is not set. Add it in Settings or backend/.env."

    model = read_effective_model_for_provider(settings.data_dir, "anthropic", settings)
    runner = SpotifyToolRunner(settings=settings, conversation_id=conversation_id)
    hist = _coerce_chat_history(history)
    seed_runner_from_chat_history(runner, hist, conversation_id=conversation_id)
    shortcut = gemini_deterministic_shortcut_reply(
        user_text, runner, conversation_id=conversation_id, emit=emit
    )
    if shortcut is not None:
        runner.close()
        return shortcut

    informational_turn = prompt_is_informational(user_text)
    system = _SHARED_SYSTEM + load_optional_agent_context_markdown(settings)
    if informational_turn:
        system = system + informational_system_suffix(user_text)

    messages = _history_to_anthropic_messages(hist, user_text)
    tools = _openai_tools_to_anthropic(OLLAMA_TOOLS)
    state = ToolLoopState()
    headers = {
        "x-api-key": key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    max_steps = int(settings.agent_max_steps)

    try:
        with httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0)) as client:
            for _ in range(max_steps):
                body: dict[str, Any] = {
                    "model": model,
                    "max_tokens": 8192,
                    "system": system,
                    "messages": messages,
                    "tools": tools,
                }
                resp = client.post(_ANTHROPIC_API, headers=headers, json=body)
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    detail = redact_known_api_keys((exc.response.text or "")[:500], [key])
                    return f"Anthropic request failed (HTTP {exc.response.status_code}). {detail[:240]}"
                data = resp.json()
                content = data.get("content")
                if not isinstance(content, list):
                    return "Unexpected Anthropic response shape."

                tool_uses: list[dict[str, Any]] = []
                text_chunks: list[str] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use":
                        tool_uses.append(block)
                    elif block.get("type") == "text" and isinstance(block.get("text"), str):
                        text_chunks.append(block["text"])

                if tool_uses:
                    messages.append({"role": "assistant", "content": content})
                    openai_style_calls: list[dict[str, Any]] = []
                    for tu in tool_uses:
                        tid = str(tu.get("id") or tu.get("name") or "tool")
                        name = str(tu.get("name") or "")
                        args = tu.get("input") if isinstance(tu.get("input"), dict) else {}
                        openai_style_calls.append(
                            {
                                "id": tid,
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(args)},
                            }
                        )
                    tool_msgs = run_tool_calls(
                        runner,
                        openai_style_calls,
                        state,
                        informational_turn=informational_turn,
                        emit=emit,
                    )
                    tool_result_blocks: list[dict[str, Any]] = []
                    for tm in tool_msgs:
                        tool_result_blocks.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tm.get("tool_call_id"),
                                "content": tm.get("content", ""),
                            }
                        )
                    messages.append({"role": "user", "content": tool_result_blocks})
                    continue

                joined = "\n".join(text_chunks).strip()
                action = finalize_assistant_text(joined, state, user_text=user_text)
                if action.kind == "reprompt":
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": action.reprompt_user_content})
                    continue
                if action.kind == "return":
                    return action.text
        return "Stopped after maximum tool steps. Try a simpler request."
    except httpx.HTTPError as exc:
        msg = redact_known_api_keys(str(exc), [key])
        logger.warning("anthropic http error: %s", msg)
        return f"Could not reach Anthropic ({msg[:200]})."
    finally:
        runner.close()


def iter_anthropic_chat_events(
    user_text: str,
    settings: Settings,
    history: list[dict[str, str]] | None = None,
    *,
    conversation_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    def _emit(ev: dict[str, Any]) -> None:
        events.append(ev)

    yield {"type": "status", "message": "Calling Anthropic…"}
    try:
        text = run_chat_turn_anthropic(
            user_text,
            settings,
            history=history,
            emit=_emit,
            conversation_id=conversation_id,
        )
        for ev in events:
            yield ev
        yield {"type": "final", "text": text}
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": redact_known_api_keys(f"{type(exc).__name__}: {exc}")}
