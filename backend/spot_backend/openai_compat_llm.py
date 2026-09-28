"""OpenAI Chat Completions tool loop (OpenAI + xAI Grok API)."""

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
from spot_backend.agent_system_extras import shared_agent_system_suffix
from spot_backend.gemini_llm import _SYSTEM as _SHARED_SYSTEM
from spot_backend.llm_prefs import read_effective_model_for_provider
from spot_backend.llm_secret_safety import redact_known_api_keys
from spot_backend.llm_tool_loop import ToolLoopState, finalize_assistant_text, run_tool_calls
from spot_backend.prompt_intent import informational_system_suffix, prompt_is_informational
from spot_backend.spotify_tools import OLLAMA_TOOLS, SpotifyToolRunner

logger = logging.getLogger(__name__)


def _parse_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    raw = message.get("tool_calls")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for tc in raw:
        if isinstance(tc, dict) and tc.get("type") == "function":
            out.append(tc)
    return out


def _assistant_message_from_api(message: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        out["tool_calls"] = message["tool_calls"]
    return out


def run_chat_turn_openai_compat(
    user_text: str,
    settings: Settings,
    *,
    base_url: str,
    api_key: str,
    provider_id: str,
    history: list[dict[str, str]] | None = None,
    emit: Callable[[dict[str, Any]], None] | None = None,
    conversation_id: str | None = None,
    status_label: str = "Calling model…",
) -> str:
    key = (api_key or "").strip()
    if not key:
        return f"{provider_id.upper()} API key is not set. Add it in Settings or backend/.env."

    model = read_effective_model_for_provider(settings.data_dir, provider_id, settings)
    runner = SpotifyToolRunner(settings=settings, conversation_id=conversation_id)
    hist = _coerce_chat_history(history)
    seed_runner_from_chat_history(runner, hist, conversation_id=conversation_id)
    shortcut = gemini_deterministic_shortcut_reply(
        user_text,
        runner,
        conversation_id=conversation_id,
        emit=emit,
        settings=settings,
        known_secrets=[key],
    )
    if shortcut is not None:
        runner.close()
        return shortcut

    informational_turn = prompt_is_informational(user_text)
    system = _SHARED_SYSTEM + shared_agent_system_suffix() + load_optional_agent_context_markdown(settings)
    if informational_turn:
        system = system + informational_system_suffix(user_text)

    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for turn in hist:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_text})

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    state = ToolLoopState()
    max_steps = int(settings.agent_max_steps)

    try:
        with httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0)) as client:
            for _ in range(max_steps):
                body: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "tools": OLLAMA_TOOLS,
                    "tool_choice": "auto",
                }
                resp = client.post(url, headers=headers, json=body)
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    detail = redact_known_api_keys((exc.response.text or "")[:500], [key])
                    return (
                        f"{provider_id} request failed (HTTP {exc.response.status_code}). "
                        f"{detail[:240]}"
                    )
                data = resp.json()
                choices = data.get("choices")
                if not isinstance(choices, list) or not choices:
                    return f"No response from {provider_id}."
                message = choices[0].get("message")
                if not isinstance(message, dict):
                    return f"Unexpected {provider_id} response shape."

                tool_calls = _parse_tool_calls(message)
                if tool_calls:
                    messages.append(_assistant_message_from_api(message))
                    tool_msgs = run_tool_calls(
                        runner,
                        tool_calls,
                        state,
                        informational_turn=informational_turn,
                        user_text=user_text,
                        emit=emit,
                        trace_data_dir=settings.data_dir,
                        trace_conversation_id=conversation_id,
                        trace_secrets=[key],
                    )
                    messages.extend(tool_msgs)
                    continue

                content = message.get("content")
                text = content if isinstance(content, str) else ""
                action = finalize_assistant_text(text, state, user_text=user_text)
                if action.kind == "reprompt":
                    messages.append(_assistant_message_from_api(message))
                    messages.append({"role": "user", "content": action.reprompt_user_content})
                    continue
                if action.kind == "return":
                    return action.text
        return (
            "Stopped after maximum tool steps. Try a simpler request."
        )
    except httpx.HTTPError as exc:
        msg = redact_known_api_keys(str(exc), [key])
        logger.warning("%s http error: %s", provider_id, msg)
        return f"Could not reach {provider_id} ({msg[:200]})."
    finally:
        runner.close()


def iter_openai_compat_chat_events(
    user_text: str,
    settings: Settings,
    *,
    base_url: str,
    api_key: str,
    provider_id: str,
    history: list[dict[str, str]] | None = None,
    conversation_id: str | None = None,
    status_label: str,
) -> Iterator[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    def _emit(ev: dict[str, Any]) -> None:
        events.append(ev)

    yield {"type": "status", "message": status_label}
    try:
        text = run_chat_turn_openai_compat(
            user_text,
            settings,
            base_url=base_url,
            api_key=api_key,
            provider_id=provider_id,
            history=history,
            emit=_emit,
            conversation_id=conversation_id,
            status_label=status_label,
        )
        for ev in events:
            yield ev
        yield {"type": "final", "text": text}
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": redact_known_api_keys(f"{type(exc).__name__}: {exc}")}


def iter_openai_chat_events(
    user_text: str,
    settings: Settings,
    history: list[dict[str, str]] | None = None,
    *,
    conversation_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    key = (settings.openai_api_key or "").strip()
    yield from iter_openai_compat_chat_events(
        user_text,
        settings,
        base_url="https://api.openai.com/v1",
        api_key=key,
        provider_id="openai",
        history=history,
        conversation_id=conversation_id,
        status_label="Calling OpenAI…",
    )


def iter_xai_chat_events(
    user_text: str,
    settings: Settings,
    history: list[dict[str, str]] | None = None,
    *,
    conversation_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    key = (settings.xai_api_key or "").strip()
    yield from iter_openai_compat_chat_events(
        user_text,
        settings,
        base_url="https://api.x.ai/v1",
        api_key=key,
        provider_id="xai",
        history=history,
        conversation_id=conversation_id,
        status_label="Calling xAI…",
    )
