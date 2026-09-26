"""Ollama-specific agent profile: small-model mode, prompt size estimates."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from spot_backend.config import Settings
from spot_backend.llm_prefs import read_ollama_small_model_mode
logger = logging.getLogger(__name__)

SMALL_MODEL_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "spotify_search",
        "spotify_play_playlist",
        "spotify_start_resume_playback",
        "spotify_create_playlist",
        "spotify_top_artists",
        "spotify_top_tracks",
        "spotify_pause",
        "spotify_skip_next",
        "spotify_skip_previous",
        "spotify_add_to_queue",
        "spotify_devices",
        "spotify_playback_state",
        "spotify_user_playlists",
        "spotify_add_tracks_by_query",
        "spotify_set_volume",
        "spotify_set_shuffle",
        "spotify_set_repeat",
        "spotify_unfollow_playlist",
    }
)


def small_model_route_hint(user_text: str) -> str | None:
    """Lightweight intent → tool hint for tests and small-model routing."""
    low = (user_text or "").lower()
    if "playlist" in low and any(w in low for w in ("delete", "remove", "unfollow")):
        return "spotify_unfollow_playlist"
    return None

SMALL_MODEL_SYSTEM_PROMPT = """You are a Spotify assistant with a small set of tools.
Use tools instead of guessing IDs. For play requests use spotify_play_playlist or spotify_start_resume_playback.
For search + add use spotify_add_tracks_by_query. Queue only when the user says next/queue.
Summarize tool results briefly for the user."""


def parse_ollama_model_billions(model_tag: str) -> float | None:
    low = (model_tag or "").lower()
    m = re.search(r":(\d+(?:\.\d+)?)\s*b\b", low)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*b(?:-instruct|-chat)?\b", low)
    if m:
        return float(m.group(1))
    return None


def small_model_auto_enabled(model_tag: str) -> bool:
    billions = parse_ollama_model_billions(model_tag)
    if billions is None:
        return False
    return billions < 8.0


def use_small_model_mode(settings: Settings, model_tag: str) -> bool:
    mode = read_ollama_small_model_mode(settings.data_dir)
    if mode == "on":
        return True
    if mode == "off":
        return False
    return small_model_auto_enabled(model_tag)


def filter_ollama_tools(tools: list[dict[str, Any]], *, small: bool) -> list[dict[str, Any]]:
    if not small:
        return tools
    out: list[dict[str, Any]] = []
    for entry in tools:
        fn = entry.get("function") if isinstance(entry, dict) else None
        if isinstance(fn, dict) and fn.get("name") in SMALL_MODEL_TOOL_NAMES:
            out.append(entry)
    return out


def estimate_prompt_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> int:
    chars = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif content is not None:
            chars += len(str(content))
    if tools:
        chars += len(json.dumps(tools, ensure_ascii=False))
    return max(1, chars // 4)


def apply_ollama_request_tuning(
    body: dict[str, Any],
    settings: Settings,
    ollama_options: dict[str, Any],
    *,
    omit_think_field: bool = False,
) -> None:
    if ollama_options:
        existing = body.get("options")
        if isinstance(existing, dict):
            body["options"] = {**ollama_options, **existing}
        else:
            body["options"] = dict(ollama_options)
    keep_alive = (settings.ollama_keep_alive or "").strip()
    if keep_alive:
        body["keep_alive"] = keep_alive
    if not settings.ollama_think and not omit_think_field:
        body["think"] = False


def should_retry_ollama_without_think(status_code: int, response_text: str, body: dict[str, Any]) -> bool:
    if status_code != 400:
        return False
    if "think" not in body:
        return False
    low = (response_text or "").lower()
    return "think" in low


def maybe_warn_prompt_exceeds_ctx(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    num_ctx: int,
) -> None:
    if num_ctx <= 0:
        return
    est = estimate_prompt_tokens(messages, tools)
    if est > num_ctx:
        logger.warning(
            "Ollama prompt estimate ~%s tokens exceeds num_ctx=%s; Ollama may truncate silently",
            est,
            num_ctx,
        )
