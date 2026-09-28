"""Single source for LLM provider ids, default models, and UI model pick lists."""

from __future__ import annotations

from typing import Literal

LlmProviderId = Literal["ollama", "gemini", "openai", "anthropic", "xai"]

ALL_LLM_PROVIDERS: tuple[LlmProviderId, ...] = (
    "ollama",
    "gemini",
    "openai",
    "anthropic",
    "xai",
)

CLOUD_API_KEY_PROVIDERS: frozenset[LlmProviderId] = frozenset(
    {"gemini", "openai", "anthropic", "xai"}
)

# Default model when env + UI override are unset (also used in docs / tests).
DEFAULT_MODEL_BY_PROVIDER: dict[str, str] = {
    "ollama": "qwen2.5:3b",
    "gemini": "gemini-3.5-flash-lite",
    "openai": "gpt-6-luna",
    "anthropic": "claude-haiku-4-5-20251001",
    "xai": "grok-4.3",
}

# Curated lists for settings dropdowns (live APIs may return more; these are fallbacks).
PICKER_MODELS_BY_PROVIDER: dict[str, list[str]] = {
    "ollama": [
        "qwen2.5:3b",
        "qwen2.5:7b",
        "llama3.2:3b",
        "gemma2:2b",
    ],
    "gemini": [
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.5-flash",
    ],
    "openai": [
        "gpt-6-luna",
        "gpt-6-astra",
        "gpt-6-sol",
    ],
    "anthropic": [
        "claude-haiku-4-5-20251001",
        "claude-opus-5-5",
        "claude-sonnet-5",
    ],
    "xai": [
        "grok-4.3",
        "grok-4.7",
        "grok-4.6",
    ],
}

PROVIDER_DISPLAY_NAME: dict[str, str] = {
    "ollama": "Ollama",
    "gemini": "Gemini",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "xai": "xAI",
}

SECRET_KEY_BY_PROVIDER: dict[str, str] = {
    "gemini": "gemini_api_key",
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
    "xai": "xai_api_key",
}


def normalize_provider(value: str) -> LlmProviderId | None:
    p = (value or "").strip().lower()
    if p in ALL_LLM_PROVIDERS:
        return p  # type: ignore[return-value]
    return None


def catalog_for_api() -> dict[str, object]:
    """Expose defaults and picker lists to the frontend."""
    return {
        "providers": list(ALL_LLM_PROVIDERS),
        "default_models": dict(DEFAULT_MODEL_BY_PROVIDER),
        "picker_models": dict(PICKER_MODELS_BY_PROVIDER),
        "display_names": dict(PROVIDER_DISPLAY_NAME),
    }
