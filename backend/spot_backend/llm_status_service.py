"""Build /api/llm status payloads for each provider."""

from __future__ import annotations

import httpx

from spot_backend.config import Settings
from spot_backend.llm_catalog import (
    PICKER_MODELS_BY_PROVIDER,
    PROVIDER_DISPLAY_NAME,
    SECRET_KEY_BY_PROVIDER,
    catalog_for_api,
    normalize_provider,
)
from spot_backend.llm_prefs import (
    model_override_active,
    read_effective_llm_provider,
    read_effective_model_for_provider,
)
from spot_backend.llm_provider_lists import (
    fetch_gemini_chat_model_names,
    fetch_ollama_model_names,
    ollama_model_name_matches_installed,
)
from spot_backend.llm_secret_safety import redact_known_api_keys
from spot_backend.secrets_store import mask_secret, provider_api_key_configured, read_secret


def _cloud_key(settings: Settings, provider: str) -> str:
    field = SECRET_KEY_BY_PROVIDER.get(provider, "")
    if not field:
        return ""
    return (getattr(settings, field, "") or "").strip()


def llm_status_payload(settings: Settings, *, ui_override: bool) -> dict[str, object]:
    env_provider = normalize_provider(settings.llm_provider or "ollama") or "ollama"
    active = normalize_provider(
        read_effective_llm_provider(settings.data_dir, settings.llm_provider)
    ) or "ollama"

    effective = read_effective_model_for_provider(settings.data_dir, active, settings)
    out: dict[str, object] = {
        "provider": active,
        "provider_display": PROVIDER_DISPLAY_NAME.get(active, active),
        "env_provider": env_provider,
        "ui_override": ui_override,
        "configured_model": effective,
        "reachable": False,
        "error": None,
        "models": list(PICKER_MODELS_BY_PROVIDER.get(active, [])),
        "model_installed": True,
        "catalog": catalog_for_api(),
    }

    if active == "ollama":
        base = settings.ollama_host.rstrip("/")
        out["configured_host"] = base
        out["env_ollama_model"] = settings.ollama_model
        out["ollama_model_ui_override"] = model_override_active(settings.data_dir, "ollama")
        try:
            names = fetch_ollama_model_names(base, timeout=5.0)
            out["reachable"] = True
            out["models"] = names
            out["model_installed"] = ollama_model_name_matches_installed(names, effective)
        except httpx.RequestError as e:
            out["error"] = redact_known_api_keys(str(e))
        except httpx.HTTPStatusError as e:
            out["error"] = redact_known_api_keys(
                f"HTTP {e.response.status_code}: {(e.response.text or '')[:200]}"
            )
        return out

    secret_key = SECRET_KEY_BY_PROVIDER.get(active, "")
    env_key = _cloud_key(settings, active)
    masked = ""
    if secret_key and provider_api_key_configured(settings.data_dir, secret_key, env_key):
        masked = mask_secret(read_secret(settings.data_dir, secret_key) or env_key)
    out["api_key_masked"] = masked
    out["api_key_configured"] = bool(masked or env_key)
    out[f"env_{active}_model"] = _env_model_field(settings, active)
    out[f"{active}_model_ui_override"] = model_override_active(settings.data_dir, active)

    if not env_key and not (secret_key and read_secret(settings.data_dir, secret_key)):
        out["error"] = (
            f"{PROVIDER_DISPLAY_NAME.get(active, active)} API key is not configured. "
            "Use Settings or backend/.env."
        )
        return out

    key = env_key or read_secret(settings.data_dir, secret_key)
    try:
        if active == "gemini":
            names = fetch_gemini_chat_model_names(key, page_size=200, timeout=10.0)
            out["reachable"] = True
            out["models"] = names
            want = effective.strip().lower()
            out["model_installed"] = any(isinstance(n, str) and n.lower() == want for n in names)
        else:
            # OpenAI / Anthropic / xAI: use curated picker list; optional lightweight probe.
            out["reachable"] = True
            out["models"] = list(PICKER_MODELS_BY_PROVIDER.get(active, []))
            out["model_installed"] = effective in out["models"]
    except httpx.RequestError as e:
        out["error"] = redact_known_api_keys(str(e), [key])
    except httpx.HTTPStatusError as e:
        out["error"] = redact_known_api_keys(
            f"HTTP {e.response.status_code}: {(e.response.text or '')[:200]}",
            [key],
        )
    return out


def _env_model_field(settings: Settings, provider: str) -> str:
    return {
        "gemini": settings.gemini_model,
        "openai": settings.openai_model,
        "anthropic": settings.anthropic_model,
        "xai": settings.xai_model,
    }.get(provider, "")
