"""Spot-AI-fy: persisted LLM prefs (provider + per-provider model overrides)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spot_backend.config import Settings
from spot_backend.json_store import load_json_dict
from spot_backend.llm_catalog import ALL_LLM_PROVIDERS, DEFAULT_MODEL_BY_PROVIDER, normalize_provider

_PREFS_FILE = "llm_provider.json"

_MODEL_PREF_KEYS = {
    "ollama": "ollama_model",
    "gemini": "gemini_model",
    "openai": "openai_model",
    "anthropic": "anthropic_model",
    "xai": "xai_model",
}


def _prefs_path(data_dir: Path) -> Path:
    return data_dir / _PREFS_FILE


def _load_prefs_raw(data_dir: Path) -> dict[str, Any]:
    return load_json_dict(_prefs_path(data_dir))


def _persist_prefs(data_dir: Path, raw: dict[str, Any]) -> None:
    """Write normalized prefs, or remove the file if nothing to store."""
    from spot_backend.data_dir_lock import data_dir_lock

    with data_dir_lock(data_dir):
        _persist_prefs_unlocked(data_dir, raw)


def _persist_prefs_unlocked(data_dir: Path, raw: dict[str, Any]) -> None:
    clean: dict[str, Any] = {}
    p = normalize_provider(str(raw.get("provider", "")))
    if p:
        clean["provider"] = p
    for prov, key in _MODEL_PREF_KEYS.items():
        val = raw.get(key)
        if isinstance(val, str) and val.strip():
            clean[key] = val.strip()
    sm = raw.get("ollama_small_model")
    if isinstance(sm, str) and sm.strip().lower() in ("auto", "on", "off"):
        clean["ollama_small_model"] = sm.strip().lower()
    path = _prefs_path(data_dir)
    if not clean:
        if path.is_file():
            path.unlink()
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")


def read_effective_llm_provider(data_dir: Path, env_provider: str) -> str:
    """UI override file wins for `provider` when set; otherwise use LLM_PROVIDER from settings (.env)."""
    raw = _load_prefs_raw(data_dir)
    p = normalize_provider(str(raw.get("provider", "")))
    if p:
        return p
    env_p = normalize_provider(env_provider or "ollama")
    return env_p or "ollama"


def _env_model_for_provider(settings: Settings, provider: str) -> str:
    mapping = {
        "ollama": settings.ollama_model,
        "gemini": settings.gemini_model,
        "openai": settings.openai_model,
        "anthropic": settings.anthropic_model,
        "xai": settings.xai_model,
    }
    return (mapping.get(provider) or DEFAULT_MODEL_BY_PROVIDER.get(provider, "")).strip()


def _strip_gemini_prefix(name: str) -> str:
    n = name.strip()
    return n.split("/", 1)[1] if n.startswith("models/") else n


def read_effective_model_for_provider(data_dir: Path, provider: str, settings: Settings) -> str:
    prov = normalize_provider(provider) or "ollama"
    raw = _load_prefs_raw(data_dir)
    key = _MODEL_PREF_KEYS[prov]
    override = raw.get(key)
    if isinstance(override, str) and override.strip():
        val = override.strip()
        return _strip_gemini_prefix(val) if prov == "gemini" else val
    env_val = _env_model_for_provider(settings, prov)
    if prov == "gemini":
        return _strip_gemini_prefix(env_val or DEFAULT_MODEL_BY_PROVIDER["gemini"])
    return env_val or DEFAULT_MODEL_BY_PROVIDER.get(prov, "")


def read_effective_ollama_model(data_dir: Path, env_model: str) -> str:
    return read_effective_model_for_provider(
        data_dir, "ollama", Settings(ollama_model=env_model or DEFAULT_MODEL_BY_PROVIDER["ollama"])
    )


def read_effective_gemini_model(data_dir: Path, env_model: str) -> str:
    return read_effective_model_for_provider(
        data_dir, "gemini", Settings(gemini_model=env_model or DEFAULT_MODEL_BY_PROVIDER["gemini"])
    )


def write_llm_provider(data_dir: Path, provider: str) -> None:
    p = normalize_provider(provider)
    if not p:
        raise ValueError(f"provider must be one of {', '.join(ALL_LLM_PROVIDERS)}")
    raw = _load_prefs_raw(data_dir)
    raw["provider"] = p
    _persist_prefs(data_dir, raw)


def write_model_override(data_dir: Path, provider: str, model: str | None) -> None:
    prov = normalize_provider(provider)
    if not prov:
        raise ValueError("unknown provider")
    key = _MODEL_PREF_KEYS[prov]
    raw = _load_prefs_raw(data_dir)
    if model is not None and str(model).strip():
        val = str(model).strip()
        if prov == "gemini":
            val = _strip_gemini_prefix(val)
        raw[key] = val
    else:
        raw.pop(key, None)
    _persist_prefs(data_dir, raw)


def write_ollama_model_override(data_dir: Path, model: str | None) -> None:
    write_model_override(data_dir, "ollama", model)


def write_gemini_model_override(data_dir: Path, model: str | None) -> None:
    write_model_override(data_dir, "gemini", model)


def clear_llm_provider_override(data_dir: Path) -> None:
    path = _prefs_path(data_dir)
    if path.is_file():
        path.unlink()


def prefs_path_exists(data_dir: Path) -> bool:
    return _prefs_path(data_dir).is_file()


def model_override_active(data_dir: Path, provider: str) -> bool:
    prov = normalize_provider(provider)
    if not prov:
        return False
    key = _MODEL_PREF_KEYS[prov]
    val = _load_prefs_raw(data_dir).get(key)
    return isinstance(val, str) and bool(val.strip())


def ollama_model_override_active(data_dir: Path) -> bool:
    return model_override_active(data_dir, "ollama")


def gemini_model_override_active(data_dir: Path) -> bool:
    return model_override_active(data_dir, "gemini")


def read_ollama_small_model_mode(data_dir: Path) -> str:
    raw = _load_prefs_raw(data_dir).get("ollama_small_model")
    if isinstance(raw, str) and raw.strip().lower() in ("auto", "on", "off"):
        return raw.strip().lower()
    return "auto"


def write_ollama_small_model_mode(data_dir: Path, mode: str) -> None:
    m = mode.strip().lower()
    if m not in ("auto", "on", "off"):
        raise ValueError("ollama_small_model must be auto, on, or off")
    raw = _load_prefs_raw(data_dir)
    raw["ollama_small_model"] = m
    _persist_prefs(data_dir, raw)
