"""Spot-AI-fy: persisted LLM prefs (Ollama vs Gemini, optional Ollama model tag)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_PREFS_FILE = "llm_provider.json"


def _prefs_path(data_dir: Path) -> Path:
    return data_dir / _PREFS_FILE


def _load_prefs_raw(data_dir: Path) -> dict[str, Any]:
    path = _prefs_path(data_dir)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (json.JSONDecodeError, OSError, TypeError):
        return {}


def _persist_prefs(data_dir: Path, raw: dict[str, Any]) -> None:
    """Write normalized prefs, or remove the file if nothing to store."""
    clean: dict[str, Any] = {}
    p = str(raw.get("provider", "")).strip().lower()
    if p in ("ollama", "gemini"):
        clean["provider"] = p
    om = raw.get("ollama_model")
    if isinstance(om, str) and om.strip():
        clean["ollama_model"] = om.strip()
    gm = raw.get("gemini_model")
    if isinstance(gm, str) and gm.strip():
        clean["gemini_model"] = gm.strip()
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
    p = str(raw.get("provider", "")).strip().lower()
    if p in ("ollama", "gemini"):
        return p
    return (env_provider or "ollama").strip().lower()


def read_effective_ollama_model(data_dir: Path, env_model: str) -> str:
    """Optional `ollama_model` in prefs overrides OLLAMA_MODEL from .env."""
    raw = _load_prefs_raw(data_dir)
    om = raw.get("ollama_model")
    if isinstance(om, str) and om.strip():
        return om.strip()
    return (env_model or "gemma2:2b").strip()


def write_llm_provider(data_dir: Path, provider: str) -> None:
    p = provider.strip().lower()
    if p not in ("ollama", "gemini"):
        raise ValueError("provider must be ollama or gemini")
    raw = _load_prefs_raw(data_dir)
    raw["provider"] = p
    _persist_prefs(data_dir, raw)


def write_ollama_model_override(data_dir: Path, model: str | None) -> None:
    """Persist Ollama model tag, or clear override when model is None/empty."""
    raw = _load_prefs_raw(data_dir)
    if model is not None and str(model).strip():
        raw["ollama_model"] = str(model).strip()
    else:
        raw.pop("ollama_model", None)
    _persist_prefs(data_dir, raw)


def _strip_gemini_prefix(name: str) -> str:
    """Gemini REST returns full-form names ("models/gemini-2.0-flash") but the
    generateContent URL is built as `/models/{name}:…`, so a stored full-form
    name would produce `/models/models/…` → 404. Normalize to the short name."""
    n = name.strip()
    return n.split("/", 1)[1] if n.startswith("models/") else n


def read_effective_gemini_model(data_dir: Path, env_model: str) -> str:
    """Optional `gemini_model` in prefs overrides GEMINI_MODEL from .env."""
    raw = _load_prefs_raw(data_dir)
    gm = raw.get("gemini_model")
    if isinstance(gm, str) and gm.strip():
        return _strip_gemini_prefix(gm)
    return _strip_gemini_prefix(env_model or "gemini-2.5-flash")


def write_gemini_model_override(data_dir: Path, model: str | None) -> None:
    """Persist Gemini model tag, or clear override when model is None/empty."""
    raw = _load_prefs_raw(data_dir)
    if model is not None and str(model).strip():
        raw["gemini_model"] = _strip_gemini_prefix(str(model))
    else:
        raw.pop("gemini_model", None)
    _persist_prefs(data_dir, raw)


def clear_llm_provider_override(data_dir: Path) -> None:
    """Remove all UI LLM overrides (provider and model overrides)."""
    path = _prefs_path(data_dir)
    if path.is_file():
        path.unlink()


def prefs_path_exists(data_dir: Path) -> bool:
    return _prefs_path(data_dir).is_file()


def ollama_model_override_active(data_dir: Path) -> bool:
    om = _load_prefs_raw(data_dir).get("ollama_model")
    return isinstance(om, str) and bool(om.strip())


def gemini_model_override_active(data_dir: Path) -> bool:
    gm = _load_prefs_raw(data_dir).get("gemini_model")
    return isinstance(gm, str) and bool(gm.strip())
