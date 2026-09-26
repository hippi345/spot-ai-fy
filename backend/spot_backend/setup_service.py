"""First-run setup: Spotify app credentials and LLM provider configuration."""

from __future__ import annotations

import logging
from typing import Any, Literal

import httpx

from spot_backend.config import Settings, get_settings
from spot_backend.llm_prefs import (
    read_effective_gemini_model,
    read_effective_llm_provider,
    read_effective_ollama_model,
    write_gemini_model_override,
    write_llm_provider,
    write_ollama_model_override,
)
from spot_backend.secrets_store import (
    gemini_key_configured,
    mask_secret,
    merge_settings_from_store,
    read_secret,
    read_setup_fields,
    spotify_client_configured,
    write_secret,
    write_setup_fields,
)
from spot_backend.spotify_setup import validate_spotify_client_id
from spot_backend.token_store import load_tokens
from spot_backend.url_safety import OllamaUrlNotAllowedError, validate_ollama_base_url

logger = logging.getLogger(__name__)


def _signed_in(settings: Settings) -> bool:
    bundle = load_tokens(settings.resolved_token_path)
    return bool(bundle and bundle.access_token)


def _ollama_ready(
    settings: Settings,
    *,
    model_override: str | None = None,
    allow_public: bool = False,
) -> tuple[bool, str | None]:
    host = settings.ollama_host.rstrip("/")
    model = (model_override or "").strip() or read_effective_ollama_model(
        settings.data_dir, settings.ollama_model
    )
    if not host:
        return False, "Ollama URL is required"
    if not model:
        return False, "Select or enter an Ollama model name"
    try:
        base = validate_ollama_base_url(host, allow_public=allow_public)
    except OllamaUrlNotAllowedError as e:
        return False, str(e)
    try:
        r = httpx.get(f"{base}/api/tags", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        names = [
            str(m["name"])
            for m in data.get("models", [])
            if isinstance(m, dict) and m.get("name") is not None
        ]
        want = model.strip().lower()
        want_base = want.split(":", 1)[0]
        installed = any(
            isinstance(n, str) and (n.lower() == want or n.lower().split(":", 1)[0] == want_base)
            for n in names
        )
        if not installed:
            return False, f"Model {model!r} is not installed on Ollama at {host}"
        return True, None
    except httpx.RequestError as e:
        return False, str(e)
    except httpx.HTTPStatusError as e:
        return False, f"HTTP {e.response.status_code}"


def _gemini_ready(settings: Settings, *, model_override: str | None = None) -> tuple[bool, str | None]:
    key = (settings.gemini_api_key or "").strip()
    if not key:
        return False, "Gemini API key is not configured"
    model = (model_override or "").strip() or read_effective_gemini_model(
        settings.data_dir, settings.gemini_model
    )
    try:
        r = httpx.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": key, "pageSize": 10},
            timeout=10.0,
        )
        r.raise_for_status()
        data = r.json()
        names: list[str] = []
        for m in data.get("models", []):
            if not isinstance(m, dict):
                continue
            full = str(m.get("name", ""))
            if not full:
                continue
            methods = m.get("supportedGenerationMethods") or []
            if isinstance(methods, list) and "generateContent" not in methods:
                continue
            short = full.split("/", 1)[1] if full.startswith("models/") else full
            names.append(short)
        want = model.strip().lower()
        if names and not any(n.lower() == want for n in names):
            return False, f"Model {model!r} is not available for this API key"
        return True, None
    except httpx.RequestError as e:
        return False, str(e)
    except httpx.HTTPStatusError as e:
        return False, f"HTTP {e.response.status_code}: {(e.response.text or '')[:200]}"


def _trial_settings_for_llm_test(
    *,
    provider: Literal["gemini", "ollama"],
    gemini_api_key: str | None,
    ollama_host: str | None,
    _ollama_model: str | None,
    data_dir,
    allow_public: bool = False,
) -> Settings:
    """Build settings as if proposed values were saved (without persisting)."""
    base = merge_settings_from_store(Settings())
    updates: dict[str, Any] = {}
    if provider == "gemini":
        key_in = (gemini_api_key or "").strip()
        if key_in:
            updates["gemini_api_key"] = key_in
        elif not (base.gemini_api_key or "").strip():
            existing = read_secret(data_dir, "gemini_api_key")
            if existing:
                updates["gemini_api_key"] = existing
    else:
        host_in = (ollama_host or "").strip()
        if host_in:
            updates["ollama_host"] = validate_ollama_base_url(host_in, allow_public=allow_public)
    return base.model_copy(update=updates) if updates else base


def setup_status() -> dict[str, Any]:
    base = Settings()
    s = merge_settings_from_store(base)
    provider = read_effective_llm_provider(s.data_dir, s.llm_provider)
    llm_ready = False
    llm_error: str | None = None
    if provider == "gemini":
        llm_ready, llm_error = _gemini_ready(s)
    else:
        llm_ready, llm_error = _ollama_ready(s)

    setup = read_setup_fields(s.data_dir)
    gemini_masked = mask_secret(read_secret(s.data_dir, "gemini_api_key")) if gemini_key_configured(
        s.data_dir, base.gemini_api_key
    ) else ""

    return {
        "spotify_configured": spotify_client_configured(s.data_dir, base.spotify_client_id),
        "spotify_signed_in": _signed_in(s),
        "llm_ready": llm_ready,
        "provider": provider,
        "redirect_uri": s.spotify_redirect_uri,
        "spotify_client_id_masked": mask_secret(
            (base.spotify_client_id or "").strip()
            or str(setup.get("spotify_client_id") or "")
        )
        if spotify_client_configured(s.data_dir, base.spotify_client_id)
        else "",
        "gemini_api_key_masked": gemini_masked,
        "ollama_host": s.ollama_host.rstrip("/"),
        "ollama_model": read_effective_ollama_model(s.data_dir, s.ollama_model),
        "gemini_model": read_effective_gemini_model(s.data_dir, s.gemini_model),
        "llm_error": llm_error,
        "setup_complete": (
            spotify_client_configured(s.data_dir, base.spotify_client_id)
            and _signed_in(s)
            and llm_ready
        ),
    }


def save_spotify_app(client_id: str) -> dict[str, Any]:
    cid = validate_spotify_client_id(client_id)
    s = get_settings()
    write_setup_fields(s.data_dir, {"spotify_client_id": cid})
    logger.info("spotify app client id saved to setup.json (masked=%s)", mask_secret(cid))
    return {
        "ok": True,
        "redirect_uri": s.spotify_redirect_uri,
        "spotify_client_id_masked": mask_secret(cid),
    }


def _list_gemini_models(api_key: str) -> list[str]:
    r = httpx.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": api_key, "pageSize": 200},
        timeout=10.0,
    )
    r.raise_for_status()
    data = r.json()
    names: list[str] = []
    for m in data.get("models", []):
        if not isinstance(m, dict):
            continue
        full = str(m.get("name", ""))
        if not full:
            continue
        methods = m.get("supportedGenerationMethods") or []
        if isinstance(methods, list) and "generateContent" not in methods:
            continue
        short = full.split("/", 1)[1] if full.startswith("models/") else full
        names.append(short)
    names.sort()
    return names


def save_llm_setup(
    *,
    provider: Literal["gemini", "ollama"],
    gemini_api_key: str | None = None,
    ollama_host: str | None = None,
    ollama_model: str | None = None,
    gemini_model: str | None = None,
    test: bool = True,
    ollama_allow_public: bool = False,
    ollama_small_model_mode: str | None = None,
) -> dict[str, Any]:
    s = get_settings()
    env_base = Settings()
    gemini_model_eff = (gemini_model or "").strip() or None
    ollama_model_eff = (ollama_model or "").strip() or None

    if provider == "gemini":
        key_in = (gemini_api_key or "").strip()
        if not key_in and not gemini_key_configured(s.data_dir, env_base.gemini_api_key):
            raise ValueError("gemini_api_key is required when switching to Gemini")
    else:
        if not (ollama_host or "").strip() and not read_setup_fields(s.data_dir).get("ollama_host"):
            if not (env_base.ollama_host or "").strip():
                raise ValueError("ollama_host is required when switching to Ollama")
        if not ollama_model_eff and not read_effective_ollama_model(s.data_dir, env_base.ollama_model):
            raise ValueError("ollama_model is required when switching to Ollama")

    if test:
        trial = _trial_settings_for_llm_test(
            provider=provider,
            gemini_api_key=gemini_api_key,
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            data_dir=s.data_dir,
            allow_public=ollama_allow_public,
        )
        if provider == "gemini":
            ready, err = _gemini_ready(trial, model_override=gemini_model_eff)
        else:
            ready, err = _ollama_ready(
                trial, model_override=ollama_model_eff, allow_public=ollama_allow_public
            )
        if not ready:
            raise ValueError(err or f"{provider} validation failed")

    write_llm_provider(s.data_dir, provider)
    if ollama_small_model_mode:
        from spot_backend.llm_prefs import write_ollama_small_model_mode

        write_ollama_small_model_mode(s.data_dir, ollama_small_model_mode)

    if provider == "gemini":
        key_in = (gemini_api_key or "").strip()
        if key_in:
            backend = write_secret(s.data_dir, "gemini_api_key", key_in)
            logger.info("gemini api key stored via %s", backend)
        if gemini_model_eff:
            write_gemini_model_override(s.data_dir, gemini_model_eff)
    else:
        patch: dict[str, Any] = {}
        if ollama_host and ollama_host.strip():
            patch["ollama_host"] = validate_ollama_base_url(
                ollama_host.strip(), allow_public=ollama_allow_public
            )
        if patch:
            write_setup_fields(s.data_dir, patch)
        if ollama_model_eff:
            write_ollama_model_override(s.data_dir, ollama_model_eff)

    merged = merge_settings_from_store(Settings())
    out: dict[str, Any] = {
        "ok": True,
        "provider": provider,
        "gemini_api_key_masked": mask_secret(read_secret(s.data_dir, "gemini_api_key"))
        if provider == "gemini"
        else "",
    }

    if not test:
        return out

    if provider == "gemini":
        out["reachable"] = True
        out["error"] = None
        out["models"] = _list_gemini_models(merged.gemini_api_key)
    else:
        out["reachable"] = True
        out["error"] = None
        host = merged.ollama_host.rstrip("/")
        r = httpx.get(f"{host}/api/tags", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        out["models"] = [
            str(m["name"])
            for m in data.get("models", [])
            if isinstance(m, dict) and m.get("name") is not None
        ]
    return out


def probe_ollama(host: str, *, allow_public: bool = False) -> dict[str, Any]:
    try:
        base = validate_ollama_base_url(host, allow_public=allow_public)
    except OllamaUrlNotAllowedError as e:
        return {"reachable": False, "models": [], "error": str(e)}
    try:
        r = httpx.get(f"{base}/api/tags", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        models = [
            str(m["name"])
            for m in data.get("models", [])
            if isinstance(m, dict) and m.get("name") is not None
        ]
        return {"reachable": True, "models": models, "error": None}
    except httpx.RequestError as e:
        return {"reachable": False, "models": [], "error": str(e)}
    except httpx.HTTPStatusError as e:
        return {
            "reachable": False,
            "models": [],
            "error": f"HTTP {e.response.status_code}: {(e.response.text or '')[:200]}",
        }
