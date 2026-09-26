"""Shared HTTP helpers to list Ollama / Gemini models (app + setup wizard)."""

from __future__ import annotations

import httpx


def ollama_model_names_from_tags_payload(data: dict) -> list[str]:
    return [
        str(m["name"])
        for m in data.get("models", [])
        if isinstance(m, dict) and m.get("name") is not None
    ]


def fetch_ollama_model_names(base_url: str, *, timeout: float = 5.0) -> list[str]:
    base = base_url.rstrip("/")
    r = httpx.get(f"{base}/api/tags", timeout=timeout)
    r.raise_for_status()
    return ollama_model_names_from_tags_payload(r.json())


def ollama_model_name_matches_installed(names: list[str], model: str) -> bool:
    want = model.strip().lower()
    want_base = want.split(":", 1)[0]
    return any(
        isinstance(n, str) and (n.lower() == want or n.lower().split(":", 1)[0] == want_base)
        for n in names
    )


def gemini_chat_model_names_from_list_payload(data: dict) -> list[str]:
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


def fetch_gemini_chat_model_names(
    api_key: str,
    *,
    page_size: int = 200,
    timeout: float = 10.0,
) -> list[str]:
    r = httpx.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": api_key, "pageSize": page_size},
        timeout=timeout,
    )
    r.raise_for_status()
    return gemini_chat_model_names_from_list_payload(r.json())
