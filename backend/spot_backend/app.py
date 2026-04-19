from __future__ import annotations

import urllib.parse
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from spot_backend.agent import iter_chat_events, run_chat_turn
from spot_backend.chat_sse import sse_data
from spot_backend.config import get_settings
from spot_backend.llm_prefs import (
    clear_llm_provider_override,
    gemini_model_override_active,
    ollama_model_override_active,
    prefs_path_exists,
    read_effective_gemini_model,
    read_effective_llm_provider,
    read_effective_ollama_model,
    write_gemini_model_override,
    write_llm_provider,
    write_ollama_model_override,
)
from spot_backend.pkce import new_pkce_params
from spot_backend.spotify_client import DEFAULT_SCOPES, SpotifyAuthError, SpotifyClient
from spot_backend.token_store import DeviceSelection, load_device, load_tokens, save_device

app = FastAPI(title="Spot-AI-fy API")

# state -> code_verifier for Spotify PKCE (in-memory; cleared after callback).
_pkce_pending: dict[str, str] = {}

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_settings.frontend_origin, "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DeviceBody(BaseModel):
    device_id: str = Field(..., min_length=1)


class ChatHistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=48_000)


class ChatBody(BaseModel):
    message: str = Field(..., min_length=1, max_length=48_000)
    history: list[ChatHistoryTurn] | None = Field(default=None, max_length=48)


def _dump_chat_history(body: ChatBody) -> list[dict[str, str]] | None:
    if not body.history:
        return None
    return [{"role": t.role, "content": t.content} for t in body.history]


class LlmProviderBody(BaseModel):
    provider: Literal["ollama", "gemini"]


class OllamaModelBody(BaseModel):
    model: str = Field(..., min_length=1, max_length=200)


class GeminiModelBody(BaseModel):
    model: str = Field(..., min_length=1, max_length=200)


@app.get("/login")
def login() -> RedirectResponse:
    s = get_settings()
    if not s.spotify_client_id.strip():
        raise HTTPException(
            status_code=500,
            detail=(
                "SPOTIFY_CLIENT_ID is not set. Put it in backend/.env (gitignored) — "
                "see backend/.env.example — or export it, then restart the API. "
                "Do not commit credentials to GitHub."
            ),
        )
    verifier, challenge, state = new_pkce_params()
    _pkce_pending[state] = verifier
    auth_params: dict[str, str] = {
        "client_id": s.spotify_client_id,
        "response_type": "code",
        "redirect_uri": s.spotify_redirect_uri,
        "scope": DEFAULT_SCOPES,
        "state": state,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
    }
    if s.spotify_show_dialog:
        auth_params["show_dialog"] = "true"
    q = urllib.parse.urlencode(auth_params)
    return RedirectResponse(url=f"https://accounts.spotify.com/authorize?{q}")


@app.get("/callback")
def callback(
    code: str | None = None,
    error: str | None = None,
    state: str | None = None,
) -> RedirectResponse:
    s = get_settings()
    front = s.frontend_origin.rstrip("/")
    if error:
        return RedirectResponse(url=f"{front}/?spotify=error&reason={urllib.parse.quote(error)}")
    if not code:
        return RedirectResponse(url=f"{front}/?spotify=error&reason=missing_code")
    if not state:
        return RedirectResponse(url=f"{front}/?spotify=error&reason=missing_state")
    verifier = _pkce_pending.pop(state, None)
    if not verifier:
        return RedirectResponse(url=f"{front}/?spotify=error&reason=invalid_or_expired_state")
    client = SpotifyClient(settings=s)
    try:
        client.exchange_authorization_code(code, redirect_uri=s.spotify_redirect_uri, code_verifier=verifier)
    finally:
        client.close()
    return RedirectResponse(url=f"{front}/?spotify=connected")


@app.post("/logout")
def logout() -> dict[str, str]:
    s = get_settings()
    for p in (s.resolved_token_path, s.resolved_device_path):
        if p.is_file():
            p.unlink()
    return {"ok": "true"}


def _playlist_modify_scopes_ok(scope: str) -> bool | None:
    """True if at least one playlist-modify-* scope is present; False if known but neither; None if no scope string."""
    s = (scope or "").strip()
    if not s:
        return None
    parts = set(s.replace(",", " ").split())
    if "playlist-modify-public" in parts or "playlist-modify-private" in parts:
        return True
    return False


@app.get("/api/session")
def session() -> dict[str, Any]:
    s = get_settings()
    bundle = load_tokens(s.resolved_token_path)
    device = load_device(s.resolved_device_path)
    signed = bool(bundle and bundle.access_token)
    granted = (bundle.scope or "").strip() if bundle else ""
    return {
        "signed_in": signed,
        "device_id": device.device_id if device else None,
        "spotify_granted_scopes": granted if granted else None,
        "spotify_playlist_write_ok": _playlist_modify_scopes_ok(granted) if signed else None,
    }


@app.get("/api/devices")
def devices() -> Any:
    s = get_settings()
    client = SpotifyClient(settings=s)
    try:
        return client.api_get("/me/player/devices")
    except SpotifyAuthError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    finally:
        client.close()


@app.post("/api/device")
def set_device(body: DeviceBody) -> dict[str, str]:
    s = get_settings()
    save_device(s.resolved_device_path, DeviceSelection(device_id=body.device_id))
    return {"ok": "true", "device_id": body.device_id}


@app.post("/api/chat")
def chat(body: ChatBody) -> dict[str, str]:
    s = get_settings()
    active = read_effective_llm_provider(s.data_dir, s.llm_provider)
    ollama_model = read_effective_ollama_model(s.data_dir, s.ollama_model)
    gemini_model = read_effective_gemini_model(s.data_dir, s.gemini_model)
    hist = _dump_chat_history(body)
    try:
        text = run_chat_turn(body.message, s, history=hist)
    except httpx.HTTPStatusError as e:
        snippet = (e.response.text or "")[:400]
        if active == "gemini":
            code = e.response.status_code
            if code == 429:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Gemini hit its rate limit / quota for {gemini_model}. On the free tier this typically "
                        "resets daily. Try a lighter model from the Settings dropdown (e.g. gemini-2.5-flash-lite), "
                        "switch to Ollama in Settings, or enable billing on your Google AI key if you need more headroom."
                    ),
                ) from e
            if code == 503:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Gemini is overloaded right now — Google has been returning 'service unavailable' for "
                        f"{gemini_model}. Please try again in a minute, pick a lighter model from the Settings "
                        "dropdown (e.g. gemini-2.5-flash-lite or gemini-1.5-flash), or switch to Ollama in Settings."
                    ),
                ) from e
            if code == 404:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"The Gemini model {gemini_model} isn't available for your API key right now (it may have "
                        "been retired, or your key isn't enabled for it). Pick a different model from the Settings dropdown."
                    ),
                ) from e
            if code in (401, 403):
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Google rejected the Gemini API key (it's missing, expired, or doesn't have access to this "
                        "model). Please double-check GEMINI_API_KEY in backend/.env and restart the backend, or switch "
                        "to Ollama in Settings if you don't have a working key handy."
                    ),
                ) from e
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Gemini ran into an unexpected problem on that request ({gemini_model}). Please try again in "
                    "a moment, pick a different model from the Settings dropdown, or switch to Ollama in Settings."
                ),
            ) from e
        raise HTTPException(
            status_code=502,
            detail=(
                f"Ollama returned an error for model {ollama_model!r} at {s.ollama_host}. "
                f"Try: ollama pull {ollama_model}. Response: {snippet or str(e)}"
            ),
        ) from e
    except httpx.RequestError as e:
        if active == "gemini":
            raise HTTPException(
                status_code=503,
                detail=(
                    "I couldn't reach Gemini just now (network error talking to Google's API). "
                    "Please try again in a minute, check GEMINI_API_KEY in backend/.env, or "
                    "switch to Ollama in Settings if it keeps happening."
                ),
            ) from e
        raise HTTPException(
            status_code=503,
            detail=(
                f"No connection to Ollama at {s.ollama_host} ({e}). "
                "Install and start Ollama from https://ollama.com (the tray app must be running), "
                f"then run: ollama pull {ollama_model}. "
                "If Ollama listens elsewhere, set OLLAMA_HOST in backend/.env."
            ),
        ) from e
    return {"reply": text}


@app.post("/api/chat/stream")
def chat_stream(body: ChatBody) -> StreamingResponse:
    """SSE stream of Spot-AI-fy agent progress (Ollama token deltas, tool steps, Gemini status)."""
    s = get_settings()
    hist = _dump_chat_history(body)

    def event_gen():
        try:
            for ev in iter_chat_events(body.message, s, history=hist):
                yield sse_data(ev)
        finally:
            yield sse_data({"type": "done"})

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)


@app.post("/api/llm/provider")
def set_llm_provider(body: LlmProviderBody) -> dict[str, str]:
    s = get_settings()
    write_llm_provider(s.data_dir, body.provider)
    return {"ok": "true", "provider": body.provider}


@app.delete("/api/llm/provider")
def reset_llm_provider() -> dict[str, str]:
    s = get_settings()
    clear_llm_provider_override(s.data_dir)
    return {"ok": "true"}


@app.post("/api/llm/ollama-model")
def set_ollama_model(body: OllamaModelBody) -> dict[str, str]:
    s = get_settings()
    write_ollama_model_override(s.data_dir, body.model)
    return {"ok": "true", "model": body.model.strip()}


@app.delete("/api/llm/ollama-model")
def reset_ollama_model() -> dict[str, str]:
    s = get_settings()
    write_ollama_model_override(s.data_dir, None)
    return {"ok": "true"}


@app.post("/api/llm/gemini-model")
def set_gemini_model(body: GeminiModelBody) -> dict[str, str]:
    s = get_settings()
    write_gemini_model_override(s.data_dir, body.model)
    return {"ok": "true", "model": body.model.strip()}


@app.delete("/api/llm/gemini-model")
def reset_gemini_model() -> dict[str, str]:
    s = get_settings()
    write_gemini_model_override(s.data_dir, None)
    return {"ok": "true"}


@app.get("/api/llm")
def llm_status() -> dict[str, Any]:
    """Reachability for the active LLM provider (Ollama or Gemini)."""
    s = get_settings()
    env_provider = (s.llm_provider or "ollama").strip().lower()
    active = read_effective_llm_provider(s.data_dir, s.llm_provider)
    out: dict[str, Any] = {
        "provider": active,
        "env_provider": env_provider,
        "ui_override": prefs_path_exists(s.data_dir),
        "reachable": False,
        "error": None,
    }

    if active == "gemini":
        effective_gemini = read_effective_gemini_model(s.data_dir, s.gemini_model)
        out["env_gemini_model"] = s.gemini_model
        out["configured_model"] = effective_gemini
        out["gemini_model_ui_override"] = gemini_model_override_active(s.data_dir)
        key = (s.gemini_api_key or "").strip()
        if not key:
            out["error"] = "GEMINI_API_KEY is not set in backend/.env"
            return out
        try:
            # pageSize=200 so the UI can list every model the key can access.
            r = httpx.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": key, "pageSize": 200},
                timeout=10.0,
            )
            r.raise_for_status()
            data = r.json()
            # Keep only text-generation capable models so the dropdown is useful for chat.
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
            out["reachable"] = True
            out["models"] = names
            want = effective_gemini.strip().lower()
            out["model_installed"] = any(
                isinstance(n, str) and n.lower() == want for n in names
            )
        except httpx.RequestError as e:
            out["error"] = str(e)
        except httpx.HTTPStatusError as e:
            out["error"] = f"HTTP {e.response.status_code}: {(e.response.text or '')[:200]}"
        return out

    base = s.ollama_host.rstrip("/")
    effective_ollama = read_effective_ollama_model(s.data_dir, s.ollama_model)
    out["configured_host"] = base
    out["env_ollama_model"] = s.ollama_model
    out["configured_model"] = effective_ollama
    out["ollama_model_ui_override"] = ollama_model_override_active(s.data_dir)
    try:
        r = httpx.get(f"{base}/api/tags", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        names = [
            str(m["name"])
            for m in data.get("models", [])
            if isinstance(m, dict) and m.get("name") is not None
        ]
        want = effective_ollama.strip().lower()
        want_base = want.split(":", 1)[0]
        out["reachable"] = True
        out["models"] = names
        out["model_installed"] = any(
            isinstance(n, str) and (n.lower() == want or n.lower().split(":", 1)[0] == want_base)
            for n in names
        )
    except httpx.RequestError as e:
        out["error"] = str(e)
    except httpx.HTTPStatusError as e:
        out["error"] = f"HTTP {e.response.status_code}: {(e.response.text or '')[:200]}"
    return out


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
