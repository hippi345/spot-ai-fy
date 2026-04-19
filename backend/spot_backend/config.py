from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# `backend/` (parent of the `spot_backend/` package).
_BACKEND_DIR = Path(__file__).resolve().parent.parent
# Repo root (Spot-AI-fy project folder), one level above `backend/`.
_REPO_ROOT = _BACKEND_DIR.parent


class Settings(BaseSettings):
    # Merge env files so the server finds credentials even when cwd is not `backend/`.
    # Later entries override earlier ones (`backend/.env` wins over repo-root `.env`).
    model_config = SettingsConfigDict(
        env_file=(_REPO_ROOT / ".env", _BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    spotify_client_id: str = ""
    # Optional: only needed if you use the classic confidential OAuth flow instead of PKCE.
    spotify_client_secret: str = ""
    spotify_redirect_uri: str = "http://127.0.0.1:8765/callback"
    # If true, /login adds show_dialog=true so Spotify always shows the consent screen (picks up new scopes).
    spotify_show_dialog: bool = True

    api_host: str = "127.0.0.1"
    api_port: int = 8765

    frontend_origin: str = "http://localhost:5173"

    data_dir: Path = Path.home() / ".spot_ai_fy"
    token_file: Path | None = None

    # LLM: "ollama" (local) or "gemini" (Google AI Studio / Gemini API key).
    llm_provider: str = "ollama"

    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "gemma2:2b"
    # Ollama context window (tokens). Default 4096 silently truncates agent prompts that
    # include a long system prompt + chat history + tool results. Bumping to 8192 costs
    # a little extra KV cache (~1 GB RAM for 8B-class models) but keeps the full prompt.
    # Set to 0 to let Ollama use its per-model default.
    ollama_num_ctx: int = 8192
    # How long Ollama keeps the model resident after the last request. A short keep-alive
    # means every prompt can pay a ~2-minute reload penalty on CPU-only machines.
    # Accepts Ollama duration strings ("30m", "2h") or "-1" to keep forever.
    ollama_keep_alive: str = "30m"
    # Number of previous chat messages to replay when talking to Ollama. Every round of
    # the agent re-processes the full prompt, so history length directly multiplies
    # prompt-eval time on CPU. 0 = pass all messages the frontend sent (currently 40).
    ollama_history_messages: int = 0
    # Character cap on tool results injected back into Ollama conversation. Smaller caps
    # leave more context-window budget for the model's reply and shrink prompt-eval time
    # for every subsequent round. Applies only to Ollama; Gemini keeps the original cap.
    ollama_tool_result_max: int = 0
    # Per-provider override for agent_max_steps when talking to Ollama. 0 = use
    # agent_max_steps. Lower values bail out of runaway tool loops faster when a local
    # model gets confused.
    ollama_max_steps: int = 0

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    agent_max_steps: int = 16

    # Optional path to a markdown file appended to the Spot-AI-fy system prompt (Ollama + Gemini).
    agent_context_file: str = ""

    @property
    def resolved_token_path(self) -> Path:
        if self.token_file:
            return Path(self.token_file)
        return self.data_dir / "tokens.json"

    @property
    def resolved_device_path(self) -> Path:
        return self.data_dir / "device.json"


def get_settings() -> Settings:
    return Settings()
