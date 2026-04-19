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
