"""Default configuration values."""

from spot_backend.config import Settings


def test_ollama_defaults() -> None:
    settings = Settings()
    assert settings.ollama_model == "qwen2.5:3b"
    assert settings.ollama_num_ctx == 16384
