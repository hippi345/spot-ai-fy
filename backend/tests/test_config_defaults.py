"""Default configuration values."""

from spot_backend.config import Settings


def test_ollama_defaults() -> None:
    settings = Settings()
    assert settings.ollama_model == "qwen3:4b-instruct"
    assert settings.ollama_num_ctx == 16384
