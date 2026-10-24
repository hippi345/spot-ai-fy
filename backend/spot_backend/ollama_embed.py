"""Local Ollama embedding client (nomic-embed-text)."""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

from spot_backend.config import Settings

logger = logging.getLogger(__name__)

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    @property
    def model_name(self) -> str: ...
    @property
    def dimensions(self) -> int: ...


class OllamaEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.ollama_embed_model
        self._dims: int | None = None
        self._unavailable_logged = False

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def dimensions(self) -> int:
        return self._dims or 768

    def _post_embed(self, inputs: list[str]) -> list[list[float]]:
        url = f"{self.settings.ollama_host.rstrip('/')}/api/embed"
        body = {"model": self.model, "input": inputs}
        with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            r = client.post(url, json=body)
            r.raise_for_status()
            data = r.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError("Ollama embed response missing embeddings")
        out: list[list[float]] = []
        for row in embeddings:
            if not isinstance(row, list):
                raise RuntimeError("invalid embedding row")
            out.append([float(x) for x in row])
        if out and self._dims is None:
            self._dims = len(out[0])
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [DOC_PREFIX + (t or "") for t in texts]
        batches: list[list[float]] = []
        batch_size = 64
        for i in range(0, len(prefixed), batch_size):
            batches.extend(self._post_embed(prefixed[i : i + batch_size]))
        return batches

    def embed_query(self, text: str) -> list[float]:
        return self._post_embed([QUERY_PREFIX + (text or "")])[0]

    def ping(self) -> bool:
        try:
            self._post_embed(["ping"])
            return True
        except Exception:
            if not self._unavailable_logged:
                logger.warning("Ollama embedder unavailable (model %s)", self.model)
                self._unavailable_logged = True
            return False


class HashingEmbedder:
    """Deterministic fake embedder for unit tests (no network)."""

    def __init__(self, dims: int = 32) -> None:
        self._dims = dims

    @property
    def model_name(self) -> str:
        return "hashing-test"

    @property
    def dimensions(self) -> int:
        return self._dims

    def _vec(self, text: str) -> list[float]:
        import hashlib

        h = hashlib.sha256(text.encode("utf-8")).digest()
        out: list[float] = []
        while len(out) < self._dims:
            for i in range(0, len(h), 4):
                chunk = h[i : i + 4]
                if len(chunk) < 4:
                    break
                out.append((int.from_bytes(chunk, "big") % 1000) / 1000.0)
                if len(out) >= self._dims:
                    break
            h = hashlib.sha256(h).digest()
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(DOC_PREFIX + t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(QUERY_PREFIX + text)
