"""Embedding model factory — routes to Ollama or HuggingFace backends."""
from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Ollama's llama-server runner crashes ("connection reset by peer" on its
# internal /tokenize port) when asked to embed an empty string, and oversized
# inputs can trip the tokenizer inside the runner. Guard against both, and
# retry transient runner restarts with bounded backoff.
_EMPTY_TEXT_PLACEHOLDER = " "
# Conservative character cap well below nomic-embed-text's 2048-token
# context (roughly 4 chars/token for English prose).
_MAX_TEXT_CHARS = 6000
_EMBED_RETRIES = 5
_RETRY_BASE_DELAY = 2.0

T = TypeVar("T")


def _ollama_embeddings_class() -> type:
    """Import and harden OllamaEmbeddings lazily (mirrors prior lazy imports)."""
    from langchain_ollama import OllamaEmbeddings

    base_cls = OllamaEmbeddings

    class ResilientOllamaEmbeddings(OllamaEmbeddings):
        """Ollama embeddings hardened against runner crashes.

        Sanitizes inputs (empty/whitespace-only strings crash Ollama's
        embedding runner; oversized texts can break its tokenizer) and
        retries transient connection errors with exponential backoff.
        Subclassing keeps ``isinstance`` checks elsewhere intact.
        """

        @staticmethod
        def _sanitize(text: str) -> str:
            cleaned = text if text and text.strip() else _EMPTY_TEXT_PLACEHOLDER
            return cleaned[:_MAX_TEXT_CHARS]

        def _embed_with_retry(self, call: Callable[[], T]) -> T:
            import httpx
            import ollama

            last_exc: Exception | None = None
            for attempt in range(_EMBED_RETRIES):
                try:
                    return call()
                except (ollama.ResponseError, httpx.HTTPError) as exc:
                    last_exc = exc
                    delay = _RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        "Ollama embedding call failed (attempt %d/%d): %s — "
                        "retrying in %.1fs",
                        attempt + 1,
                        _EMBED_RETRIES,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            cleaned = [self._sanitize(t) for t in texts]
            return self._embed_with_retry(
                lambda: base_cls.embed_documents(self, cleaned)
            )

        def embed_query(self, text: str) -> list[float]:
            return self._embed_with_retry(
                lambda: base_cls.embed_query(self, self._sanitize(text))
            )

    return ResilientOllamaEmbeddings


def get_embedding_model(
    model_name: str,
    base_url: str,
    api_key: str | None = None,
    *,
    provider: str = "ollama",
) -> Embeddings:
    """Create an embedding model for the given provider.

    Parameters
    ----------
    model_name:
        Model identifier understood by the backend.
    base_url:
        Server base URL (used by Ollama; ignored by HuggingFace).
    api_key:
        Optional Bearer token (used by Ollama; ignored by HuggingFace).
    provider:
        ``"ollama"`` or ``"huggingface"``.
    """
    if provider == "ollama":
        kwargs: dict = {"model": model_name, "base_url": base_url}
        if api_key:
            kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {api_key}"}}
        return _ollama_embeddings_class()(**kwargs)

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model_name)

    raise ValueError(
        f"Unknown embedding provider '{provider}'. Supported: 'ollama', 'huggingface'."
    )
