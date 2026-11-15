"""Embedding model factory — routes to Ollama or HuggingFace backends."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from functools import cache
from typing import Awaitable, Callable, TypeVar

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Ollama's llama-server runner crashes ("connection reset by peer" on its
# internal /tokenize port) when asked to embed an empty string, and oversized
# inputs can trip the tokenizer inside the runner. Guard against both, and
# retry transient runner restarts with bounded backoff.
#
# The placeholder must be non-whitespace: whitespace-only input is exactly
# what crashes the embedding runner.
_EMPTY_TEXT_PLACEHOLDER = "empty"
# Conservative character cap well below nomic-embed-text's 2048-token
# context (roughly 4 chars/token for English prose). Override via the
# EMBEDDING_MAX_TEXT_CHARS environment variable for long-context models
# (e.g. bge-m3).
_DEFAULT_MAX_TEXT_CHARS = 6000
_EMBED_RETRIES = 5
_RETRY_BASE_DELAY = 2.0
# langchain_ollama sends the whole input list as ONE /api/embed POST; large
# batches (thousands of chunks) crash Ollama's llama-server runner during
# tokenization ("connection reset by peer" on its internal /tokenize port).
# Single-text requests never crash, so keep batches small.
_EMBED_BATCH_SIZE = 64

T = TypeVar("T")


def _max_text_chars() -> int:
    """Return the character cap for Ollama embedding inputs."""
    raw = os.getenv("EMBEDDING_MAX_TEXT_CHARS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning(
                "Invalid EMBEDDING_MAX_TEXT_CHARS=%r — falling back to %d",
                raw,
                _DEFAULT_MAX_TEXT_CHARS,
            )
    return _DEFAULT_MAX_TEXT_CHARS


# Log the truncation warning at most once per process (it can fire for
# thousands of chunks during a single indexing run).
_truncation_warned = False


@cache
def _ollama_embeddings_class() -> type:
    """Build (once) the hardened OllamaEmbeddings subclass.

    Cached so repeated ``get_embedding_model`` calls reuse the same class
    instead of rebuilding the Pydantic model class and producing unpicklable
    ``<locals>`` qualnames.
    """
    from langchain_ollama import OllamaEmbeddings

    base_cls = OllamaEmbeddings

    class ResilientOllamaEmbeddings(OllamaEmbeddings):
        """Ollama embeddings hardened against runner crashes.

        Sanitizes inputs (empty/whitespace-only strings crash Ollama's
        embedding runner; oversized texts can break its tokenizer), keeps
        request batches small and retries transient connection errors with
        exponential backoff — for both the sync and async code paths (RAGAS
        goes through ``aembed_*`` via ``LangchainEmbeddingsWrapper``).
        Subclassing keeps ``isinstance`` checks elsewhere intact.
        """

        @staticmethod
        def _sanitize(text: str) -> str:
            global _truncation_warned
            cleaned = (
                text if text and text.strip() else _EMPTY_TEXT_PLACEHOLDER
            )
            limit = _max_text_chars()
            if len(cleaned) > limit:
                if not _truncation_warned:
                    logger.warning(
                        "Truncating embedding input longer than %d chars "
                        "(set EMBEDDING_MAX_TEXT_CHARS to raise the cap)",
                        limit,
                    )
                    _truncation_warned = True
                cleaned = cleaned[:limit]
            return cleaned

        @staticmethod
        def _is_transient(exc: BaseException) -> bool:
            """True for errors worth retrying (5xx / connection failures).

            The ollama client maps ``httpx.ConnectError`` to the built-in
            ``ConnectionError``, so a refused connection during a server
            restart is retried too. Client errors (400/404 — e.g. a typo in
            the model name) fail fast instead of burning retry attempts.
            """
            import httpx
            import ollama

            if isinstance(exc, ConnectionError):  # httpx.ConnectError mapped
                return True
            if isinstance(exc, httpx.HTTPError):
                return True
            if isinstance(exc, ollama.ResponseError):
                return 500 <= (exc.status_code or 0) < 600
            return False

        def _call_with_retry(self, call: Callable[[], T]) -> T:
            """Retry a single (batched) sync call with exponential backoff."""
            last_exc: BaseException | None = None
            for attempt in range(_EMBED_RETRIES):
                try:
                    return call()
                except Exception as exc:  # noqa: BLE001 — classified below
                    if not self._is_transient(exc):
                        raise
                    last_exc = exc
                    if attempt < _EMBED_RETRIES - 1:
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

        async def _acall_with_retry(self, call: Callable[[], Awaitable[T]]) -> T:
            """Retry a single (batched) async call with exponential backoff."""
            last_exc: BaseException | None = None
            for attempt in range(_EMBED_RETRIES):
                try:
                    return await call()
                except Exception as exc:  # noqa: BLE001 — classified below
                    if not self._is_transient(exc):
                        raise
                    last_exc = exc
                    if attempt < _EMBED_RETRIES - 1:
                        delay = _RETRY_BASE_DELAY * (2**attempt)
                        logger.warning(
                            "Ollama embedding call failed (attempt %d/%d): %s — "
                            "retrying in %.1fs",
                            attempt + 1,
                            _EMBED_RETRIES,
                            exc,
                            delay,
                        )
                        await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            # Retry per batch: re-embedding every already-succeeded batch on
            # one failure makes large corpora effectively never finish.
            return [
                vec
                for start in range(0, len(texts), _EMBED_BATCH_SIZE)
                for vec in self._call_with_retry(
                    lambda batch=texts[start : start + _EMBED_BATCH_SIZE]: [
                        vec
                        for vec in base_cls.embed_documents(
                            self, [self._sanitize(t) for t in batch]
                        )
                    ]
                )
            ]

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            """Async path used by RAGAS (LangchainEmbeddingsWrapper)."""
            vectors: list[list[float]] = []
            for start in range(0, len(texts), _EMBED_BATCH_SIZE):
                batch = [self._sanitize(t) for t in texts[start : start + _EMBED_BATCH_SIZE]]
                vectors.extend(
                    await self._acall_with_retry(
                        lambda batch=batch: base_cls.aembed_documents(self, batch)
                    )
                )
            return vectors

        def embed_query(self, text: str) -> list[float]:
            return self._call_with_retry(
                lambda: base_cls.embed_query(self, self._sanitize(text))
            )

        async def aembed_query(self, text: str) -> list[float]:
            """Async path used by RAGAS (LangchainEmbeddingsWrapper)."""
            return await self._acall_with_retry(
                lambda: base_cls.aembed_query(self, self._sanitize(text))
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
