"""Embedding model factory — routes to Ollama or HuggingFace backends."""
from __future__ import annotations

from langchain_core.embeddings import Embeddings


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
        from langchain_ollama import OllamaEmbeddings

        kwargs: dict = {"model": model_name, "base_url": base_url}
        if api_key:
            kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {api_key}"}}
        return OllamaEmbeddings(**kwargs)

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model_name)

    raise ValueError(
        f"Unknown embedding provider '{provider}'. Supported: 'ollama', 'huggingface'."
    )
