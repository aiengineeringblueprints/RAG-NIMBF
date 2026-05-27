"""RAG system adapter registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from benchmark.adapters.base import RagSystemAdapter, RagSystemOutput
from benchmark.adapters.http import HttpRagAdapter

RagAdapterFactory = Callable[[Any], RagSystemAdapter | None]

RAG_ADAPTER_REGISTRY: dict[str, RagAdapterFactory] = {}

__all__ = [
    "HttpRagAdapter",
    "RAG_ADAPTER_REGISTRY",
    "RagAdapterFactory",
    "RagSystemAdapter",
    "RagSystemOutput",
    "get_rag_adapter",
    "register_rag_adapter",
]


def register_rag_adapter(name: str, factory: RagAdapterFactory) -> None:
    """Register a RAG system adapter factory by config name."""
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("RAG adapter name must not be empty")
    RAG_ADAPTER_REGISTRY[normalized_name] = factory


register_rag_adapter("internal", lambda config: None)
register_rag_adapter("http", HttpRagAdapter.from_config)


def get_rag_adapter(config) -> RagSystemAdapter | None:
    """Return an external adapter, or None for the built-in pipeline."""
    adapter_name = str(config.rag_system_adapter).strip().lower()
    try:
        factory = RAG_ADAPTER_REGISTRY[adapter_name]
    except KeyError as exc:
        available = ", ".join(sorted(RAG_ADAPTER_REGISTRY))
        raise ValueError(
            f"Unsupported RAG_SYSTEM_ADAPTER={config.rag_system_adapter!r}. "
            f"Available: {available}"
        ) from exc
    return factory(config)
