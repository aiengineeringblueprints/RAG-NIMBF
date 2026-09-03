"""RAG system adapter registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from benchmark.adapters.base import (
    AdapterCapabilities,
    AdapterGenerationResult,
    ManagedRagSystemAdapter,
    PreparedTarget,
    RagSystemAdapter,
    RagSystemOutput,
    RetrievedChunk,
    RetrievalResult,
)
from benchmark.adapters.baselines import (
    NoRetrievalAdapter,
    OracleRetrievalAdapter,
    RandomRetrievalAdapter,
)
from benchmark.adapters.components import ComponentBundle, build_components
from benchmark.adapters.http import HttpRagAdapter
from benchmark.adapters.mcp import McpRagAdapter
from benchmark.adapters.ragflow import RagflowAdapter
from benchmark.adapters.agentic import AgenticRagAdapter
from benchmark.adapters.lifecycle import (
    UnsupportedAdapterCapability,
    cleanup_adapter,
    generate_adapter,
    get_adapter_capabilities,
    prepare_adapter,
    require_adapter_capabilities,
    retrieve_adapter,
)

RagAdapterFactory = Callable[
    [Any], RagSystemAdapter | ManagedRagSystemAdapter | None
]

RAG_ADAPTER_REGISTRY: dict[str, RagAdapterFactory] = {}

__all__ = [
    "ComponentBundle",
    "AdapterCapabilities",
    "AdapterGenerationResult",
    "HttpRagAdapter",
    "McpRagAdapter",
    "NoRetrievalAdapter",
    "OracleRetrievalAdapter",
    "RandomRetrievalAdapter",
    "RagflowAdapter",
    "ManagedRagSystemAdapter",
    "PreparedTarget",
    "RAG_ADAPTER_REGISTRY",
    "RagAdapterFactory",
    "RagSystemAdapter",
    "RagSystemOutput",
    "RetrievedChunk",
    "RetrievalResult",
    "UnsupportedAdapterCapability",
    "build_components",
    "cleanup_adapter",
    "generate_adapter",
    "get_rag_adapter",
    "get_adapter_capabilities",
    "prepare_adapter",
    "register_rag_adapter",
    "require_adapter_capabilities",
    "retrieve_adapter",
]


def register_rag_adapter(name: str, factory: RagAdapterFactory) -> None:
    """Register a RAG system adapter factory by config name."""
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("RAG adapter name must not be empty")
    RAG_ADAPTER_REGISTRY[normalized_name] = factory


register_rag_adapter("internal", lambda config: None)
register_rag_adapter("http", HttpRagAdapter.from_config)
register_rag_adapter("mcp", McpRagAdapter.from_config)
register_rag_adapter("ragflow", RagflowAdapter.from_config)
register_rag_adapter("optimaiserag", RagflowAdapter.from_config)
register_rag_adapter("no_retrieval", NoRetrievalAdapter.from_config)
register_rag_adapter("random_retrieval", RandomRetrievalAdapter.from_config)
register_rag_adapter("agentic", AgenticRagAdapter.from_config)
register_rag_adapter("oracle_retrieval", OracleRetrievalAdapter.from_config)


def get_rag_adapter(
    config: Any,
) -> RagSystemAdapter | ManagedRagSystemAdapter | None:
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
