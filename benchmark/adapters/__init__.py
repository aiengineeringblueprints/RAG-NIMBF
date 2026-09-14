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
    RetrievalResult,
    RetrievedChunk,
)
from benchmark.adapters.components import (
    ComponentBundle,
    build_components,
    inject_components,
)
from benchmark.adapters.http import HttpRagAdapter
from benchmark.adapters.internal import InternalRagAdapter
from benchmark.adapters.lifecycle import (
    UnsupportedAdapterCapability,
    adapter_aggregate_metrics,
    adapter_stage_timings,
    cleanup_adapter,
    generate_adapter,
    get_adapter_capabilities,
    prepare_adapter,
    require_adapter_capabilities,
    retrieve_adapter,
)
from benchmark.adapters.mcp import McpRagAdapter
from benchmark.adapters.ragflow import RagflowAdapter

RagAdapterFactory = Callable[
    [Any], RagSystemAdapter | ManagedRagSystemAdapter | None
]

RAG_ADAPTER_REGISTRY: dict[str, RagAdapterFactory] = {}

__all__ = [
    "RAG_ADAPTER_REGISTRY",
    "AdapterCapabilities",
    "AdapterGenerationResult",
    "ComponentBundle",
    "HttpRagAdapter",
    "InternalRagAdapter",
    "ManagedRagSystemAdapter",
    "McpRagAdapter",
    "PreparedTarget",
    "RagAdapterFactory",
    "RagSystemAdapter",
    "RagSystemOutput",
    "RagflowAdapter",
    "RetrievalResult",
    "RetrievedChunk",
    "UnsupportedAdapterCapability",
    "adapter_aggregate_metrics",
    "adapter_stage_timings",
    "build_components",
    "cleanup_adapter",
    "generate_adapter",
    "get_adapter_capabilities",
    "get_rag_adapter",
    "inject_components",
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


register_rag_adapter("internal", InternalRagAdapter.from_config)
register_rag_adapter("http", HttpRagAdapter.from_config)
register_rag_adapter("mcp", McpRagAdapter.from_config)
register_rag_adapter("ragflow", RagflowAdapter.from_config)
register_rag_adapter("optimaiserag", RagflowAdapter.from_config)


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
