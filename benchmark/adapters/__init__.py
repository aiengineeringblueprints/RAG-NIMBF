"""RAG system adapter registry."""

from benchmark.adapters.base import RagSystemAdapter, RagSystemOutput
from benchmark.adapters.http import HttpRagAdapter

__all__ = ["HttpRagAdapter", "RagSystemAdapter", "RagSystemOutput", "get_rag_adapter"]


def get_rag_adapter(config) -> RagSystemAdapter | None:
    """Return an external adapter, or None for the built-in pipeline."""
    if config.rag_system_adapter == "internal":
        return None
    if config.rag_system_adapter == "http":
        return HttpRagAdapter.from_config(config)
    raise ValueError(f"Unsupported RAG_SYSTEM_ADAPTER={config.rag_system_adapter!r}")
