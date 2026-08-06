"""Pluggable vector-store adapters.

This package implements the RAGPerf-style ``DBInstance`` adapter layer
(arXiv:2603.10765v1, Figure 4 / Sec. 3.3.2) so the benchmark can drive
Chroma, Qdrant, Milvus, and LanceDB through a single interface.

Public surface:

- :class:`VectorStoreAdapter` -- ABC implemented by every backend.
- :func:`get_vector_store` -- factory that resolves an adapter by name.
- :data:`SUPPORTED_BACKENDS` -- tuple of registered backend names.

The legacy ``benchmark.retrieval`` factory (``build_vector_store``,
``ChromaVectorStoreBackend``, ``LanceDBVectorStoreBackend``, ...) remains
unchanged for backward compatibility; the new layer is opt-in.
"""

from __future__ import annotations

from benchmark.vectorstore.base import (
    MetricType,
    SearchHit,
    VectorStoreAdapter,
    VectorStoreConfig,
)
from benchmark.vectorstore.factory import (
    SUPPORTED_BACKENDS,
    get_vector_store,
    register_adapter,
)

__all__ = [
    "MetricType",
    "SUPPORTED_BACKENDS",
    "SearchHit",
    "VectorStoreAdapter",
    "VectorStoreConfig",
    "get_vector_store",
    "register_adapter",
]
