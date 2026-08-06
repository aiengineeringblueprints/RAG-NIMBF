"""Factory + registry for vector-store adapters.

Usage::

    from benchmark.vectorstore import get_vector_store, VectorStoreConfig

    cfg = VectorStoreConfig(backend="qdrant", path=".qdrant", dimension=768)
    store = get_vector_store(cfg)
    store.build_index("hnsw", "cosine", "my_collection")
    store.insert(vectors, chunks, "my_collection")
    hits = store.search(query_vectors, "my_collection", top_k=5)

A YAML ``vector_store:`` block maps directly onto ``VectorStoreConfig``::

    vector_store:
      backend: qdrant          # chroma | qdrant | milvus | lancedb
      path: ./qdrant_data
      collection: default
      index_type: hnsw         # hnsw | ivf | flat (per-backend support varies)
      metric: cosine           # cosine | l2 | ip
      host: null               # optional, server backends only
      port: null
      dimension: 768           # required by qdrant/milvus on build_index

If ``backend`` is omitted the factory defaults to ``"chroma"`` and reproduces
the pre-refactor behaviour.
"""

from __future__ import annotations

from typing import Any, Callable

from benchmark.vectorstore.base import VectorStoreAdapter, VectorStoreConfig


# Adapter factories are registered lazily so importing this module does not
# require the optional dependencies (qdrant-client, pymilvus). Each factory
# receives the resolved ``VectorStoreConfig``.
_ADAPTER_FACTORIES: dict[str, Callable[[VectorStoreConfig], VectorStoreAdapter]] = {}


def register_adapter(
    name: str, factory: Callable[[VectorStoreConfig], VectorStoreAdapter]
) -> None:
    """Register a new adapter factory under ``name`` (case-insensitive)."""
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("adapter name must not be empty")
    _ADAPTER_FACTORIES[normalized] = factory


def SUPPORTED_BACKENDS() -> tuple[str, ...]:
    """Names of registered adapters."""
    return tuple(sorted(_ADAPTER_FACTORIES))


# ---- Built-in registrations -------------------------------------------------
# These factories do the lazy imports; importing this module is always cheap.


def _build_chroma(cfg: VectorStoreConfig) -> VectorStoreAdapter:
    from benchmark.vectorstore.chroma_store import ChromaVectorStore

    return ChromaVectorStore(db_type="chroma", db_path=cfg.path)


def _build_qdrant(cfg: VectorStoreConfig) -> VectorStoreAdapter:
    from benchmark.vectorstore.qdrant_store import QdrantVectorStore

    return QdrantVectorStore(
        db_type="qdrant",
        db_path=cfg.path,
        host=cfg.host,
        port=cfg.port,
        api_key=cfg.api_key,
        dimension=cfg.dimension,
    )


def _build_milvus(cfg: VectorStoreConfig) -> VectorStoreAdapter:
    from benchmark.vectorstore.milvus_store import MilvusVectorStore

    return MilvusVectorStore(
        db_type="milvus",
        db_path=cfg.path,
        host=cfg.host,
        port=cfg.port,
        api_key=cfg.api_key,
        dimension=cfg.dimension,
    )


def _build_lancedb(cfg: VectorStoreConfig) -> VectorStoreAdapter:
    from benchmark.vectorstore.lancedb_store import LanceDBVectorStore

    return LanceDBVectorStore(db_type="lancedb", db_path=cfg.path)


register_adapter("chroma", _build_chroma)
register_adapter("qdrant", _build_qdrant)
register_adapter("milvus", _build_milvus)
register_adapter("lancedb", _build_lancedb)


def get_vector_store(
    config: VectorStoreConfig | dict[str, Any] | None = None,
) -> VectorStoreAdapter:
    """Return an adapter for the given config.

    ``config`` may be a :class:`VectorStoreConfig`, a YAML-style dict, or
    ``None`` (defaults to Chroma). Raises ``ValueError`` for unknown
    backends and ``ImportError`` if the backend's optional dependency is
    missing.
    """

    if config is None:
        config = VectorStoreConfig()
    elif isinstance(config, dict):
        config = VectorStoreConfig.from_dict(config)
    elif not isinstance(config, VectorStoreConfig):
        raise TypeError(
            "get_vector_store expects a VectorStoreConfig, dict, or None; "
            f"got {type(config).__name__}"
        )

    factory = _ADAPTER_FACTORIES.get(config.backend.lower())
    if factory is None:
        raise ValueError(
            f"Unknown vector_store backend {config.backend!r}. "
            f"Registered: {', '.join(SUPPORTED_BACKENDS())}"
        )
    return factory(config)
