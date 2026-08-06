"""Abstract base class for vector-store adapters.

The interface mirrors RAGPerf's ``DBInstance`` (Figure 4 of arXiv:2603.10765v1):

- ``__init__(db_type, db_path, client)``
- ``build_index(index_type, metric_type, collection_name)``
- ``insert(vectors, chunks, collection_name)``
- ``search(vectors, collection_name, top_k)``

Two convenience helpers (``count``, ``drop``) and two record-level helpers
(``delete``, ``update``) are added so benchmark cleanup paths can stay
backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


# Metric names are normalised to the RAGPerf vocabulary. Each adapter maps
# them to its backend-specific identifier (e.g. Chroma uses "cosine" verbatim,
# Qdrant uses Distance.COSINE, Milvus uses "IP" for inner-product).
MetricType = str  # "cosine" | "l2" | "ip"


class _EmbeddingLike(Protocol):
    """Minimal embedding-model contract used by record-level helpers."""

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class SearchHit:
    """A single search result.

    Adapters return these instead of backend-specific objects so callers do
    not need to know whether the underlying row came from Chroma, Qdrant, etc.
    """

    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    """Backend-agnostic configuration for an adapter.

    Adapters may ignore fields that do not apply (e.g. Chroma ignores
    ``index_type`` because it always uses HNSW). Unknown index types for a
    given backend must raise ``NotImplementedError`` from ``build_index``.
    """

    backend: str = "chroma"
    path: str = ".chroma"
    collection: str = "default"
    index_type: str = "hnsw"  # hnsw | ivf | flat (support varies)
    metric: str = "cosine"  # cosine | l2 | ip
    # Optional connection overrides (Qdrant/Milvus server modes):
    host: str | None = None
    port: int | None = None
    api_key: str | None = None
    # Embedding dimension is required for Qdrant/Milvus collections:
    dimension: int | None = None
    # Free-form backend-specific options (e.g. {"ef_construct": 200}).
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VectorStoreConfig":
        """Build a config from a YAML-style dict, ignoring unknown keys.

        Missing keys default to Chroma-equivalent behaviour so an absent
        ``vector_store:`` block reproduces the pre-refactor default.
        """

        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        kwargs: dict[str, Any] = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


class VectorStoreAdapter(ABC):
    """Abstract vector-store adapter.

    Concrete adapters implement the RAGPerf ``DBInstance`` API. The base
    class deliberately does NOT mandate the ``__init__`` signature so each
    backend can require the connection parameters it actually needs; the
    factory (:func:`benchmark.vectorstore.get_vector_store`) knows how to
    construct each one.
    """

    #: Short identifier ("chroma", "qdrant", ...). Set by subclasses.
    db_type: str = ""

    # ---- RAGPerf DBInstance API --------------------------------------------

    @abstractmethod
    def build_index(
        self,
        index_type: str,
        metric_type: MetricType,
        collection_name: str,
    ) -> None:
        """Create an (empty) collection/index ready for inserts.

        Implementations must be idempotent: re-creating an existing
        collection with the same parameters is a no-op.
        """

    @abstractmethod
    def insert(
        self,
        vectors: Sequence[Sequence[float]],
        chunks: Sequence[Any],
        collection_name: str,
    ) -> int:
        """Insert pre-computed ``vectors`` alongside ``chunks``.

        ``chunks`` is any sequence whose elements expose ``page_content``
        (LangChain ``Document``) or ``text``/``metadata`` (plain dicts).
        Returns the number of rows actually inserted.
        """

    @abstractmethod
    def search(
        self,
        vectors: Sequence[Sequence[float]],
        collection_name: str,
        top_k: int,
    ) -> list[list[SearchHit]]:
        """Run one search per query vector, returning top_k hits per query."""

    # ---- Record-level helpers ----------------------------------------------

    @abstractmethod
    def delete(self, file_id: str) -> None:
        """Delete all rows whose metadata ``file_id``/``id`` matches."""

    def update(self, file_id: str, new_text: str) -> None:
        """Replace ``file_id``'s text with ``new_text``.

        Default implementation: delete + re-insert. Backends that support
        in-place update can override.
        """
        raise NotImplementedError(
            f"{self.db_type} adapter does not implement update(); override "
            "in subclass or use delete()+insert() at the call site."
        )

    @abstractmethod
    def count(self, collection_name: str | None = None) -> int:
        """Number of rows in the (current/default) collection."""

    @abstractmethod
    def drop(self, collection_name: str | None = None) -> None:
        """Drop a collection. Safe to call on a non-existent collection."""

    # ---- Optional helpers --------------------------------------------------

    def insert_texts(
        self,
        texts: Sequence[str],
        embedding_model: _EmbeddingLike,
        collection_name: str,
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> int:
        """Embed ``texts`` with ``embedding_model`` then ``insert``.

        Convenience wrapper for callers that do not want to pre-compute
        vectors. Not part of the strict RAGPerf API but small enough to
        share.
        """

        vectors = embedding_model.embed_documents(list(texts))
        meta_list = list(metadata) if metadata is not None else []
        chunks = [
            {
                "text": text,
                "metadata": dict(meta_list[i]) if i < len(meta_list) else {},
            }
            for i, text in enumerate(texts)
        ]
        return self.insert(vectors, chunks, collection_name)

    def search_queries(
        self,
        queries: Sequence[str],
        embedding_model: _EmbeddingLike,
        collection_name: str,
        top_k: int,
    ) -> list[list[SearchHit]]:
        """Embed ``queries`` then ``search``. Mirrors ``insert_texts``."""
        vectors = [embedding_model.embed_query(q) for q in queries]
        return self.search(vectors, collection_name, top_k)


def chunk_text(chunk: Any) -> str:
    """Extract textual content from a LangChain Document or plain dict."""
    if hasattr(chunk, "page_content"):
        return str(chunk.page_content)
    if isinstance(chunk, dict):
        return str(chunk.get("text") or chunk.get("page_content") or "")
    return str(chunk)


def chunk_metadata(chunk: Any) -> dict[str, Any]:
    """Extract metadata from a LangChain Document or plain dict."""
    if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
        return dict(chunk.metadata)
    if isinstance(chunk, dict):
        meta = chunk.get("metadata")
        if isinstance(meta, dict):
            return dict(meta)
    return {}
