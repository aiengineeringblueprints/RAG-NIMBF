import hashlib
import logging
import mlflow
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from benchmark.embedding import get_embedding_model

logger = logging.getLogger(__name__)

_CHROMA_DIR = Path(".chroma")

# Single shared client + lock: ChromaDB's Rust bindings crash when
# PersistentClient is created from multiple threads simultaneously.
_chroma_client: Any = None
_chroma_lock = threading.Lock()

# Embedding cache: keyed by every input that can affect retrieval contents.
# Stores already-built vector stores so repeated configs skip embedding.
_vector_store_cache: dict[str, Any] = {}


def _cache_key(
    embedding_model_name: str,
    chunk_size: int | None,
    chunk_overlap: int | None,
    chunking_strategy: str,
    dataset_name: str = "",
    *,
    embedding_provider: str = "",
    dataset_subset: str = "",
    dataset_sample_size: int | None = None,
    corpus_fingerprint: str = "",
    vector_db_backend: str = "chroma",
) -> str:
    """Deterministic key derived from the parameters that affect embeddings."""
    raw = (
        f"{embedding_provider}|{embedding_model_name}|{chunk_size}|"
        f"{chunk_overlap}|{chunking_strategy}|{dataset_name}|"
        f"{dataset_subset}|{dataset_sample_size}|{corpus_fingerprint}|"
        f"{vector_db_backend}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()


class LanceDBVectorStore:
    """Small LanceDB adapter with LangChain-like retrieval methods."""

    def __init__(
        self,
        path: str,
        table_name: str,
        embedding_model: Any,
        *,
        create_if_missing: bool,
        chunks: list[Document] | None = None,
    ) -> None:
        import lancedb

        self.db = lancedb.connect(path)
        self.table_name = table_name
        self.embedding_model = embedding_model
        existing = set(self.db.table_names())
        if table_name in existing:
            self.table = self.db.open_table(table_name)
            return
        if not create_if_missing:
            raise RuntimeError(
                f"LanceDB table '{table_name}' missing at {path}. "
                "Run BENCHMARK_STAGE=index or BENCHMARK_STAGE=all first."
            )
        if not chunks:
            raise RuntimeError("Cannot create LanceDB table without chunks.")

        texts = [doc.page_content for doc in chunks]
        vectors = embedding_model.embed_documents(texts)
        data = [
            {
                "vector": vector,
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc, vector in zip(chunks, vectors)
        ]
        self.table = self.db.create_table(table_name, data=data)

    def similarity_search(self, query: str, k: int = 3, **_: Any) -> list[Document]:
        vector = self.embedding_model.embed_query(query)
        rows = self.table.search(vector).limit(k).to_list()
        return [
            Document(
                page_content=str(row.get("text", "")),
                metadata=row.get("metadata") or {},
            )
            for row in rows
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 3,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **_: Any,
    ) -> list[Document]:
        # LanceDB search already returns ranked candidates. Keep interface
        # compatible; full MMR can be added once we store candidate vectors.
        return self.similarity_search(query, k=k)


@dataclass(frozen=True)
class VectorStoreBuildContext:
    """Inputs needed by a vector-store backend implementation."""

    chunks: list[Document]
    embedding_model: Any
    collection_name: str
    expected_count: int
    create_if_missing: bool
    lancedb_path: str


class VectorStoreBackend(Protocol):
    """Factory seam for persisted vector-store backends."""

    name: str

    def build(self, context: VectorStoreBuildContext) -> Any:
        """Build or reuse a vector store for the given context."""


class LanceDBVectorStoreBackend:
    """Factory for LanceDB-backed vector stores."""

    name = "lancedb"

    def build(self, context: VectorStoreBuildContext) -> Any:
        return LanceDBVectorStore(
            context.lancedb_path,
            context.collection_name,
            context.embedding_model,
            create_if_missing=context.create_if_missing,
            chunks=context.chunks,
        )


class ChromaVectorStoreBackend:
    """Factory for Chroma-backed vector stores."""

    name = "chroma"

    def build(self, context: VectorStoreBuildContext) -> Any:
        client = _get_client()
        with _chroma_lock:
            existing = [c.name for c in client.list_collections()]
            if context.collection_name in existing:
                collection = client.get_collection(context.collection_name)
                persisted_count = collection.count()
                if persisted_count == context.expected_count:
                    logger.info(
                        "Reusing persisted collection '%s' with %d documents",
                        context.collection_name,
                        persisted_count,
                    )
                    return Chroma(
                        client=client,
                        collection_name=context.collection_name,
                        embedding_function=context.embedding_model,
                    )

                message = (
                    f"Chroma collection '{context.collection_name}' has {persisted_count} "
                    f"documents, expected {context.expected_count}."
                )
                if not context.create_if_missing:
                    raise RuntimeError(
                        f"{message} Rebuild the index with BENCHMARK_STAGE=index "
                        "or BENCHMARK_STAGE=all before querying."
                    )

                logger.warning("%s Rebuilding collection.", message)
                client.delete_collection(context.collection_name)

            if not context.create_if_missing:
                raise RuntimeError(
                    f"Chroma collection '{context.collection_name}' missing. "
                    "Run BENCHMARK_STAGE=index or BENCHMARK_STAGE=all first."
                )

            vector_store = Chroma(
                client=client,
                collection_name=context.collection_name,
                embedding_function=context.embedding_model,
            )
            vector_store.add_documents(context.chunks)
            logger.info(
                "Built new collection '%s' with %d chunks",
                context.collection_name,
                len(context.chunks),
            )
            return vector_store


_VECTOR_STORE_BACKENDS: dict[str, VectorStoreBackend] = {
    ChromaVectorStoreBackend.name: ChromaVectorStoreBackend(),
    LanceDBVectorStoreBackend.name: LanceDBVectorStoreBackend(),
}


def register_vector_store_backend(name: str, backend: VectorStoreBackend) -> None:
    """Register a vector-store backend by config name."""
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Vector store backend name must not be empty")
    _VECTOR_STORE_BACKENDS[normalized_name] = backend


def available_vector_store_backends() -> tuple[str, ...]:
    return tuple(sorted(_VECTOR_STORE_BACKENDS))


def _get_vector_store_backend(vector_db_backend: str) -> VectorStoreBackend:
    normalized_backend = vector_db_backend.strip().lower()
    try:
        return _VECTOR_STORE_BACKENDS[normalized_backend]
    except KeyError as exc:
        available = ", ".join(available_vector_store_backends())
        raise ValueError(
            f"Unsupported vector DB backend: {vector_db_backend}. "
            f"Available: {available}"
        ) from exc


def _get_client() -> Any:
    global _chroma_client
    with _chroma_lock:
        if _chroma_client is None:
            _chroma_client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        return _chroma_client


@mlflow.trace(name="build_vector_store", span_type="func", attributes={"component": "embedding"})
def build_vector_store(
    chunks: list[Document],
    embedding_model_name: str,
    collection_name: str,
    ollama_base_url: str = "http://localhost:11434",
    ollama_api_key: str | None = None,
    cache_key: str | None = None,
    *,
    embedding_provider: str = "ollama",
    vector_db_backend: str = "chroma",
    lancedb_path: str = ".lancedb",
    create_if_missing: bool = True,
) -> Any:
    """Build (or retrieve from cache) a vector store.

    When *cache_key* is provided, the built vector store is cached under that
    key and subsequent calls with the same key return the cached store
    directly, skipping re-embedding.

    Collections are persisted to ``.chroma/`` on disk and survive across runs.
    If a collection with the same name already exists, it is reused only when
    its document count matches the newly generated chunk count.
    """
    expected_count = len(chunks)

    # Fast path: return cached store when the key matches and the cached
    # collection still has the expected number of documents.
    if cache_key is not None:
        with _chroma_lock:
            cached = _vector_store_cache.get(cache_key)
            if cached is not None:
                cached_count = _vector_store_count(cached)
                if cached_count is None or cached_count == expected_count:
                    logger.info(
                        "Reusing in-memory cached vector store (key=%s, count=%s)",
                        cache_key,
                        cached_count,
                    )
                    return cached
                logger.warning(
                    "Discarding cached vector store for key=%s: expected %d "
                    "documents, found %s",
                    cache_key,
                    expected_count,
                    cached_count,
                )
                _vector_store_cache.pop(cache_key, None)

    embedding_model = get_embedding_model(
        embedding_model_name, ollama_base_url, ollama_api_key,
        provider=embedding_provider,
    )

    backend = _get_vector_store_backend(vector_db_backend)
    context = VectorStoreBuildContext(
        chunks=chunks,
        embedding_model=embedding_model,
        collection_name=collection_name,
        expected_count=expected_count,
        create_if_missing=create_if_missing,
        lancedb_path=lancedb_path,
    )
    vector_store = backend.build(context)

    if cache_key is not None:
        with _chroma_lock:
            _vector_store_cache[cache_key] = vector_store

    return vector_store


def _vector_store_count(vector_store: Any) -> int | None:
    """Return document count for Chroma-like stores, or None if unavailable."""
    collection = getattr(vector_store, "_collection", None)
    count = getattr(collection, "count", None)
    if callable(count):
        return count()
    return None


@mlflow.trace(name="retrieve", span_type="func")
def retrieve(
    vector_store: Any,
    query: str,
    top_k: int = 3,
    *,
    retrieval_strategy: str = "similarity",
    fetch_k: int | None = None,
    mmr_lambda: float = 0.5,
    callbacks: list | None = None,
) -> list[Document]:
    """Retrieve documents from the vector store.

    Parameters
    ----------
    retrieval_strategy:
        ``"similarity"`` (default) for plain similarity search,
        ``"mmr"`` for Maximal Marginal Relevance (diversifies results).
    fetch_k:
        Number of candidates to fetch before MMR filtering.
        Defaults to ``top_k * 4`` when ``None`` and MMR is active.
    mmr_lambda:
        Balance between relevance (1.0) and diversity (0.0) for MMR.
    """
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({
            "retrieval.top_k": top_k,
            "retrieval.strategy": retrieval_strategy,
        })
    config = {"callbacks": callbacks} if callbacks else None
    if retrieval_strategy == "mmr":
        effective_fetch_k = fetch_k if fetch_k is not None else top_k * 4
        return vector_store.max_marginal_relevance_search(
            query, k=top_k, fetch_k=effective_fetch_k, lambda_mult=mmr_lambda,
            **({"configurable": config} if config else {}),
        )
    return vector_store.similarity_search(
        query, k=top_k,
        **({"configurable": config} if config else {}),
    )


def expand_query_with_hyde(
    llm: BaseChatModel,
    question: str,
    *,
    callbacks: list | None = None,
) -> str:
    """Generate a hypothetical answer for HyDE retrieval.

    Uses the LLM to produce a detailed hypothetical answer, which is then
    used as the retrieval query instead of the raw question.  This improves
    retrieval for short or ambiguous queries because the hypothetical answer
    is semantically closer to the actual document chunks.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(
            content=(
                "Write a detailed, factual answer to the question. "
                "Be specific and informative."
            )
        ),
        HumanMessage(content=question),
    ]
    try:
        response = llm.invoke(messages, config={"callbacks": callbacks or []})
        content = str(response.content) if response.content else ""
        if content.strip():
            logger.debug("HyDE expanded query for: %s", question[:60])
            return content
    except Exception as exc:
        logger.warning("HyDE expansion failed, using original query: %s", exc)
    return question


def cleanup_collection(collection_name: str, cache_key: str | None = None) -> None:
    """Delete a single ChromaDB collection by name.

    Safe to call even if the collection does not exist.
    If *cache_key* is provided, the corresponding vector-store cache entry
    is also removed so that subsequent configs don't reference a deleted
    collection.
    """
    client = _get_client()
    with _chroma_lock:
        if cache_key is not None:
            _vector_store_cache.pop(cache_key, None)
        existing = [c.name for c in client.list_collections()]
        if collection_name in existing:
            client.delete_collection(collection_name)


def clear_cache() -> None:
    """Clear the embedding cache and delete all ChromaDB collections."""
    client = _get_client()
    with _chroma_lock:
        _vector_store_cache.clear()
        for collection in client.list_collections():
            client.delete_collection(collection.name)
