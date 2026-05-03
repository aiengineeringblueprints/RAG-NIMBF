import hashlib
import logging
import mlflow
import threading
from pathlib import Path
from typing import Any

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
# Stores already-built Chroma vector stores so repeated configs skip embedding.
_vector_store_cache: dict[str, Chroma] = {}


def _cache_key(
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    chunking_strategy: str,
    dataset_name: str = "",
    *,
    embedding_provider: str = "",
    dataset_subset: str = "",
    dataset_sample_size: int | None = None,
    corpus_fingerprint: str = "",
) -> str:
    """Deterministic key derived from the parameters that affect embeddings."""
    raw = (
        f"{embedding_provider}|{embedding_model_name}|{chunk_size}|"
        f"{chunk_overlap}|{chunking_strategy}|{dataset_name}|"
        f"{dataset_subset}|{dataset_sample_size}|{corpus_fingerprint}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()


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
) -> Chroma:
    """Build (or retrieve from cache) a Chroma vector store.

    When *cache_key* is provided, the built vector store is cached under that
    key and subsequent calls with the same key return the cached store
    directly, skipping re-embedding.

    Collections are persisted to ``.chroma/`` on disk and survive across runs.
    If a collection with the same name already exists, it is reused as-is.
    """
    # Fast path: return cached store when the key matches.
    if cache_key is not None:
        with _chroma_lock:
            cached = _vector_store_cache.get(cache_key)
            if cached is not None:
                logger.info("Reusing in-memory cached vector store (key=%s)", cache_key)
                return cached

    client = _get_client()
    embedding_model = get_embedding_model(
        embedding_model_name, ollama_base_url, ollama_api_key,
        provider=embedding_provider,
    )

    with _chroma_lock:
        existing = [c.name for c in client.list_collections()]
        if collection_name in existing:
            logger.info("Reusing persisted collection '%s'", collection_name)
            vector_store = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embedding_model,
            )
            if cache_key is not None:
                _vector_store_cache[cache_key] = vector_store
            return vector_store

        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )
        vector_store.add_documents(chunks)
        logger.info("Built new collection '%s' with %d chunks", collection_name, len(chunks))

    if cache_key is not None:
        with _chroma_lock:
            _vector_store_cache[cache_key] = vector_store

    return vector_store


@mlflow.trace(name="retrieve", span_type="func")
def retrieve(
    vector_store: Chroma,
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
