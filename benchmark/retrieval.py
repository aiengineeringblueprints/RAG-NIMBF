import hashlib
import logging
import threading
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from benchmark.embedding import get_embedding_model

logger = logging.getLogger(__name__)

# Single shared client + lock: ChromaDB's Rust bindings crash when EphemeralClient
# is created from multiple threads simultaneously.
_chroma_client: Any = None
_chroma_lock = threading.Lock()

# Embedding cache: keyed by a hash of (embedding_model_name, chunk_size,
# chunk_overlap, chunking_strategy).  Stores the already-built Chroma vector
# store so it can be reused when the same combination appears again.
_vector_store_cache: dict[str, Chroma] = {}


def _cache_key(
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    chunking_strategy: str,
) -> str:
    """Deterministic key derived from the parameters that affect embeddings."""
    raw = f"{embedding_model_name}|{chunk_size}|{chunk_overlap}|{chunking_strategy}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_client() -> Any:
    global _chroma_client
    with _chroma_lock:
        if _chroma_client is None:
            _chroma_client = chromadb.EphemeralClient()
        return _chroma_client


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
    """
    # Fast path: return cached store when the key matches.
    if cache_key is not None:
        with _chroma_lock:
            cached = _vector_store_cache.get(cache_key)
            if cached is not None:
                return cached

    client = _get_client()
    embedding_model = get_embedding_model(
        embedding_model_name, ollama_base_url, ollama_api_key,
        provider=embedding_provider,
    )

    # The lock must cover the entire build sequence (delete + create + add)
    # so that concurrent threads never race on the same collection.
    with _chroma_lock:
        existing = [c.name for c in client.list_collections()]
        if collection_name in existing:
            client.delete_collection(collection_name)

        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )
        vector_store.add_documents(chunks)

    if cache_key is not None:
        with _chroma_lock:
            _vector_store_cache[cache_key] = vector_store

    return vector_store


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
