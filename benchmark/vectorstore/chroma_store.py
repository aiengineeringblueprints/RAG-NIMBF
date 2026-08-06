"""Chroma adapter.

Wraps a ``chromadb.PersistentClient`` (the same one the legacy
``benchmark.retrieval`` module caches) so collections created here are
visible to ``cleanup_collection`` / ``clear_cache`` and vice-versa.

Required dependency (already in requirements.txt):
    pip install chromadb langchain-chroma
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Sequence

from benchmark.vectorstore.base import (
    MetricType,
    SearchHit,
    VectorStoreAdapter,
    chunk_metadata,
    chunk_text,
)

logger = logging.getLogger(__name__)

# Chroma always uses HNSW under the hood. We surface this in error messages
# so callers know "ivf"/"flat" are not negotiable.
_SUPPORTED_INDEX_TYPES = {"hnsw"}
_SUPPORTED_METRICS = {"cosine", "l2", "ip"}


class ChromaVectorStore(VectorStoreAdapter):
    """RAGPerf-style Chroma adapter.

    Matches Figure 4 of arXiv:2603.10765v1::

        __init__(db_type, db_path, client)
        build_index(index_type, metric_type, collection_name)
        insert(vectors, chunks, collection_name)
        search(vectors, collection_name, top_k)

    The constructor signature mirrors RAGPerf exactly. ``client`` may be
    ``None`` in which case a shared ``chromadb.PersistentClient`` is obtained
    from ``benchmark.retrieval`` so cleanup helpers keep working.
    """

    db_type = "chroma"

    def __init__(
        self,
        db_type: str = "chroma",
        db_path: str = ".chroma",
        client: Any = None,
    ) -> None:
        if db_type and db_type.lower() != "chroma":
            raise ValueError(
                f"ChromaVectorStore received db_type={db_type!r}; expected 'chroma'."
            )
        self.db_path = db_path
        self._client = client
        # Chroma's PersistentClient is not safe to share across threads for
        # concurrent writes; the legacy module already serialises access with
        # a global lock. Reuse it.
        from benchmark.retrieval import _chroma_lock, _get_client

        self._lock = _chroma_lock
        self._client = client if client is not None else _get_client()
        # Persist db_path for adapters constructed without a parent retrieval
        # context — used in error/log messages only.
        self._retrieval_get_client = _get_client

    # ---- DBInstance API ----------------------------------------------------

    def build_index(
        self,
        index_type: str,
        metric_type: MetricType,
        collection_name: str,
    ) -> None:
        if index_type.lower() not in _SUPPORTED_INDEX_TYPES:
            raise NotImplementedError(
                f"Chroma only supports index_type='hnsw'; got {index_type!r}."
            )
        if metric_type.lower() not in _SUPPORTED_METRICS:
            raise ValueError(
                f"Chroma metric must be one of {_SUPPORTED_METRICS}; "
                f"got {metric_type!r}."
            )
        # Chroma's "hnsw:space" metadata key selects cosine/l2/ip.
        with self._lock:
            existing = {c.name for c in self._client.list_collections()}
            if collection_name in existing:
                logger.debug(
                    "Chroma collection %r already exists; build_index is a no-op.",
                    collection_name,
                )
                return
            self._client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": metric_type.lower()},
            )

    def insert(
        self,
        vectors: Sequence[Sequence[float]],
        chunks: Sequence[Any],
        collection_name: str,
    ) -> int:
        if len(vectors) != len(chunks):
            raise ValueError(
                f"vectors ({len(vectors)}) and chunks ({len(chunks)}) must "
                "have the same length."
            )
        if not vectors:
            return 0
        with self._lock:
            collection = self._client.get_or_create_collection(
                name=collection_name,
            )
            ids = [str(uuid.uuid4()) for _ in vectors]
            documents = [chunk_text(c) for c in chunks]
            metadatas = [chunk_metadata(c) for c in chunks]
            # Chroma rejects empty metadata dicts ("non-empty dict required").
            # Pass None for empty rows so callers don't have to set a key.
            metadatas_for_chroma = [m or None for m in metadatas]
            collection.add(
                ids=ids,
                embeddings=[list(v) for v in vectors],
                documents=documents,
                metadatas=metadatas_for_chroma,
            )
        return len(vectors)

    def search(
        self,
        vectors: Sequence[Sequence[float]],
        collection_name: str,
        top_k: int,
    ) -> list[list[SearchHit]]:
        if not vectors:
            return []
        with self._lock:
            try:
                collection = self._client.get_collection(collection_name)
            except Exception as exc:
                # Chroma raises a custom exception; normalise to a clear msg.
                raise RuntimeError(
                    f"Chroma collection {collection_name!r} not found: {exc}"
                ) from exc
            query_result = collection.query(
                query_embeddings=[list(v) for v in vectors],
                n_results=top_k,
                include=["metadatas", "documents", "distances"],
            )

        results: list[list[SearchHit]] = []
        ids_batch = query_result.get("ids", [])
        docs_batch = query_result.get("documents", [])
        meta_batch = query_result.get("metadatas", [])
        dist_batch = query_result.get("distances", [])
        for i in range(len(vectors)):
            ids = ids_batch[i] if i < len(ids_batch) else []
            docs = docs_batch[i] if i < len(docs_batch) else []
            metas = meta_batch[i] if i < len(meta_batch) else []
            dists = dist_batch[i] if i < len(dist_batch) else []
            hits: list[SearchHit] = []
            for j, doc_id in enumerate(ids):
                distance = float(dists[j]) if j < len(dists) else float("nan")
                # Chroma returns distances (lower = better). Convert to a
                # similarity-like score for the cosine space (1 - distance).
                score = 1.0 - distance
                hits.append(
                    SearchHit(
                        id=str(doc_id),
                        score=score,
                        text=str(docs[j]) if j < len(docs) else "",
                        metadata=dict(metas[j]) if j < len(metas) else {},
                    )
                )
            results.append(hits)
        return results

    # ---- Record-level helpers ---------------------------------------------

    def delete(self, file_id: str) -> None:
        """Delete rows whose metadata ``file_id`` matches.

        Scans every collection because Chroma's where filter requires the
        collection to be known up-front. The benchmark only ever needs to
        delete by file_id across the active collection, so this is good
        enough for now and easy to make stricter later.
        """
        with self._lock:
            for collection in self._client.list_collections():
                coll = self._client.get_collection(collection.name)
                # ``where`` filter on metadata key file_id.
                try:
                    coll.delete(where={"file_id": file_id})
                except Exception:
                    # Metadata key may not exist on this collection; skip.
                    continue

    def count(self, collection_name: str | None = None) -> int:
        with self._lock:
            if collection_name is None:
                # Sum across all collections; matches what callers expect
                # when they haven't pinned a single collection.
                return sum(
                    self._client.get_collection(c.name).count()
                    for c in self._client.list_collections()
                )
            try:
                return int(self._client.get_collection(collection_name).count())
            except Exception as exc:
                raise RuntimeError(
                    f"Chroma collection {collection_name!r} not found: {exc}"
                ) from exc

    def drop(self, collection_name: str | None = None) -> None:
        with self._lock:
            if collection_name is None:
                for collection in self._client.list_collections():
                    self._client.delete_collection(collection.name)
                return
            existing = {c.name for c in self._client.list_collections()}
            if collection_name in existing:
                self._client.delete_collection(collection_name)
