"""Qdrant adapter.

Required dependency (optional, install only when using this backend):

    pip install qdrant-client

Supports both embedded (``:memory:`` or local path) and server modes. The
default is a local-file Qdrant at ``db_path``.
"""

from __future__ import annotations

import logging
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

_SUPPORTED_INDEX_TYPES = {"hnsw"}
_SUPPORTED_METRICS = {"cosine", "l2", "ip"}


def _import_qdrant():
    """Lazy import; raise a clear error when the dep is missing."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels
    except ImportError as exc:  # pragma: no cover - exercised by skip-tests
        raise ImportError(
            "qdrant-client is required for the Qdrant adapter. "
            "Install with: pip install qdrant-client"
        ) from exc
    return QdrantClient, qmodels


_METRIC_MAP = {
    "cosine": "COSINE",
    "l2": "EUCLID",
    "ip": "DOT",
}


class QdrantVectorStore(VectorStoreAdapter):
    """RAGPerf-style Qdrant adapter."""

    db_type = "qdrant"

    def __init__(
        self,
        db_type: str = "qdrant",
        db_path: str = ".qdrant",
        client: Any = None,
        *,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        dimension: int | None = None,
    ) -> None:
        if db_type and db_type.lower() != "qdrant":
            raise ValueError(
                f"QdrantVectorStore received db_type={db_type!r}; expected 'qdrant'."
            )
        self.db_path = db_path
        self._dimension = dimension
        QdrantClient, _ = _import_qdrant()  # type: ignore[misc]
        if client is not None:
            self._client = client
        elif host:
            self._client = QdrantClient(
                host=host,
                port=port or 6333,
                api_key=api_key,
            )
        else:
            # Embedded local mode. ``path`` triggers the bundled RocksDB
            # backend, no server required.
            try:
                self._client = QdrantClient(path=db_path)
            except Exception as exc:
                raise ConnectionError(
                    f"Could not open Qdrant at {db_path!r}: {exc}. "
                    "For server mode pass host=<host>."
                ) from exc

    # ---- DBInstance API ----------------------------------------------------

    def build_index(
        self,
        index_type: str,
        metric_type: MetricType,
        collection_name: str,
    ) -> None:
        _, qmodels = _import_qdrant()  # type: ignore[misc]
        if index_type.lower() not in _SUPPORTED_INDEX_TYPES:
            raise NotImplementedError(
                f"Qdrant adapter only supports index_type='hnsw' via this "
                f"benchmark; got {index_type!r}."
            )
        metric_key = metric_type.lower()
        if metric_key not in _SUPPORTED_METRICS:
            raise ValueError(
                f"Qdrant metric must be one of {_SUPPORTED_METRICS}; got "
                f"{metric_type!r}."
            )
        if self._dimension is None:
            raise ValueError(
                "QdrantVectorStore.build_index requires a known embedding "
                "dimension; pass dimension=... to the constructor."
            )
        distance = getattr(qmodels.Distance, _METRIC_MAP[metric_key])
        existing = {c.name for c in self._client.get_collections().collections}
        if collection_name in existing:
            logger.debug(
                "Qdrant collection %r already exists; build_index is a no-op.",
                collection_name,
            )
            return
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=self._dimension,
                distance=distance,
            ),
        )

    def insert(
        self,
        vectors: Sequence[Sequence[float]],
        chunks: Sequence[Any],
        collection_name: str,
    ) -> int:
        _, qmodels = _import_qdrant()  # type: ignore[misc]
        if len(vectors) != len(chunks):
            raise ValueError(
                f"vectors ({len(vectors)}) and chunks ({len(chunks)}) must "
                "have the same length."
            )
        if not vectors:
            return 0
        # Infer dimension on first insert if not provided.
        if self._dimension is None:
            self._dimension = len(vectors[0])
        existing = {c.name for c in self._client.get_collections().collections}
        if collection_name not in existing:
            distance = getattr(
                qmodels.Distance, _METRIC_MAP["cosine"]
            )
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self._dimension,
                    distance=distance,
                ),
            )
        points = []
        for vec, chunk in zip(vectors, chunks):
            metadata = chunk_metadata(chunk)
            points.append(
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=list(vec),
                    payload={"text": chunk_text(chunk), **metadata},
                )
            )
        self._client.upsert(collection_name=collection_name, points=points)
        return len(points)

    def search(
        self,
        vectors: Sequence[Sequence[float]],
        collection_name: str,
        top_k: int,
    ) -> list[list[SearchHit]]:
        _, qmodels = _import_qdrant()  # type: ignore[misc]
        if not vectors:
            return []
        results: list[list[SearchHit]] = []
        for vec in vectors:
            try:
                response = self._client.search(
                    collection_name=collection_name,
                    query_vector=list(vec),
                    limit=top_k,
                    with_payload=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Qdrant search on {collection_name!r} failed: {exc}"
                ) from exc
            hits: list[SearchHit] = []
            for scored in response:
                payload = dict(scored.payload or {})
                hits.append(
                    SearchHit(
                        id=str(scored.id),
                        score=float(scored.score),
                        text=str(payload.pop("text", "")),
                        metadata=payload,
                    )
                )
            results.append(hits)
        return results

    # ---- Record-level helpers ---------------------------------------------

    def delete(self, file_id: str) -> None:
        _, qmodels = _import_qdrant()  # type: ignore[misc]
        for collection in self._client.get_collections().collections:
            self._client.delete(
                collection_name=collection.name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="file_id",
                                match=qmodels.MatchValue(value=file_id),
                            )
                        ]
                    )
                ),
            )

    def count(self, collection_name: str | None = None) -> int:
        if collection_name is None:
            total = 0
            for collection in self._client.get_collections().collections:
                total += self._client.count(
                    collection_name=collection.name,
                    exact=True,
                ).count
            return total
        try:
            return int(
                self._client.count(
                    collection_name=collection_name, exact=True
                ).count
            )
        except Exception as exc:
            raise RuntimeError(
                f"Qdrant collection {collection_name!r} not found: {exc}"
            ) from exc

    def drop(self, collection_name: str | None = None) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if collection_name is None:
            for name in list(existing):
                self._client.delete_collection(collection_name=name)
            return
        if collection_name in existing:
            self._client.delete_collection(collection_name=collection_name)
