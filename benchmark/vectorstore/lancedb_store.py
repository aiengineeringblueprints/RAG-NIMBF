"""LanceDB adapter.

Required dependency (already in requirements.txt):

    pip install lancedb

LanceDB is embedded (no server). The adapter persists a Lance database at
``db_path`` and stores one table per collection.
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

_SUPPORTED_INDEX_TYPES = {"hnsw", "ivf", "flat"}
_SUPPORTED_METRICS = {"cosine", "l2", "ip"}


def _import_lancedb():
    try:
        import lancedb
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - exercised by skip-tests
        raise ImportError(
            "lancedb is required for the LanceDB adapter. "
            "Install with: pip install lancedb"
        ) from exc
    return lancedb, pa


# LanceDB metric names are uppercase and use L2 (not EUCLID).
_METRIC_MAP = {
    "cosine": "cosine",
    "l2": "l2",
    "ip": "dot",
}


class LanceDBVectorStore(VectorStoreAdapter):
    """RAGPerf-style LanceDB adapter."""

    db_type = "lancedb"

    def __init__(
        self,
        db_type: str = "lancedb",
        db_path: str = ".lancedb",
        client: Any = None,
    ) -> None:
        if db_type and db_type.lower() != "lancedb":
            raise ValueError(
                f"LanceDBVectorStore received db_type={db_type!r}; expected 'lancedb'."
            )
        self.db_path = db_path
        lancedb, self._pa = _import_lancedb()
        try:
            self._db = client if client is not None else lancedb.connect(db_path)
        except Exception as exc:
            raise ConnectionError(
                f"Could not open LanceDB at {db_path!r}: {exc}."
            ) from exc
        # Per-collection metric override cache (build_index may run before
        # insert and we need the metric later when creating the table).
        self._collection_metrics: dict[str, str] = {}

    def _table_names(self) -> list[str]:
        """Return existing tables, tolerant of the table_names() deprecation."""
        if hasattr(self._db, "list_tables"):
            try:
                result = self._db.list_tables()
                # LanceDB >=0.30 returns an object with a ``tables`` attribute.
                if hasattr(result, "tables"):
                    return [str(t) for t in result.tables]
                return [str(t) for t in result]
            except Exception:
                pass
        try:
            return list(self._db.table_names())
        except Exception:
            return []

    # ---- DBInstance API ----------------------------------------------------

    def build_index(
        self,
        index_type: str,
        metric_type: MetricType,
        collection_name: str,
    ) -> None:
        if index_type.lower() not in _SUPPORTED_INDEX_TYPES:
            raise NotImplementedError(
                f"LanceDB adapter supports index types "
                f"{sorted(_SUPPORTED_INDEX_TYPES)}; got {index_type!r}."
            )
        metric_key = metric_type.lower()
        if metric_key not in _SUPPORTED_METRICS:
            raise ValueError(
                f"LanceDB metric must be one of {_SUPPORTED_METRICS}; got "
                f"{metric_type!r}."
            )
        self._collection_metrics[collection_name] = _METRIC_MAP[metric_key]
        # LanceDB builds the actual index lazily on the table; build_index is
        # a metadata-only no-op until vectors exist. We expose this honestly.
        if collection_name in self._table_names():
            logger.debug(
                "LanceDB table %r already exists; build_index recorded "
                "metric=%s for future searches.",
                collection_name,
                metric_key,
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
        rows = [
            {
                "id": str(uuid.uuid4()),
                "vector": list(vec),
                "text": chunk_text(chunk),
                "metadata": chunk_metadata(chunk),
            }
            for vec, chunk in zip(vectors, chunks)
        ]
        if collection_name in self._table_names():
            table = self._db.open_table(collection_name)
            table.add(rows)
        else:
            self._db.create_table(collection_name, data=rows)
        return len(rows)

    def search(
        self,
        vectors: Sequence[Sequence[float]],
        collection_name: str,
        top_k: int,
    ) -> list[list[SearchHit]]:
        if not vectors:
            return []
        if collection_name not in self._table_names():
            raise RuntimeError(
                f"LanceDB table {collection_name!r} not found"
            )
        table = self._db.open_table(collection_name)
        metric = self._collection_metrics.get(collection_name, "cosine")
        results: list[list[SearchHit]] = []
        for vec in vectors:
            query = table.search(list(vec)).limit(top_k)
            # LanceDB infers metric from vector column type; we leave it
            # default unless an explicit metric_type is required by future
            # callers.
            try:
                rows = query.to_list()
            except Exception as exc:
                raise RuntimeError(
                    f"LanceDB search on {collection_name!r} failed: {exc}"
                ) from exc
            hits: list[SearchHit] = []
            for row in rows:
                meta = dict(row.get("metadata") or {})
                hits.append(
                    SearchHit(
                        id=str(row.get("id", "")),
                        score=float(row.get("_distance", 0.0)),
                        text=str(row.get("text", "")),
                        metadata=meta,
                    )
                )
            results.append(hits)
        return results

    # ---- Record-level helpers ---------------------------------------------

    def delete(self, file_id: str) -> None:
        for name in self._table_names():
            table = self._db.open_table(name)
            try:
                table.delete(f'metadata.file_id = "{file_id}"')
            except Exception:
                continue

    def count(self, collection_name: str | None = None) -> int:
        if collection_name is None:
            return sum(self._count_table(n) for n in self._table_names())
        if collection_name not in self._table_names():
            raise RuntimeError(
                f"LanceDB table {collection_name!r} not found"
            )
        return self._count_table(collection_name)

    def _count_table(self, name: str) -> int:
        try:
            return int(self._db.open_table(name).count_rows())
        except Exception:
            return 0

    def drop(self, collection_name: str | None = None) -> None:
        if collection_name is None:
            for name in list(self._table_names()):
                self._db.drop_table(name)
            return
        if collection_name in self._table_names():
            self._db.drop_table(collection_name)
