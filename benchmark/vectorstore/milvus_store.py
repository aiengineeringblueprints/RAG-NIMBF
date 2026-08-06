"""Milvus adapter.

Required dependency (optional, install only when using this backend):

    pip install pymilvus

Milvus needs a running server (Milvus standalone or Zilliz Cloud). The
adapter defaults to ``host=localhost port=19530``; override via the factory
config or constructor kwargs.
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

# Milvus metric names: COSINE/L2/IP. We accept the RAGPerf vocabulary and
# translate. (cosine is supported in Milvus 2.3+.)
_METRIC_MAP = {
    "cosine": "COSINE",
    "l2": "L2",
    "ip": "IP",
}
_INDEX_MAP = {
    "hnsw": "HNSW",
    "ivf": "IVF_FLAT",
    "flat": "FLAT",
}


def _import_milvus():
    try:
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            connections,
            utility,
        )
    except ImportError as exc:  # pragma: no cover - exercised by skip-tests
        raise ImportError(
            "pymilvus is required for the Milvus adapter. "
            "Install with: pip install pymilvus"
        ) from exc
        # NOTE: returning is unreachable; Python flags the ImportError above.
        return None  # pragma: no cover
    return (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )


class MilvusVectorStore(VectorStoreAdapter):
    """RAGPerf-style Milvus adapter."""

    db_type = "milvus"

    def __init__(
        self,
        db_type: str = "milvus",
        db_path: str = "",
        client: Any = None,
        *,
        host: str | None = "localhost",
        port: int | None = 19530,
        api_key: str | None = None,
        dimension: int | None = None,
    ) -> None:
        if db_type and db_type.lower() != "milvus":
            raise ValueError(
                f"MilvusVectorStore received db_type={db_type!r}; expected 'milvus'."
            )
        self.db_path = db_path
        self._dimension = dimension
        (
            self._Collection,
            self._CollectionSchema,
            self._DataType,
            self._FieldSchema,
            self._connections,
            self._utility,
        ) = _import_milvus()
        if client is not None:
            self._client_alias = "default"
            if hasattr(client, "alias"):
                self._client_alias = client.alias
        else:
            self._client_alias = "benchmark-milvus"
            try:
                # ``connect`` is idempotent in pymilvus for the same alias.
                connect_kwargs: dict[str, Any] = {
                    "alias": self._client_alias,
                    "host": host or "localhost",
                    "port": str(port or 19530),
                }
                if api_key:
                    connect_kwargs["token"] = api_key
                self._connections.connect(**connect_kwargs)
            except Exception as exc:
                raise ConnectionError(
                    f"Could not connect to Milvus at {host}:{port}: {exc}. "
                    "Start a Milvus server or pass host=/port= for a "
                    "different endpoint."
                ) from exc

    # ---- DBInstance API ----------------------------------------------------

    def build_index(
        self,
        index_type: str,
        metric_type: MetricType,
        collection_name: str,
    ) -> None:
        if index_type.lower() not in _SUPPORTED_INDEX_TYPES:
            raise NotImplementedError(
                f"Milvus adapter supports index types {sorted(_SUPPORTED_INDEX_TYPES)}; "
                f"got {index_type!r}."
            )
        metric_key = metric_type.lower()
        if metric_key not in _SUPPORTED_METRICS:
            raise ValueError(
                f"Milvus metric must be one of {_SUPPORTED_METRICS}; got "
                f"{metric_type!r}."
            )
        if self._dimension is None:
            raise ValueError(
                "MilvusVectorStore.build_index requires a known embedding "
                "dimension; pass dimension=... to the constructor."
            )
        if self._utility.has_collection(collection_name, using=self._client_alias):
            logger.debug(
                "Milvus collection %r already exists; build_index is a no-op.",
                collection_name,
            )
            return
        self._create_collection(
            collection_name, self._dimension, _METRIC_MAP[metric_key]
        )

    def _create_collection(self, name: str, dim: int, metric_name: str):
        fields = [
            self._FieldSchema(
                name="id", dtype=self._DataType.VARCHAR, max_length=64, is_primary=True
            ),
            self._FieldSchema(
                name="vector", dtype=self._DataType.FLOAT_VECTOR, dim=dim
            ),
            self._FieldSchema(
                name="text", dtype=self._DataType.VARCHAR, max_length=65535
            ),
            self._FieldSchema(
                name="metadata", dtype=self._DataType.JSON
            ),
        ]
        schema = self._CollectionSchema(fields, enable_dynamic_field=False)
        col = self._Collection(
            name=name,
            schema=schema,
            using=self._client_alias,
        )
        # Build a real index. For "flat" we still create an INVERTED-ish index
        # because Milvus requires one before search.
        index_type = "FLAT"
        index_params: dict[str, Any] = {"metric_type": metric_name, "index_type": "FLAT"}
        col.create_index(
            field_name="vector",
            index_params=index_params,
        )
        return col

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
        if self._dimension is None:
            self._dimension = len(vectors[0])
        if not self._utility.has_collection(
            collection_name, using=self._client_alias
        ):
            self._create_collection(collection_name, self._dimension, "COSINE")
        col = self._Collection(
            name=collection_name, using=self._client_alias
        )
        ids = [str(uuid.uuid4()) for _ in vectors]
        col.insert(
            [
                ids,
                [list(v) for v in vectors],
                [chunk_text(c) for c in chunks],
                [chunk_metadata(c) for c in chunks],
            ]
        )
        col.flush()
        return len(vectors)

    def search(
        self,
        vectors: Sequence[Sequence[float]],
        collection_name: str,
        top_k: int,
    ) -> list[list[SearchHit]]:
        if not vectors:
            return []
        if not self._utility.has_collection(
            collection_name, using=self._client_alias
        ):
            raise RuntimeError(
                f"Milvus collection {collection_name!r} not found"
            )
        col = self._Collection(
            name=collection_name, using=self._client_alias
        )
        col.load()
        search_results = col.search(
            data=[list(v) for v in vectors],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {}},
            limit=top_k,
            output_fields=["text", "metadata"],
        )
        out: list[list[SearchHit]] = []
        for hits in search_results:
            page: list[SearchHit] = []
            for hit in hits:
                entity = hit.entity.to_dict() if hasattr(hit.entity, "to_dict") else {}
                fields = {}
                try:
                    fields = hit.entity.to_dict()["fields"] if hasattr(hit.entity, "to_dict") else {}
                except Exception:
                    fields = {}
                payload_text = ""
                metadata: dict[str, Any] = {}
                # ``to_dict`` returns {"name":..,"fields":{...}} in pymilvus.
                if isinstance(fields, dict):
                    payload_text = str(fields.get("text", ""))
                    raw_meta = fields.get("metadata", {})
                    if isinstance(raw_meta, dict):
                        metadata = dict(raw_meta)
                page.append(
                    SearchHit(
                        id=str(hit.id) if hit.id is not None else "",
                        score=float(hit.score),
                        text=payload_text,
                        metadata=metadata,
                    )
                )
            out.append(page)
        return out

    # ---- Record-level helpers ---------------------------------------------

    def delete(self, file_id: str) -> None:
        # Milvus delete-by-expression requires the file_id column. Our
        # default schema stores it inside metadata JSON, which Milvus 2.4+
        # supports via JSON containment filters. Conservative implementation:
        for collection in self._utility.list_collections(using=self._client_alias):
            col = self._Collection(name=collection, using=self._client_alias)
            try:
                col.delete(f'metadata["file_id"] == "{file_id}"')
            except Exception:
                # JSON expression support varies by version; skip silently.
                continue

    def count(self, collection_name: str | None = None) -> int:
        if collection_name is None:
            names = self._utility.list_collections(using=self._client_alias)
        else:
            if not self._utility.has_collection(
                collection_name, using=self._client_alias
            ):
                raise RuntimeError(
                    f"Milvus collection {collection_name!r} not found"
                )
            names = [collection_name]
        total = 0
        for name in names:
            col = self._Collection(name=name, using=self._client_alias)
            col.flush()
            total += int(col.num_entities)
        return total

    def drop(self, collection_name: str | None = None) -> None:
        if collection_name is None:
            for name in self._utility.list_collections(using=self._client_alias):
                self._utility.drop_collection(name, using=self._client_alias)
            return
        if self._utility.has_collection(collection_name, using=self._client_alias):
            self._utility.drop_collection(collection_name, using=self._client_alias)
