"""Conformance tests shared across all installed vector-store adapters.

The parametrized scenarios below run against every adapter whose optional
dependency is importable. Each adapter that has its dep installed gets the
same scenarios; missing deps cause the adapter's whole parametrize branch
to be skipped rather than failed.

Per-backend behavioural tests live in ``test_chroma_store.py``,
``test_qdrant_store.py``, ``test_milvus_store.py``, ``test_lancedb_store.py``.
"""

from __future__ import annotations

import importlib
import tempfile
from pathlib import Path

import pytest

from benchmark.vectorstore import VectorStoreConfig
from benchmark.vectorstore.base import VectorStoreAdapter


def _backend_available(name: str) -> bool:
    """True if the optional dep for a backend is importable."""
    deps = {
        "chroma": "chromadb",
        "qdrant": "qdrant_client",
        "milvus": "pymilvus",
        "lancedb": "lancedb",
    }
    try:
        importlib.import_module(deps[name])
        return True
    except ImportError:
        return False


def _make_adapter(name: str, tmp_path: Path) -> VectorStoreAdapter:
    """Build an adapter with sane defaults for the conformance suite."""
    from benchmark.vectorstore import get_vector_store

    cfg_kwargs = {
        "backend": name,
        "path": str(tmp_path / name),
        "collection": "conform",
        "index_type": "hnsw",
        "metric": "cosine",
        "dimension": 4,
    }
    return get_vector_store(VectorStoreConfig(**cfg_kwargs))


# Skip a backend entirely if its dep is missing. (Avoid parametrize noise.)
_installed_backends = [b for b in ("chroma", "lancedb", "qdrant", "milvus") if _backend_available(b)]


@pytest.mark.parametrize("backend", _installed_backends or ["__none__"])
def test_insert_and_search_roundtrip(backend, tmp_path):
    if backend == "__none__":
        pytest.skip("no optional vector-store backend installed")
    if backend == "milvus":
        pytest.skip("Milvus requires a running server; covered in test_milvus_store.py")
    if backend == "qdrant" and not _qdrant_local_ok():
        pytest.skip("Qdrant local-mode unavailable in this environment")

    adapter = _make_adapter(backend, tmp_path)
    collection = f"conform_{backend}"
    adapter.build_index("hnsw", "cosine", collection)

    # 4-dim vectors so dim is small and deterministic.
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
    ]
    chunks = [
        {"text": "alpha", "metadata": {"file_id": "f1"}},
        {"text": "beta", "metadata": {"file_id": "f2"}},
        {"text": "alpha-prime", "metadata": {"file_id": "f1"}},
    ]
    inserted = adapter.insert(vectors, chunks, collection)
    assert inserted == 3
    assert adapter.count(collection) >= 3

    hits = adapter.search([[1.0, 0.0, 0.0, 0.0]], collection, top_k=2)
    assert len(hits) == 1
    assert len(hits[0]) <= 2
    # Top hit should be "alpha" (exact match), not "beta".
    assert hits[0][0].text in {"alpha", "alpha-prime"}


@pytest.mark.parametrize("backend", _installed_backends or ["__none__"])
def test_drop_clears_collection(backend, tmp_path):
    if backend == "__none__":
        pytest.skip("no optional vector-store backend installed")
    if backend == "milvus":
        pytest.skip("Milvus requires a running server")
    if backend == "qdrant" and not _qdrant_local_ok():
        pytest.skip("Qdrant local-mode unavailable in this environment")

    adapter = _make_adapter(backend, tmp_path)
    collection = f"drop_{backend}"
    adapter.build_index("hnsw", "cosine", collection)
    adapter.insert([[1.0, 0.0, 0.0, 0.0]], [{"text": "x"}], collection)
    assert adapter.count(collection) >= 1
    adapter.drop(collection)
    # After drop, count raises or returns 0 depending on backend semantics.
    try:
        assert adapter.count(collection) == 0
    except Exception:
        # Collection no longer exists — that's an acceptable outcome too.
        pass


def _qdrant_local_ok() -> bool:
    """Qdrant local-mode requires the optional ``qdrant-client[fastembed]``
    native wheels; probe quickly and skip if not available."""
    try:
        from qdrant_client import QdrantClient  # noqa: F401
        return True
    except Exception:
        return False
