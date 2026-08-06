"""Tests for the Qdrant adapter.

Skipped entirely when ``qdrant-client`` is not installed.
"""

from __future__ import annotations

import pytest

qdrant = pytest.importorskip("qdrant_client")

from benchmark.vectorstore import VectorStoreConfig, get_vector_store
from benchmark.vectorstore.qdrant_store import QdrantVectorStore


@pytest.fixture()
def adapter(tmp_path):
    return get_vector_store(
        VectorStoreConfig(
            backend="qdrant",
            path=str(tmp_path / "qdrant"),
            dimension=4,
        )
    )


def test_qdrant_build_index_and_search_roundtrip(adapter):
    adapter.build_index("hnsw", "cosine", "cq")
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    chunks = [
        {"text": "alpha", "metadata": {"file_id": "f1"}},
        {"text": "beta", "metadata": {"file_id": "f2"}},
    ]
    n = adapter.insert(vectors, chunks, "cq")
    assert n == 2
    assert adapter.count("cq") == 2

    hits = adapter.search([[1.0, 0.0, 0.0, 0.0]], "cq", top_k=1)
    assert len(hits) == 1
    assert len(hits[0]) == 1
    assert hits[0][0].text == "alpha"


def test_qdrant_build_index_requires_dimension(tmp_path):
    a = QdrantVectorStore(db_type="qdrant", db_path=str(tmp_path / "q2"))
    with pytest.raises(ValueError, match="dimension"):
        a.build_index("hnsw", "cosine", "no_dim")


def test_qdrant_build_index_rejects_unsupported_index(adapter):
    with pytest.raises(NotImplementedError, match="hnsw"):
        adapter.build_index("flat", "cosine", "x")


def test_qdrant_drop(adapter):
    adapter.build_index("hnsw", "cosine", "dropme")
    adapter.insert([[1.0, 0.0, 0.0, 0.0]], [{"text": "x"}], "dropme")
    assert adapter.count("dropme") == 1
    adapter.drop("dropme")
    # After drop, count should raise.
    with pytest.raises(RuntimeError):
        adapter.count("dropme")


def test_qdrant_lazy_import_error(monkeypatch):
    """When qdrant_client is uninstalled, the adapter raises ImportError."""
    import sys

    # Remove any cached import.
    for mod in list(sys.modules):
        if mod.startswith("qdrant_client"):
            monkeypatch.setitem(sys.modules, mod, None)
    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    with pytest.raises(ImportError, match="qdrant-client"):
        QdrantVectorStore(db_type="qdrant", db_path="/tmp/never")
