"""Tests for the Milvus adapter.

Skipped entirely when ``pymilvus`` is not installed. The behavioural
round-trip tests also require a running Milvus server and are skipped if
the connection cannot be established.
"""

from __future__ import annotations

import socket

import pytest

pymilvus = pytest.importorskip("pymilvus")

from benchmark.vectorstore import VectorStoreConfig, get_vector_store
from benchmark.vectorstore.milvus_store import MilvusVectorStore


def _milvus_reachable() -> bool:
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect(("localhost", 19530))
        return True
    except OSError:
        return False
    finally:
        s.close()


@pytest.fixture()
def adapter(tmp_path):
    if not _milvus_reachable():
        pytest.skip("Milvus server not reachable at localhost:19530")
    return get_vector_store(
        VectorStoreConfig(
            backend="milvus",
            path=str(tmp_path / "milvus"),
            host="localhost",
            port=19530,
            dimension=4,
        )
    )


def test_milvus_build_index_and_search_roundtrip(adapter):
    adapter.build_index("hnsw", "cosine", "cm")
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    chunks = [
        {"text": "alpha", "metadata": {"file_id": "f1"}},
        {"text": "beta", "metadata": {"file_id": "f2"}},
    ]
    n = adapter.insert(vectors, chunks, "cm")
    assert n == 2
    assert adapter.count("cm") >= 2

    hits = adapter.search([[1.0, 0.0, 0.0, 0.0]], "cm", top_k=1)
    assert len(hits) == 1
    assert hits[0][0].text == "alpha"

    adapter.drop("cm")


def test_milvus_build_index_requires_dimension():
    if not _milvus_reachable():
        pytest.skip("Milvus server not reachable")
    a = MilvusVectorStore(db_type="milvus", dimension=None)
    with pytest.raises(ValueError, match="dimension"):
        a.build_index("hnsw", "cosine", "no_dim")


def test_milvus_build_index_rejects_unsupported_index():
    if not _milvus_reachable():
        pytest.skip("Milvus server not reachable")
    a = MilvusVectorStore(db_type="milvus", dimension=4)
    with pytest.raises(NotImplementedError):
        a.build_index("hnsw-bit", "cosine", "bad")


def test_milvus_connection_error(monkeypatch):
    """Point at an unreachable port and confirm we get ConnectionError."""
    with pytest.raises(ConnectionError):
        MilvusVectorStore(db_type="milvus", host="localhost", port=1, dimension=4)
