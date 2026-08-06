"""Tests for the LanceDB adapter.

Skipped entirely when ``lancedb`` is not installed.
"""

from __future__ import annotations

import pytest

lancedb = pytest.importorskip("lancedb")

from benchmark.vectorstore import VectorStoreConfig, get_vector_store
from benchmark.vectorstore.lancedb_store import LanceDBVectorStore


@pytest.fixture()
def adapter(tmp_path):
    return get_vector_store(
        VectorStoreConfig(
            backend="lancedb",
            path=str(tmp_path / "lance"),
            dimension=4,
        )
    )


def test_lancedb_build_index_and_search_roundtrip(adapter):
    adapter.build_index("hnsw", "cosine", "cl")
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    chunks = [
        {"text": "alpha", "metadata": {"file_id": "f1"}},
        {"text": "beta", "metadata": {"file_id": "f2"}},
    ]
    n = adapter.insert(vectors, chunks, "cl")
    assert n == 2
    assert adapter.count("cl") == 2

    hits = adapter.search([[1.0, 0.0, 0.0, 0.0]], "cl", top_k=1)
    assert len(hits) == 1
    assert len(hits[0]) == 1
    assert hits[0][0].text == "alpha"


def test_lancedb_build_index_rejects_unsupported_index(adapter):
    with pytest.raises(NotImplementedError):
        adapter.build_index("annoy", "cosine", "x")


def test_lancedb_drop(adapter):
    adapter.build_index("hnsw", "cosine", "dropme")
    adapter.insert([[1.0, 0.0, 0.0, 0.0]], [{"text": "x"}], "dropme")
    assert adapter.count("dropme") == 1
    adapter.drop("dropme")
    with pytest.raises(RuntimeError):
        adapter.count("dropme")


def test_lancedb_delete_by_file_id(adapter):
    adapter.build_index("hnsw", "cosine", "del")
    adapter.insert(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [
            {"text": "x", "metadata": {"file_id": "rm"}},
            {"text": "y", "metadata": {"file_id": "keep"}},
        ],
        "del",
    )
    assert adapter.count("del") == 2
    adapter.delete("rm")
    # One row remains.
    assert adapter.count("del") == 1
