"""Tests for the Chroma adapter (``benchmark.vectorstore.chroma_store``).

These run a real Chroma client against a temp directory so they exercise
insert/search/delete/count/drop end-to-end. The shared Chroma client is
patched out so we never touch the developer's ``.chroma/`` working dir.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

chromadb = pytest.importorskip("chromadb")

from benchmark.vectorstore import VectorStoreConfig, get_vector_store
from benchmark.vectorstore.base import VectorStoreConfig as VSC
from benchmark.vectorstore.chroma_store import ChromaVectorStore


@pytest.fixture()
def isolated_chroma(tmp_path, monkeypatch):
    """Force a fresh PersistentClient at a tmp path for each test."""
    import benchmark.retrieval as ret_mod

    # Reset the cached client so the new tmp_path is picked up.
    monkeypatch.setattr(ret_mod, "_chroma_client", None)
    monkeypatch.setattr(ret_mod, "_CHROMA_DIR", Path(tmp_path / "chroma"))
    # Now create the client explicitly at our tmp path.
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    monkeypatch.setattr(ret_mod, "_chroma_client", client)
    return client


def _adapter(isolated_chroma) -> ChromaVectorStore:
    return ChromaVectorStore(db_type="chroma", db_path=str(isolated_chroma), client=isolated_chroma)


def test_get_vector_store_defaults_to_chroma(isolated_chroma):
    adapter = get_vector_store(VSC(path=str(isolated_chroma)))
    assert isinstance(adapter, ChromaVectorStore)
    assert adapter.db_type == "chroma"


def test_build_index_creates_collection(isolated_chroma):
    adapter = _adapter(isolated_chroma)
    adapter.build_index("hnsw", "cosine", "col1")
    names = {c.name for c in isolated_chroma.list_collections()}
    assert "col1" in names


def test_build_index_rejects_unsupported_index_type(isolated_chroma):
    adapter = _adapter(isolated_chroma)
    with pytest.raises(NotImplementedError, match="hnsw"):
        adapter.build_index("ivf", "cosine", "col1")


def test_build_index_rejects_unknown_metric(isolated_chroma):
    adapter = _adapter(isolated_chroma)
    with pytest.raises(ValueError, match="metric"):
        adapter.build_index("hnsw", "manhattan", "col1")


def test_build_index_is_idempotent(isolated_chroma):
    adapter = _adapter(isolated_chroma)
    adapter.build_index("hnsw", "cosine", "col1")
    # Calling again with same params is a no-op.
    adapter.build_index("hnsw", "cosine", "col1")
    names = {c.name for c in isolated_chroma.list_collections()}
    assert names == {"col1"}


def test_insert_then_search_returns_relevant_text(isolated_chroma):
    adapter = _adapter(isolated_chroma)
    adapter.build_index("hnsw", "cosine", "docs")
    vectors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.95, 0.05, 0.0],
    ]
    chunks = [
        {"text": "alpha", "metadata": {"file_id": "f1"}},
        {"text": "beta", "metadata": {"file_id": "f2"}},
        {"text": "alpha-like", "metadata": {"file_id": "f1"}},
    ]
    n = adapter.insert(vectors, chunks, "docs")
    assert n == 3
    assert adapter.count("docs") == 3

    hits = adapter.search([[1.0, 0.0, 0.0]], "docs", top_k=2)
    assert len(hits) == 1
    assert {h.text for h in hits[0]} <= {"alpha", "alpha-like", "beta"}
    # Top hit must be one of the alpha docs, not beta.
    assert hits[0][0].text in {"alpha", "alpha-like"}


def test_insert_mismatched_lengths_raises(isolated_chroma):
    adapter = _adapter(isolated_chroma)
    with pytest.raises(ValueError, match="same length"):
        adapter.insert([[1.0, 0.0]], [{"text": "colA"}, {"text": "b"}], "coll")


def test_insert_empty_returns_zero(isolated_chroma):
    adapter = _adapter(isolated_chroma)
    assert adapter.insert([], [], "coll") == 0


def test_search_unknown_collection_raises(isolated_chroma):
    adapter = _adapter(isolated_chroma)
    with pytest.raises(RuntimeError, match="not found"):
        adapter.search([[1.0, 0.0]], "nope", 5)


def test_search_empty_vectors_returns_empty(isolated_chroma):
    adapter = _adapter(isolated_chroma)
    assert adapter.search([], "anything", 5) == []


def test_delete_removes_rows_matching_file_id(isolated_chroma):
    adapter = _adapter(isolated_chroma)
    adapter.build_index("hnsw", "cosine", "docs")
    adapter.insert(
        [[1.0, 0.0], [0.0, 1.0]],
        [
            {"text": "x", "metadata": {"file_id": "to_remove"}},
            {"text": "y", "metadata": {"file_id": "keep"}},
        ],
        "docs",
    )
    assert adapter.count("docs") == 2
    adapter.delete("to_remove")
    assert adapter.count("docs") == 1


def test_drop_collection(isolated_chroma):
    adapter = _adapter(isolated_chroma)
    adapter.build_index("hnsw", "cosine", "docs")
    adapter.drop("docs")
    names = {c.name for c in isolated_chroma.list_collections()}
    assert "docs" not in names


def test_drop_unknown_is_safe(isolated_chroma):
    adapter = _adapter(isolated_chroma)
    # Should not raise.
    adapter.drop("never_existed")


def test_count_total_when_no_collection(isolated_chroma):
    adapter = _adapter(isolated_chroma)
    adapter.build_index("hnsw", "cosine", "colA")
    adapter.insert([[1.0, 0.0]], [{"text": "x"}], "colA")
    # No collection_name -> sum across all.
    assert adapter.count() == 1


def test_count_unknown_collection_raises(isolated_chroma):
    adapter = _adapter(isolated_chroma)
    with pytest.raises(RuntimeError, match="not found"):
        adapter.count("missing")


def test_db_type_mismatch_raises(isolated_chroma):
    with pytest.raises(ValueError, match="db_type"):
        ChromaVectorStore(db_type="qdrant")


def test_get_vector_store_factory_returns_chroma(tmp_path):
    cfg = VectorStoreConfig(backend="chroma", path=str(tmp_path / "coll"))
    adapter = get_vector_store(cfg)
    assert isinstance(adapter, ChromaVectorStore)


def test_get_vector_store_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown vector_store backend"):
        get_vector_store(VSC(backend="postgres"))


def test_factory_lazy_imports_optional_deps(monkeypatch):
    """``get_vector_store`` for chroma must NOT import qdrant_client / pymilvus."""
    import sys

    # Ensure the optional deps are absent from sys.modules during the call.
    for mod in ("qdrant_client", "pymilvus"):
        monkeypatch.setitem(sys.modules, mod, None)
    cfg = VectorStoreConfig(backend="chroma", path=".")
    adapter = get_vector_store(cfg)
    assert adapter.db_type == "chroma"


def test_get_vector_store_accepts_dict(tmp_path):
    adapter = get_vector_store({"backend": "chroma", "path": str(tmp_path / "x")})
    assert isinstance(adapter, ChromaVectorStore)


def test_get_vector_store_accepts_none():
    adapter = get_vector_store(None)
    assert isinstance(adapter, ChromaVectorStore)


def test_vector_store_config_from_dict_ignores_unknown_keys():
    cfg = VectorStoreConfig.from_dict({"backend": "chroma", "bogus": 1})
    assert cfg.backend == "chroma"
    assert not hasattr(cfg, "bogus")


def test_vector_store_config_from_dict_none_returns_default():
    cfg = VectorStoreConfig.from_dict(None)
    assert cfg.backend == "chroma"
    assert cfg.metric == "cosine"
