"""Tests for the shared index cache key used by internal and injected paths."""

from typing import Any
from unittest.mock import patch

from benchmark.retrieval import index_cache_key
from benchmark.adapters.components import _make_retriever_factory


class TestIndexCacheKeyContract:
    def test_public_function_is_deterministic(self):
        kwargs: dict[str, Any] = dict(
            embedding_model_name="model",
            chunk_size=1000,
            chunk_overlap=200,
            chunking_strategy="recursive",
            dataset_name="ragbench",
            embedding_provider="ollama",
            dataset_subset="FinQA",
            dataset_sample_size=50,
            corpus_fingerprint="abc",
            vector_db_backend="chroma",
        )
        assert index_cache_key(**kwargs) == index_cache_key(**kwargs)

    def test_dataset_name_changes_key(self):
        base: dict[str, Any] = dict(
            embedding_model_name="model",
            chunk_size=1000,
            chunk_overlap=200,
            chunking_strategy="recursive",
        )
        k1 = index_cache_key(**base, dataset_name="ragbench")
        k2 = index_cache_key(**base, dataset_name="hotpotqa")
        assert k1 != k2

    def test_sample_size_changes_key(self):
        base: dict[str, Any] = dict(
            embedding_model_name="model",
            chunk_size=1000,
            chunk_overlap=200,
            chunking_strategy="recursive",
            dataset_name="ragbench",
        )
        k1 = index_cache_key(**base, dataset_sample_size=50)
        k2 = index_cache_key(**base, dataset_sample_size=100)
        assert k1 != k2

    def test_subset_fingerprint_provider_and_backend_change_key(self):
        base: dict[str, Any] = dict(
            embedding_model_name="model",
            chunk_size=1000,
            chunk_overlap=200,
            chunking_strategy="recursive",
            dataset_name="ragbench",
        )
        variants = [
            {"dataset_subset": "FinQA"},
            {"dataset_subset": "HotpotQA"},
            {"corpus_fingerprint": "aaa"},
            {"corpus_fingerprint": "bbb"},
            {"embedding_provider": "ollama"},
            {"embedding_provider": "huggingface"},
            {"vector_db_backend": "chroma"},
            {"vector_db_backend": "lancedb"},
        ]
        keys = [index_cache_key(**base, **v) for v in variants]
        assert len(set(keys)) == len(keys)


class TestInjectedFactoryUsesSharedKey:
    def _config(self):
        class Cfg:
            embedding_model = "model"
            embedding_provider = "ollama"
            chunk_size = 1000
            chunk_overlap = 200
            chunking_strategy = "recursive"
            dataset_name = "ragbench"
            dataset_subset = "FinQA"
            dataset_sample_size = 50
            corpus_fingerprint = "abc"
            vector_db_backend = "chroma"
            lancedb_path = ".lancedb"

            def embedding_base_url(self):
                return "http://localhost"

            def embedding_api_key(self):
                return "key"

        return Cfg()

    def test_factory_passes_shared_cache_key_and_collection_name(self):
        chunks = []
        expected_key = index_cache_key(
            embedding_model_name="model",
            chunk_size=1000,
            chunk_overlap=200,
            chunking_strategy="recursive",
            dataset_name="ragbench",
            embedding_provider="ollama",
            dataset_subset="FinQA",
            dataset_sample_size=50,
            corpus_fingerprint="abc",
            vector_db_backend="chroma",
        )
        with patch(
            "benchmark.retrieval.build_vector_store"
        ) as mock_build, patch(
            "benchmark.retrieval.corpus_fingerprint_from_documents",
            return_value="abc",
        ):
            factory = _make_retriever_factory(self._config())
            factory(chunks)

        kwargs = mock_build.call_args.kwargs
        assert kwargs["cache_key"] == expected_key
        assert kwargs["collection_name"] == f"rag_chroma_{expected_key[:24]}"

    def test_factory_collection_name_has_no_injected_marker(self):
        with patch(
            "benchmark.retrieval.build_vector_store"
        ) as mock_build, patch(
            "benchmark.retrieval.corpus_fingerprint_from_documents",
            return_value="abc",
        ):
            factory = _make_retriever_factory(self._config())
            factory([])

        collection_name = mock_build.call_args.kwargs["collection_name"]
        assert "injected" not in collection_name
