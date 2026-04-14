"""Tests for benchmark.retrieval — vector store, cache, MMR, and HyDE logic."""

import hashlib
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from benchmark.retrieval import (
    _cache_key,
    cleanup_collection,
    clear_cache,
    retrieve,
    expand_query_with_hyde,
)


class TestCacheKey:
    def test_deterministic(self):
        k1 = _cache_key("nomic-embed-text:latest", 1000, 200, "recursive")
        k2 = _cache_key("nomic-embed-text:latest", 1000, 200, "recursive")
        assert k1 == k2

    def test_different_params_different_keys(self):
        k1 = _cache_key("model-a", 1000, 200, "recursive")
        k2 = _cache_key("model-b", 1000, 200, "recursive")
        assert k1 != k2

    def test_is_sha256_hex(self):
        k = _cache_key("model", 1000, 200, "recursive")
        assert len(k) == 64  # SHA-256 hex digest length
        assert all(c in "0123456789abcdef" for c in k)


class TestRetrieve:
    def _mock_store(self):
        store = MagicMock()
        store.similarity_search.return_value = [
            Document(page_content="doc1"),
            Document(page_content="doc2"),
        ]
        store.max_marginal_relevance_search.return_value = [
            Document(page_content="mmr1"),
            Document(page_content="mmr2"),
        ]
        return store

    def test_similarity_default(self):
        store = self._mock_store()
        results = retrieve(store, "query", 3)
        store.similarity_search.assert_called_once_with("query", k=3)
        assert len(results) == 2

    def test_similarity_explicit(self):
        store = self._mock_store()
        results = retrieve(store, "query", 5, retrieval_strategy="similarity")
        store.similarity_search.assert_called_once_with("query", k=5)

    def test_mmr_with_fetch_k(self):
        store = self._mock_store()
        results = retrieve(
            store, "query", 3,
            retrieval_strategy="mmr", fetch_k=20, mmr_lambda=0.7,
        )
        store.max_marginal_relevance_search.assert_called_once_with(
            "query", k=3, fetch_k=20, lambda_mult=0.7,
        )
        assert results[0].page_content == "mmr1"

    def test_mmr_fetch_k_defaults_to_top_k_times_4(self):
        store = self._mock_store()
        retrieve(store, "query", 5, retrieval_strategy="mmr")
        store.max_marginal_relevance_search.assert_called_once_with(
            "query", k=5, fetch_k=20, lambda_mult=0.5,
        )

    def test_mmr_respects_custom_fetch_k(self):
        store = self._mock_store()
        retrieve(store, "query", 5, retrieval_strategy="mmr", fetch_k=50)
        store.max_marginal_relevance_search.assert_called_once_with(
            "query", k=5, fetch_k=50, lambda_mult=0.5,
        )


class TestHyDE:
    def test_returns_hypothetical_answer(self):
        llm = MagicMock()
        response = MagicMock()
        response.content = "The answer is 42."
        llm.invoke.return_value = response

        result = expand_query_with_hyde(llm, "What is the answer?")
        assert result == "The answer is 42."
        llm.invoke.assert_called_once()

    def test_falls_back_to_question_on_empty_response(self):
        llm = MagicMock()
        response = MagicMock()
        response.content = ""
        llm.invoke.return_value = response

        result = expand_query_with_hyde(llm, "What is the answer?")
        assert result == "What is the answer?"

    def test_falls_back_to_question_on_none_content(self):
        llm = MagicMock()
        response = MagicMock()
        response.content = None
        llm.invoke.return_value = response

        result = expand_query_with_hyde(llm, "What is the answer?")
        assert result == "What is the answer?"

    def test_falls_back_to_question_on_exception(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM unavailable")

        result = expand_query_with_hyde(llm, "What is the answer?")
        assert result == "What is the answer?"


class TestCleanupCollection:
    @patch("benchmark.retrieval._get_client")
    def test_deletes_existing_collection(self, mock_get_client):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_client.list_collections.return_value = [mock_collection]
        mock_get_client.return_value = mock_client

        cleanup_collection("test_collection")

        mock_client.delete_collection.assert_called_once_with("test_collection")

    @patch("benchmark.retrieval._get_client")
    def test_noop_for_nonexistent_collection(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.list_collections.return_value = []
        mock_get_client.return_value = mock_client

        cleanup_collection("nonexistent")

        mock_client.delete_collection.assert_not_called()


class TestClearCache:
    @patch("benchmark.retrieval._get_client")
    def test_clears_cache_and_collections(self, mock_get_client):
        import benchmark.retrieval as ret_mod

        mock_client = MagicMock()
        mock_client.list_collections.return_value = []
        mock_get_client.return_value = mock_client

        # Manually populate the cache
        ret_mod._vector_store_cache["test_key"] = MagicMock()

        clear_cache()

        assert len(ret_mod._vector_store_cache) == 0
