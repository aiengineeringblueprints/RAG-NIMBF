"""Tests for benchmark.retrieval — vector store and cache logic."""

import hashlib
from unittest.mock import patch, MagicMock

from benchmark.retrieval import _cache_key, cleanup_collection, clear_cache


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
