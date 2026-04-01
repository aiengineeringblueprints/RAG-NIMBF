"""Tests for benchmark.embedding — embedding model factory."""

from unittest.mock import patch, MagicMock

from langchain_ollama import OllamaEmbeddings

from benchmark.embedding import get_embedding_model


class TestGetEmbeddingModel:
    def test_returns_ollama_embeddings(self):
        model = get_embedding_model("nomic-embed-text:latest", "http://localhost:11434")
        assert isinstance(model, OllamaEmbeddings)

    def test_passes_model_name(self):
        model = get_embedding_model("nomic-embed-text:latest", "http://localhost:11434")
        assert model.model == "nomic-embed-text:latest"

    def test_passes_base_url(self):
        model = get_embedding_model("nomic-embed-text:latest", "http://server:11434")
        assert model.base_url == "http://server:11434"

    def test_without_api_key_no_headers(self):
        model = get_embedding_model("test-model", "http://localhost:11434")
        # OllamaEmbeddings doesn't have a 'headers' attribute by default
        # but the constructor should succeed without error
        assert isinstance(model, OllamaEmbeddings)

    def test_with_api_key_passes_client_kwargs(self):
        model = get_embedding_model("test-model", "http://localhost:11434", api_key="mytoken")
        assert model.model == "test-model"
        # Verify client_kwargs contains the auth header
        assert model.client_kwargs == {"headers": {"Authorization": "Bearer mytoken"}}
