"""Tests for benchmark.embedding — embedding model factory."""

import pytest
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
        assert isinstance(model, OllamaEmbeddings)

    def test_with_api_key_passes_client_kwargs(self):
        model = get_embedding_model("test-model", "http://localhost:11434", api_key="mytoken")
        assert model.model == "test-model"
        assert model.client_kwargs == {"headers": {"Authorization": "Bearer mytoken"}}


class TestGetEmbeddingModelHuggingFace:
    @patch("langchain_huggingface.HuggingFaceEmbeddings")
    def test_returns_hf_embeddings(self, mock_hf_cls):
        mock_instance = MagicMock()
        mock_hf_cls.return_value = mock_instance

        model = get_embedding_model(
            "BAAI/bge-small-en-v1.5", "", provider="huggingface",
        )
        assert model is mock_instance
        mock_hf_cls.assert_called_once_with(model_name="BAAI/bge-small-en-v1.5")

    @patch("langchain_huggingface.HuggingFaceEmbeddings")
    def test_ignores_base_url_and_api_key(self, mock_hf_cls):
        mock_hf_cls.return_value = MagicMock()

        get_embedding_model(
            "BAAI/bge-small-en-v1.5",
            "http://should-be-ignored:11434",
            api_key="should-be-ignored",
            provider="huggingface",
        )
        mock_hf_cls.assert_called_once_with(model_name="BAAI/bge-small-en-v1.5")

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding provider 'openai'"):
            get_embedding_model("test", "http://localhost", provider="openai")
