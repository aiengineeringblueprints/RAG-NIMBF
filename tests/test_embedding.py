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


class TestResilientOllamaEmbeddings:
    """Guards against Ollama embedding-runner crashes (empty/oversized input)."""

    def _model(self):
        return get_embedding_model("test-model", "http://localhost:11434")

    def test_sanitize_replaces_empty_string(self):
        assert self._model()._sanitize("") == " "

    def test_sanitize_replaces_whitespace_only(self):
        assert self._model()._sanitize("  \n\t ") == " "

    def test_sanitize_truncates_oversized_text(self):
        from benchmark.embedding import _MAX_TEXT_CHARS

        assert len(self._model()._sanitize("a" * (_MAX_TEXT_CHARS + 100))) == _MAX_TEXT_CHARS

    def test_sanitize_keeps_normal_text(self):
        assert self._model()._sanitize("hello world") == "hello world"

    @patch.object(OllamaEmbeddings, "embed_documents")
    def test_embed_documents_sanitizes_batch(self, mock_embed):
        mock_embed.return_value = [[0.1], [0.2]]
        result = self._model().embed_documents(["ok text", "", "  "])
        assert result == [[0.1], [0.2]]
        assert mock_embed.call_args[0][1] == ["ok text", " ", " "]

    @patch.object(OllamaEmbeddings, "embed_query")
    def test_embed_query_sanitizes(self, mock_embed):
        mock_embed.return_value = [0.1]
        self._model().embed_query("")
        assert mock_embed.call_args[0][1] == " "

    @patch("time.sleep")
    @patch.object(OllamaEmbeddings, "embed_documents")
    def test_embed_documents_retries_on_connection_reset(self, mock_embed, mock_sleep):
        import ollama

        mock_embed.side_effect = [
            ollama.ResponseError(
                'Post "http://127.0.0.1:45887/tokenize": read: connection reset by peer',
                status_code=400,
            ),
            [[0.1]],
        ]
        result = self._model().embed_documents(["text"])
        assert result == [[0.1]]
        assert mock_embed.call_count == 2

    @patch("time.sleep")
    @patch.object(OllamaEmbeddings, "embed_documents")
    def test_embed_documents_raises_after_max_retries(self, mock_embed, mock_sleep):
        import httpx
        import ollama

        mock_embed.side_effect = ollama.ResponseError("boom", status_code=400)
        with pytest.raises(ollama.ResponseError):
            self._model().embed_documents(["text"])
        from benchmark.embedding import _EMBED_RETRIES

        assert mock_embed.call_count == _EMBED_RETRIES

    @patch("time.sleep")
    @patch.object(OllamaEmbeddings, "embed_documents")
    def test_embed_documents_retries_on_httpx_error(self, mock_embed, mock_sleep):
        import httpx

        mock_embed.side_effect = [httpx.RemoteProtocolError("peer closed"), [[0.1]]]
        assert self._model().embed_documents(["text"]) == [[0.1]]


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
