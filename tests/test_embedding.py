"""Tests for benchmark.embedding — embedding model factory."""

import asyncio
import httpx
import ollama
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

    def test_class_is_reused_across_calls(self):
        cls_a = type(get_embedding_model("m", "http://localhost:11434"))
        cls_b = type(get_embedding_model("m", "http://localhost:11434"))
        assert cls_a is cls_b


class TestResilientOllamaEmbeddings:
    """Guards against Ollama embedding-runner crashes (empty/oversized input)."""

    def _model(self):
        return get_embedding_model("test-model", "http://localhost:11434")

    def test_sanitize_replaces_empty_string_with_non_whitespace(self):
        # Whitespace-only input is what crashes the runner — the placeholder
        # itself must not be whitespace-only.
        sanitized = self._model()._sanitize("")
        assert sanitized
        assert sanitized.strip()

    def test_sanitize_replaces_whitespace_only(self):
        sanitized = self._model()._sanitize("  \n\t ")
        assert sanitized
        assert sanitized.strip()

    def test_sanitize_truncates_oversized_text(self):
        from benchmark.embedding import _max_text_chars

        assert len(self._model()._sanitize("a" * (_max_text_chars() + 100))) == _max_text_chars()

    def test_sanitize_keeps_normal_text(self):
        assert self._model()._sanitize("hello world") == "hello world"

    def test_sanitize_respects_env_override(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_MAX_TEXT_CHARS", "10")
        assert len(self._model()._sanitize("a" * 50)) == 10

    def test_sanitize_rejects_invalid_env_value(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_MAX_TEXT_CHARS", "not-a-number")
        from benchmark.embedding import _DEFAULT_MAX_TEXT_CHARS, _max_text_chars

        assert _max_text_chars() == _DEFAULT_MAX_TEXT_CHARS

    @patch.object(OllamaEmbeddings, "embed_documents")
    def test_embed_documents_sanitizes_batch(self, mock_embed):
        mock_embed.side_effect = lambda self, batch: [[0.1] for _ in batch]
        result = self._model().embed_documents(["ok text", "", "  "])
        assert result == [[0.1], [0.1], [0.1]]
        assert mock_embed.call_args[0][1] == ["ok text", "empty", "empty"]

    @patch.object(OllamaEmbeddings, "embed_query")
    def test_embed_query_sanitizes(self, mock_embed):
        mock_embed.return_value = [0.1]
        self._model().embed_query("")
        assert mock_embed.call_args[0][1] == "empty"

    @patch("time.sleep")
    @patch.object(OllamaEmbeddings, "embed_documents")
    def test_embed_documents_retries_on_connection_reset(self, mock_embed, mock_sleep):
        mock_embed.side_effect = [
            ollama.ResponseError(
                'Post "http://127.0.0.1:45887/tokenize": read: connection reset by peer',
                status_code=500,
            ),
            [[0.1]],
        ]
        result = self._model().embed_documents(["text"])
        assert result == [[0.1]]
        assert mock_embed.call_count == 2

    @patch("time.sleep")
    @patch.object(OllamaEmbeddings, "embed_documents")
    def test_embed_documents_retries_on_builtin_connection_error(self, mock_embed, mock_sleep):
        # The ollama client maps httpx.ConnectError to the built-in
        # ConnectionError (e.g. server restarting after a crash).
        mock_embed.side_effect = [ConnectionError("connection refused"), [[0.1]]]
        assert self._model().embed_documents(["text"]) == [[0.1]]
        assert mock_embed.call_count == 2

    @patch("time.sleep")
    @patch.object(OllamaEmbeddings, "embed_documents")
    def test_embed_documents_raises_after_max_retries(self, mock_embed, mock_sleep):
        mock_embed.side_effect = ollama.ResponseError("boom", status_code=503)
        with pytest.raises(ollama.ResponseError):
            self._model().embed_documents(["text"])
        from benchmark.embedding import _EMBED_RETRIES

        assert mock_embed.call_count == _EMBED_RETRIES

    @patch("time.sleep")
    @patch.object(OllamaEmbeddings, "embed_documents")
    def test_embed_documents_retries_on_httpx_error(self, mock_embed, mock_sleep):
        mock_embed.side_effect = [httpx.RemoteProtocolError("peer closed"), [[0.1]]]
        assert self._model().embed_documents(["text"]) == [[0.1]]

    @patch("time.sleep")
    @patch.object(OllamaEmbeddings, "embed_documents")
    def test_embed_documents_no_retry_on_client_errors(self, mock_embed, mock_sleep):
        # 404 (model not found) / 400 will never succeed — fail fast.
        mock_embed.side_effect = ollama.ResponseError("model not found", status_code=404)
        with pytest.raises(ollama.ResponseError):
            self._model().embed_documents(["text"])
        assert mock_embed.call_count == 1
        mock_sleep.assert_not_called()

    @patch("time.sleep")
    @patch.object(OllamaEmbeddings, "embed_documents")
    def test_embed_documents_no_sleep_after_final_attempt(self, mock_embed, mock_sleep):
        from benchmark.embedding import _EMBED_RETRIES

        mock_embed.side_effect = ollama.ResponseError("boom", status_code=500)
        with pytest.raises(ollama.ResponseError):
            self._model().embed_documents(["text"])
        assert mock_sleep.call_count == _EMBED_RETRIES - 1

    @patch.object(OllamaEmbeddings, "embed_documents")
    def test_embed_documents_retries_each_batch_independently(self, mock_embed):
        # Only the second batch fails once; the first batch must not be
        # re-embedded afterwards.
        from benchmark.embedding import _EMBED_BATCH_SIZE

        texts = [f"t{i}" for i in range(_EMBED_BATCH_SIZE * 2)]
        responses = [[[0.0]] * _EMBED_BATCH_SIZE, ConnectionError("refused"), [[1.1]] * _EMBED_BATCH_SIZE]

        def side_effect(self, batch):
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        mock_embed.side_effect = side_effect
        with patch("time.sleep"):
            result = self._model().embed_documents(texts)
        assert len(result) == len(texts)
        assert mock_embed.call_count == 3  # batch1 ok, batch2 retry, batch2 ok
        assert mock_embed.call_args_list[0].args[1] == texts[:_EMBED_BATCH_SIZE]
        assert mock_embed.call_args_list[2].args[1] == texts[_EMBED_BATCH_SIZE:]

    @patch.object(OllamaEmbeddings, "embed_documents")
    def test_embed_documents_batches_large_inputs(self, mock_embed):
        from benchmark.embedding import _EMBED_BATCH_SIZE

        texts = [f"text {i}" for i in range(_EMBED_BATCH_SIZE * 3 + 1)]
        mock_embed.side_effect = lambda self, batch: [[0.0] for _ in batch]
        result = self._model().embed_documents(texts)
        assert len(result) == len(texts)
        # 3 full batches + 1 remainder = 4 requests
        assert mock_embed.call_count == 4
        batch_sizes = [len(call.args[1]) for call in mock_embed.call_args_list]
        assert batch_sizes == [_EMBED_BATCH_SIZE, _EMBED_BATCH_SIZE, _EMBED_BATCH_SIZE, 1]

    @patch.object(OllamaEmbeddings, "embed_documents")
    def test_embed_documents_preserves_order_across_batches(self, mock_embed):
        mock_embed.side_effect = lambda self, batch: [[float(len(t))] for t in batch]
        result = self._model().embed_documents(["a", "longer", "x"])
        assert result == [[1.0], [6.0], [1.0]]

    # ── Async paths (used by RAGAS via LangchainEmbeddingsWrapper) ──

    @patch.object(OllamaEmbeddings, "aembed_documents")
    def test_aembed_documents_sanitizes(self, mock_aembed):
        async def fake(self, batch):
            return [[0.1] for _ in batch]

        mock_aembed.side_effect = fake
        result = asyncio.run(self._model().aembed_documents(["ok", ""]))
        assert result == [[0.1], [0.1]]
        assert mock_aembed.call_args.args[1] == ["ok", "empty"]

    @patch.object(OllamaEmbeddings, "aembed_documents")
    def test_aembed_documents_batches_large_inputs(self, mock_aembed):
        from benchmark.embedding import _EMBED_BATCH_SIZE

        texts = [f"text {i}" for i in range(_EMBED_BATCH_SIZE * 2 + 1)]

        async def fake(self, batch):
            return [[0.0] for _ in batch]

        mock_aembed.side_effect = fake
        result = asyncio.run(self._model().aembed_documents(texts))
        assert len(result) == len(texts)
        assert mock_aembed.call_count == 3

    @patch("asyncio.sleep")
    @patch.object(OllamaEmbeddings, "aembed_documents")
    def test_aembed_documents_retries_on_builtin_connection_error(self, mock_aembed, mock_sleep):
        async def fake(self, batch):
            if mock_aembed.call_count == 1:
                raise ConnectionError("connection refused")
            return [[0.1]]

        mock_aembed.side_effect = fake
        assert asyncio.run(self._model().aembed_documents(["text"])) == [[0.1]]
        assert mock_aembed.call_count == 2

    @patch("asyncio.sleep")
    @patch.object(OllamaEmbeddings, "aembed_query")
    def test_aembed_query_sanitizes_and_retries(self, mock_aembed, mock_sleep):
        responses = [ollama.ResponseError("boom", status_code=503), [0.1]]

        async def fake(self, text):
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        mock_aembed.side_effect = fake
        result = asyncio.run(self._model().aembed_query(""))
        assert result == [0.1]
        assert mock_aembed.call_count == 2
        assert mock_aembed.call_args.args[1] == "empty"


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
