"""Tests for benchmark.reranker — reranker factory and CrossEncoderReranker."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from benchmark.reranker import CrossEncoderReranker, get_reranker


# ---------------------------------------------------------------------------
# get_reranker factory
# ---------------------------------------------------------------------------

class TestGetReranker:
    @patch("sentence_transformers.CrossEncoder")
    def test_huggingface_prefix(self, mock_ce_cls):
        mock_ce_cls.return_value = MagicMock()
        reranker = get_reranker("huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert isinstance(reranker, CrossEncoderReranker)
        mock_ce_cls.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def test_none_returns_none(self):
        assert get_reranker(None) is None

    def test_empty_string_returns_none(self):
        assert get_reranker("") is None

    def test_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported reranker provider 'ollama'"):
            get_reranker("ollama:some-model")


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------

class TestCrossEncoderReranker:
    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_sorts_by_score(self, mock_ce_cls):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])
        mock_ce_cls.return_value = mock_model

        reranker = CrossEncoderReranker("test-model")
        docs = [
            Document(page_content="low"),
            Document(page_content="high"),
            Document(page_content="mid"),
        ]
        result = reranker.rerank("query", docs, top_k=3)

        assert len(result) == 3
        assert result[0].page_content == "high"
        assert result[1].page_content == "mid"
        assert result[2].page_content == "low"

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_respects_top_k(self, mock_ce_cls):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5, 0.8, 0.3])
        mock_ce_cls.return_value = mock_model

        reranker = CrossEncoderReranker("test-model")
        docs = [Document(page_content=f"doc{i}") for i in range(5)]
        result = reranker.rerank("query", docs, top_k=2)

        assert len(result) == 2
        assert result[0].page_content == "doc1"
        assert result[1].page_content == "doc3"

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_empty_docs(self, mock_ce_cls):
        mock_ce_cls.return_value = MagicMock()
        reranker = CrossEncoderReranker("test-model")
        result = reranker.rerank("query", [], top_k=3)
        assert result == []
