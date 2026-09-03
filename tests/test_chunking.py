"""Tests for benchmark.chunking — text splitting strategies."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from benchmark.chunking import get_chunker, chunk_documents, STRATEGY_MAP


class TestGetChunker:
    def test_recursive_strategy(self):
        chunker = get_chunker("recursive", 100, 20)
        assert chunker is not None

    def test_character_strategy(self):
        chunker = get_chunker("character", 100, 20)
        assert chunker is not None

    def test_token_strategy(self):
        chunker = get_chunker("token", 100, 20)
        assert chunker is not None

    def test_markdown_strategy(self):
        chunker = get_chunker("markdown", 100, 20)
        assert chunker is not None

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker("nonexistent", 100, 20)

    def test_paragraph_strategy_returns_chunker(self):
        chunker = get_chunker("paragraph", 500, 100)
        assert chunker is not None

    def test_all_strategies_in_map(self):
        expected = {"recursive", "character", "token", "markdown", "text", "transformers"}
        assert set(STRATEGY_MAP.keys()) == expected


class TestSemanticChunker:
    def test_semantic_strategy_creates_chunker(self):
        """Semantic chunking is handled separately — verify get_chunker routes correctly."""
        mock_emb = MagicMock()
        mock_instance = MagicMock()

        with patch(
            "langchain_experimental.text_splitter.SemanticChunker",
            return_value=mock_instance,
        ) as MockSemantic:
            chunker = get_chunker(
                "semantic", 500, 100,
                embeddings=mock_emb,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95,
            )

            MockSemantic.assert_called_once_with(
                embeddings=mock_emb,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95,
            )
            assert chunker is mock_instance

    def test_semantic_without_embeddings_raises(self):
        with pytest.raises(ValueError, match="embeddings"):
            get_chunker("semantic", 500, 100)

    def test_semantic_with_custom_breakpoint(self):
        mock_emb = MagicMock()

        with patch(
            "langchain_experimental.text_splitter.SemanticChunker",
            return_value=MagicMock(),
        ) as MockSemantic:
            get_chunker(
                "semantic", 500, 100,
                embeddings=mock_emb,
                breakpoint_threshold_type="standard_deviation",
                breakpoint_threshold_amount=80,
            )

            MockSemantic.assert_called_once_with(
                embeddings=mock_emb,
                breakpoint_threshold_type="standard_deviation",
                breakpoint_threshold_amount=80,
            )


class TestChunkDocuments:
    def test_basic_chunking(self):
        chunker = get_chunker("character", 50, 10)
        docs = [
            {
                "context": "This is a test document with some content that should be split into chunks.",
                "metadata": {"source": "test"},
            }
        ]
        result = chunk_documents(chunker, docs)
        assert len(result) >= 1
        assert all(isinstance(d, Document) for d in result)

    def test_list_context_joined(self):
        chunker = get_chunker("character", 200, 20)
        docs = [
            {
                "context": ["Line one.", "Line two.", "Line three."],
            }
        ]
        result = chunk_documents(chunker, docs)
        assert len(result) >= 1
        # The joined text should appear in at least one chunk
        all_text = " ".join(d.page_content for d in result)
        assert "Line one." in all_text

    def test_metadata_preserved(self):
        chunker = get_chunker("character", 200, 20)
        docs = [{"context": "Short text.", "metadata": {"source": "test"}}]
        result = chunk_documents(chunker, docs)
        assert all(d.metadata.get("source") == "test" for d in result)

    def test_empty_documents(self):
        chunker = get_chunker("character", 100, 20)
        result = chunk_documents(chunker, [])
        assert result == []


class TestParagraphChunking:
    def test_paragraph_strategy_emits_one_chunk_per_document(self):
        chunker = get_chunker("paragraph", 500, 100)
        docs = [
            {"context": f"Paragraph {i} with plenty of text to exceed min length." * 3,
             "metadata": {"title": f"T{i}", "doc_id": f"d{i}"}}
            for i in range(3)
        ]
        result = chunk_documents(chunker, docs)
        assert len(result) == 3
        assert result[0].metadata["title"] == "T0"
        assert result[2].metadata["doc_id"] == "d2"

    def test_paragraph_strategy_never_splits(self):
        chunker = get_chunker("paragraph", 10, 0)  # tiny size must not matter
        docs = [{"context": "x" * 500, "metadata": {}}]
        result = chunk_documents(chunker, docs)
        assert len(result) == 1
        assert len(result[0].page_content) == 500
