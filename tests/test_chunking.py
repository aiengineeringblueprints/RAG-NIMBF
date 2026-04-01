"""Tests for benchmark.chunking — text splitting strategies."""

import pytest
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

    def test_all_strategies_in_map(self):
        expected = {"recursive", "character", "token", "markdown", "text", "transformers"}
        assert set(STRATEGY_MAP.keys()) == expected


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
