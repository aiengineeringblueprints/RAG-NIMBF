"""Reranker module — re-sorts retrieved documents using a cross-encoder model."""
from __future__ import annotations

from typing import Protocol

from langfuse import observe

from langchain_core.documents import Document

from benchmark.providers import parse_model_id


class Reranker(Protocol):
    """Protocol for reranker implementations."""

    def rerank(
        self, query: str, documents: list[Document], top_k: int
    ) -> list[Document]: ...


class CrossEncoderReranker:
    """Reranker using a sentence-transformers CrossEncoder model.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g.
        ``"cross-encoder/ms-marco-MiniLM-L-6-v2"``.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    @observe(name="rerank")
    def rerank(
        self, query: str, documents: list[Document], top_k: int
    ) -> list[Document]:
        """Score and sort documents by relevance to the query.

        Returns at most ``top_k`` documents, ordered by descending score.
        """
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, documents), key=lambda x: x[0], reverse=True
        )
        return [doc for _, doc in ranked[:top_k]]


def get_reranker(model_string: str | None) -> Reranker | None:
    """Create a reranker from a model identifier string.

    Returns ``None`` when *model_string* is empty / ``None`` (reranking
    disabled).  Currently only the ``"huggingface:"`` provider prefix is
    supported.

    Parameters
    ----------
    model_string:
        Prefixed model identifier, e.g.
        ``"huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2"``.
    """
    if not model_string:
        return None

    provider, model_name = parse_model_id(model_string)

    if provider == "huggingface":
        return CrossEncoderReranker(model_name)

    raise ValueError(
        f"Unsupported reranker provider '{provider}'. "
        "Supported: 'huggingface'."
    )
