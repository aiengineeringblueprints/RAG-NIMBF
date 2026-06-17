"""Framework-built components offered to external RAG adapters via injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING


@dataclass
class ComponentBundle:
    """Optional LangChain-compatible components built from .env.

    All fields default to None. Adapters pick the slots they support; the
    rest stay None. None means "Framework did not build this slot" or
    "adapter declined injection via supports_components".
    """
    chunker: Optional[Any] = None  # langchain TextSplitter
    embedder: Optional[Any] = None  # langchain Embeddings
    retriever_factory: Optional[Callable[[list[dict]], Any]] = None
    reranker: Optional[Any] = None
    llm: Optional[Any] = None  # langchain BaseChatModel
    prompt_template: Optional[str] = None


if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.embeddings import Embeddings
    from langchain_text_splitters import TextSplitter
