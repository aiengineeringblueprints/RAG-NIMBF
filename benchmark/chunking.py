import mlflow

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    TextSplitter,
    SentenceTransformersTokenTextSplitter
)


class ParagraphChunker:
    """Emits one chunk per source document without splitting.

    Designed for paragraph-level corpora (e.g. HotpotQA paragraphs), where
    each corpus document is already the natural retrieval unit. Preserves
    metadata (``title``, ``doc_id``) so gold-retrieval metrics work per chunk.
    """

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return list(documents)
from langchain_core.documents import Document

STRATEGY_MAP = {
    "recursive": RecursiveCharacterTextSplitter,
    "character": CharacterTextSplitter,
    "token": TokenTextSplitter,
    "markdown": MarkdownTextSplitter,
    "text": TextSplitter,
    "transformers": SentenceTransformersTokenTextSplitter,
}


class SizeCappedChunker:
    """Post-splits oversized chunks from splitters without a size guarantee.

    ``SemanticChunker`` has no ``chunk_size`` parameter: when no breakpoint
    falls between sentences, an entire document becomes one chunk. Embedding
    providers reject such oversized inputs (Ollama resets the connection),
    so every chunk larger than ``chunk_size`` is re-split with a recursive
    splitter. Metadata (``doc_id``, ``gold_doc_id`` …) is preserved.
    """

    def __init__(self, inner, chunk_size: int, chunk_overlap: int):
        self.inner = inner
        # Semantic strategies carry chunk_size=None (they are breakpoint
        # controlled, matrix.py maps them to (None, None)); components.py
        # passes that through as 0. Use a conservative char cap that stays
        # within embedding-provider input limits (Ollama rejects much less).
        self.chunk_size = int(chunk_size) if chunk_size and chunk_size > 0 else 4000
        self.chunk_overlap = max(int(chunk_overlap) if chunk_overlap else 0, 0)
        self._fallback = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks = self.inner.split_documents(documents)
        if not any(len(c.page_content) > self.chunk_size for c in chunks):
            return chunks
        result: list[Document] = []
        for chunk in chunks:
            if len(chunk.page_content) > self.chunk_size:
                result.extend(self._fallback.split_documents([chunk]))
            else:
                result.append(chunk)
        return result


def get_chunker(strategy: str, chunk_size: int, chunk_overlap: int, **kwargs):
    if strategy == "paragraph":
        return ParagraphChunker()
    if strategy == "semantic":
        from langchain_experimental.text_splitter import SemanticChunker

        embeddings = kwargs.get("embeddings")
        if embeddings is None:
            raise ValueError(
                "Semantic chunking requires an 'embeddings' argument. "
                "Pass a LangChain Embeddings instance via kwargs."
            )
        breakpoint_type = kwargs.get("breakpoint_threshold_type", "percentile")
        breakpoint_amount = kwargs.get("breakpoint_threshold_amount", 95)
        return SizeCappedChunker(
            SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=breakpoint_type,
                breakpoint_threshold_amount=breakpoint_amount,
            ),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    splitter_cls = STRATEGY_MAP.get(strategy)
    if splitter_cls is None:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Choose from: {list(STRATEGY_MAP.keys())} + 'semantic'")
    # CharacterTextSplitter defaults to "\n\n" as separator, which produces oversized chunks
    # when paragraphs are long. "\n" gives finer-grained splits that respect the chunk_size.
    if strategy == "character":
        return splitter_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n")
    return splitter_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


@mlflow.trace(name="chunk_documents", span_type="func")
def chunk_documents(chunker, documents: list[dict], min_chunk_length: int = 50) -> list[Document]:
    from benchmark.dataset import _chroma_safe_metadata

    docs = []
    for doc in documents:
        context = doc["context"]
        if isinstance(context, list):
            context = "\n".join(str(c) for c in context)
        docs.append(Document(
            page_content=context,
            metadata=_chroma_safe_metadata(doc.get("metadata", {}) or {}),
        ))

    chunks = chunker.split_documents(docs)
    # Filter out near-empty fragments (bibliography lines, citations, etc.)
    filtered = [c for c in chunks if len(c.page_content.strip()) >= min_chunk_length]
    return filtered or chunks
