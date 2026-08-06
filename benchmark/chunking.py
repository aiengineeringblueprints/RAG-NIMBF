import mlflow

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    TextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain_core.documents import Document

# Multi-modal "chunkers" (Docling OCR, Whisper ASR) don't go through LangChain
# text splitters. They are registered separately and surface here so config
# validation can recognise the strategy names without importing heavy deps.
try:
    from benchmark.multimodal.registry import (
        MULTIMODAL_CHUNKERS,
        is_chunker_multimodal,
    )
except ImportError:  # pragma: no cover - multimodal pkg always present
    MULTIMODAL_CHUNKERS = ()  # type: ignore[assignment]

    def is_chunker_multimodal(_strategy: str) -> bool:  # type: ignore[misc]
        return False

STRATEGY_MAP = {
    "recursive": RecursiveCharacterTextSplitter,
    "character": CharacterTextSplitter,
    "token": TokenTextSplitter,
    "markdown": MarkdownTextSplitter,
    "text": TextSplitter,
    "transformers": SentenceTransformersTokenTextSplitter,
}


def get_chunker(strategy: str, chunk_size: int, chunk_overlap: int, **kwargs):
    if is_chunker_multimodal(strategy):
        raise ValueError(
            f"Chunking strategy '{strategy}' is a multi-modal ingestion path. "
            "It produces corpus text directly from disk (PDF/image/audio) and "
            "must be dispatched via benchmark.multimodal before reaching "
            "get_chunker — it does not return a LangChain TextSplitter."
        )
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
        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=breakpoint_amount,
        )

    splitter_cls = STRATEGY_MAP.get(strategy)
    if splitter_cls is None:
        known = sorted(set(STRATEGY_MAP) | {"semantic"} | set(MULTIMODAL_CHUNKERS))
        raise ValueError(
            f"Unknown chunking strategy: {strategy}. Choose from: {known}"
        )
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


def known_chunking_strategies() -> tuple[str, ...]:
    """All valid chunking-strategy names — text + semantic + multi-modal."""
    return tuple(sorted(set(STRATEGY_MAP) | {"semantic"} | set(MULTIMODAL_CHUNKERS)))


def run_multimodal_ingestion(
    strategy: str,
    corpus_path: str,
    *,
    settings: dict | None = None,
) -> list[dict]:
    """Dispatch to a registered multi-modal chunker factory.

    Returns the same ``[{context, metadata}, ...]`` shape as the text corpus
    loader, so the downstream text chunker / embedder / retriever can consume
    the output unchanged. Heavy ML deps are imported lazily inside the factory.

    Parameters
    ----------
    strategy:
        A registered multi-modal chunker name (e.g. ``pdf_ocr_docling``).
    corpus_path:
        Directory containing the source files.
    settings:
        Optional dict of kwargs forwarded to the factory (e.g. Whisper model
        name, Docling OCR flag). Unknown keys are ignored by factories.
    """
    from benchmark.multimodal.registry import MULTIMODAL_CHUNKERS

    if strategy not in MULTIMODAL_CHUNKERS:
        raise ValueError(
            f"'{strategy}' is not a registered multi-modal chunker. "
            f"Available: {sorted(MULTIMODAL_CHUNKERS)}"
        )
    factory = MULTIMODAL_CHUNKERS[strategy]
    return factory(corpus_path=corpus_path, **(settings or {}))
