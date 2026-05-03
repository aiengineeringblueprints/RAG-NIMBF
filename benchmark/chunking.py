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

STRATEGY_MAP = {
    "recursive": RecursiveCharacterTextSplitter,
    "character": CharacterTextSplitter,
    "token": TokenTextSplitter,
    "markdown": MarkdownTextSplitter,
    "text": TextSplitter,
    "transformers": SentenceTransformersTokenTextSplitter,
}


def get_chunker(strategy: str, chunk_size: int, chunk_overlap: int, **kwargs):
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
        raise ValueError(f"Unknown chunking strategy: {strategy}. Choose from: {list(STRATEGY_MAP.keys())} + 'semantic'")
    # CharacterTextSplitter defaults to "\n\n" as separator, which produces oversized chunks
    # when paragraphs are long. "\n" gives finer-grained splits that respect the chunk_size.
    if strategy == "character":
        return splitter_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n")
    return splitter_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


@mlflow.trace(name="chunk_documents", span_type="func")
def chunk_documents(chunker, documents: list[dict], min_chunk_length: int = 50) -> list[Document]:
    docs = []
    for doc in documents:
        context = doc["context"]
        if isinstance(context, list):
            context = "\n".join(str(c) for c in context)
        docs.append(Document(page_content=context, metadata=doc.get("metadata", {})))

    chunks = chunker.split_documents(docs)
    # Filter out near-empty fragments (bibliography lines, citations, etc.)
    filtered = [c for c in chunks if len(c.page_content.strip()) >= min_chunk_length]
    return filtered or chunks
