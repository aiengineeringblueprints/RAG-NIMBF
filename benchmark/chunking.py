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


def get_chunker(strategy: str, chunk_size: int, chunk_overlap: int):
    splitter_cls = STRATEGY_MAP.get(strategy)
    if splitter_cls is None:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Choose from: {list(STRATEGY_MAP.keys())}")
    # CharacterTextSplitter defaults to "\n\n" as separator, which produces oversized chunks
    # when paragraphs are long. "\n" gives finer-grained splits that respect the chunk_size.
    if strategy == "character":
        return splitter_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n")
    return splitter_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def chunk_documents(chunker, documents: list[dict]) -> list[Document]:
    docs = []
    for doc in documents:
        context = doc["context"]
        if isinstance(context, list):
            context = "\n".join(str(c) for c in context)
        docs.append(Document(page_content=context, metadata=doc.get("metadata", {})))

    return chunker.split_documents(docs)
