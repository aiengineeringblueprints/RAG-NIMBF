# Chunking and Retrieval

Sources:

- [benchmark/chunking.py](../benchmark/chunking.py)
- [benchmark/retrieval.py](../benchmark/retrieval.py)
- [benchmark/reranker.py](../benchmark/reranker.py)
- [benchmark/embedding.py](../benchmark/embedding.py)

Chunking strategies:

- `recursive`
- `character`
- `token`
- `markdown`
- `text`
- `transformers`
- `semantic`

`get_chunker()` creates the configured LangChain text splitter. `chunk_documents()` converts normalized dataset rows into LangChain `Document` chunks and carries metadata forward.

Retrieval:

- `build_vector_store()` creates or reuses persisted Chroma collections under `.chroma/`.
- Collection/cache keys include embedding model, provider, dataset identity, chunk settings, and a corpus fingerprint.
- `retrieve()` supports similarity search and MMR.
- `expand_query_with_hyde()` generates a hypothetical answer to use as the retrieval query when HyDE is enabled.

Reranking:

- `get_reranker()` returns `None` or a `CrossEncoderReranker`.
- The current built-in reranker uses `sentence-transformers` cross encoders.
- Reranking happens after initial retrieval and before prompt construction.

Operational notes:

- `.chroma/` is intentionally persisted across runs to avoid re-embedding.
- Changing chunking, embedding model, dataset, or sample content should lead to a new cache key.
- Semantic chunking needs an embedding model before splitting.
- Cache safety depends on collection names/cache keys encoding every content-affecting input.

Related notes:

- [[Configuration Reference]]
- [[Evaluation and Metrics]]
- [[Known Risks and Gaps]]
