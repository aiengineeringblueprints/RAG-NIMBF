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

Semantic chunking is modeled differently from fixed-size splitters: `chunk_size` and `chunk_overlap` are ignored and reported as `null`. The benchmark grid creates only one semantic config per remaining combination, with granularity controlled by `SEMANTIC_BREAKPOINT_TYPE` and `SEMANTIC_BREAKPOINT_AMOUNT`.

Retrieval:

- `build_vector_store()` creates or reuses persisted Chroma collections under `.chroma/`. Before reusing a persisted Chroma collection, it compares the collection document count with the number of chunks generated for the current run. If the count differs in `all`/`index` mode, the collection is deleted and rebuilt under the same deterministic name; in `query` mode, the benchmark fails fast and asks for an index rebuild.
- Collection/cache keys include embedding model, provider, dataset identity, effective chunk settings, and a corpus fingerprint. For semantic chunking the effective chunk settings are `null`, so ignored size/overlap environment values do not create duplicate collections.
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
- Cache safety depends on collection names/cache keys encoding every content-affecting input, plus the persisted Chroma count check catching partial or stale collections.

Related notes:

- [[Configuration Reference]]
- [[Evaluation and Metrics]]
- [[Known Risks and Gaps]]
