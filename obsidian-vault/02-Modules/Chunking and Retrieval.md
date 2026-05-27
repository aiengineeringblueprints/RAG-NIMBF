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

- `build_vector_store()` creates or reuses a vector store through the `VECTOR_DB_BACKEND` seam. Supported backends are `chroma` and `lancedb`.
- Chroma persists collections under `.chroma/`. Before reusing a persisted Chroma collection, it compares the collection document count with the number of chunks generated for the current run. If the count differs in `all`/`index` mode, the collection is deleted and rebuilt under the same deterministic name; in `query` mode, the benchmark fails fast and asks for an index rebuild.
- LanceDB persists tables under `LANCEDB_PATH`, default `.lancedb/`. `LanceDBVectorStore` is a small LangChain-like adapter that implements `similarity_search()` and `max_marginal_relevance_search()` for the common retrieval interface. Current LanceDB MMR delegates to similarity ranking because candidate vectors are not yet stored for full MMR diversification.
- Collection/cache keys include embedding model, provider, dataset identity, effective chunk settings, corpus fingerprint, and vector backend. For semantic chunking the effective chunk settings are `null`, so ignored size/overlap environment values do not create duplicate collections.
- `retrieve()` is backend-agnostic as long as the store exposes LangChain-style `similarity_search()` and `max_marginal_relevance_search()` methods.
- `expand_query_with_hyde()` generates a hypothetical answer to use as the retrieval query when HyDE is enabled.

Backend extension notes:

- New vector backends should be registered with `register_vector_store_backend()` and must preserve the shared retrieval method surface used by `retrieve()`.
- Backend selection should remain part of the cache key and config name so Chroma and LanceDB indexes never alias each other.
- Query-only mode must fail closed when the expected persisted collection/table is missing instead of silently rebuilding.

Reranking:

- `get_reranker()` returns `None` or a `CrossEncoderReranker`.
- The current built-in reranker uses `sentence-transformers` cross encoders.
- Reranking happens after initial retrieval and before prompt construction.

Operational notes:

- `.chroma/` and `.lancedb/` are intentionally persisted across runs to avoid re-embedding.
- Changing chunking, embedding model, dataset, sample content, or vector backend should lead to a new cache key.
- Semantic chunking needs an embedding model before splitting.
- Cache safety depends on collection names/cache keys encoding every content-affecting input, plus the persisted Chroma count check catching partial or stale collections.

Related notes:

- [[Configuration Reference]]
- [[Evaluation and Metrics]]
- [[Known Risks and Gaps]]
