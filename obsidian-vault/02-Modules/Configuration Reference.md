# Configuration Reference

Source: [config.py](../config.py)

`BenchmarkConfig` is the central immutable configuration object. `get_all_combinations()` loads `.env`, validates values, computes the grid product, and returns one config per combination.

Core variables:

- `LLM_MODELS`: comma-separated generator models. Prefix with provider when needed, using [[Providers and Models]] parsing.
- `EMBEDDING_MODELS`: comma-separated embedding models.
- `CHUNK_SIZES`, `CHUNK_OVERLAPS`, `CHUNKING_STRATEGIES`
- `RETRIEVAL_TOP_K`
- `MAX_NEW_TOKENS`
- `DATASET_NAME`, `DATASET_SUBSET`, `DATASET_SAMPLE_SIZE`
- `EVAL_CRITIC_LLM`, `EVAL_CRITIC_EMBEDDING`, `EVAL_CRITIC_MAX_TOKENS`
- `PROMPT_TEMPLATES`
- `RERANKER_MODELS`, `RERANKER_TOP_K`

Provider URLs and keys:

- Shared defaults: `OLLAMA_BASE_URL`, `OLLAMA_API_KEY`, `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_API_KEY`
- Generator overrides: `LLM_OLLAMA_BASE_URL`, `LLM_OLLAMA_API_KEY`, `LLM_OPENAI_COMPAT_BASE_URL`, `LLM_OPENAI_COMPAT_API_KEY`
- Critic overrides: `EVAL_CRITIC_OLLAMA_BASE_URL`, `EVAL_CRITIC_OLLAMA_API_KEY`, `EVAL_CRITIC_OPENAI_COMPAT_BASE_URL`, `EVAL_CRITIC_OPENAI_COMPAT_API_KEY`
- Embedding overrides: `EMBEDDING_OLLAMA_BASE_URL`, `EMBEDDING_OLLAMA_API_KEY`

Retrieval behavior:

- `RETRIEVAL_STRATEGY`: `similarity` or `mmr`.
- `RETRIEVAL_FETCH_K`: MMR oversampling count.
- `RETRIEVAL_MMR_LAMBDA`: `0.0` means diversity, `1.0` means relevance.
- `RETRIEVAL_USE_HYDE`: enables hypothetical-answer query expansion.
- `RETRIEVAL_MODE`: `retrieval` or `direct`.
- `BENCHMARK_STAGE`: `all`, `index`, or `query`.
- `VECTOR_DB_BACKEND`: `chroma` or `lancedb`.
- `LANCEDB_PATH`: LanceDB storage path, default `.lancedb`.

Semantic chunking:

- `SEMANTIC_BREAKPOINT_TYPE`: `percentile`, `standard_deviation`, or `interquartile`.
- `SEMANTIC_BREAKPOINT_AMOUNT`: positive integer threshold amount.

Answer post-processing:

- `LLM_ANSWER_STRIP_MODE`: `full`, `tags_only`, or `off`.
- `LLM_ANSWER_VALUE_FALLBACK`: enables concise value extraction fallback.

Validation notes:

- Unknown datasets and prompt templates fail early.
- Chunk overlaps must be smaller than chunk sizes.
- Positive integer checks are enforced for sample size, token limits, chunk sizes, top-k, and semantic threshold.

Dataset notes:

- `ragperf-wikipedia-nq` mirrors RAGPerf's Wikipedia evaluation setup: it indexes
  `wikimedia/wikipedia` config `20231101.en` and uses
  `sentence-transformers/natural-questions` train rows for questions and answer
  ground truth.
- `RAGPERF_WIKIPEDIA_CORPUS_SIZE` controls how many Wikipedia documents are loaded
  for that dataset option. Default is `max(DATASET_SAMPLE_SIZE * 20, 1000)`.
- This dataset has no gold Wikipedia document/chunk IDs, so retrieval metrics are
  proxy or judge-based.

Stage and backend notes:

- `BENCHMARK_STAGE=index` chunks and indexes the selected corpus, saves per-config
  index results, and skips generation/evaluation. It requires
  `RETRIEVAL_MODE=retrieval`.
- `BENCHMARK_STAGE=query` expects the persisted vector store to already exist and
  fails early if the Chroma collection or LanceDB table is missing.
- `BENCHMARK_STAGE=all` keeps the previous end-to-end behavior.
- `VECTOR_DB_BACKEND=lancedb` uses the same embedding model and collection cache key
  as Chroma but stores data in LanceDB tables.

Related notes:

- [[Benchmark Pipeline]]
- [[Dataset Layer]]
- [[Chunking and Retrieval]]
