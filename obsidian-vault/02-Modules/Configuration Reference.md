# Configuration Reference

Source: [config.py](../config.py)

`BenchmarkConfig` is the central immutable configuration object. `get_all_combinations()` loads `.env`, validates values, computes the grid product, and returns one config per combination.

Core variables:

- `LLM_MODELS`: comma-separated generator models. Prefix with provider when needed, using [[Providers and Models]] parsing.
- `EMBEDDING_MODELS`: comma-separated embedding models.
- `CHUNK_SIZES`, `CHUNK_OVERLAPS`, `CHUNKING_STRATEGIES`
  - `semantic` ignores `CHUNK_SIZES` and `CHUNK_OVERLAPS`; it contributes one config per remaining grid combination instead of multiplying across the size/overlap grid.
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
- `CUSTOM_RETRIEVAL_METRICS_MODE`: `heuristic` keeps the existing context-overlap retrieval-style custom metrics; `gold_doc` replaces `hit@k`, `nDCG@k`, and `recall@k` with gold-document ID scoring when samples provide `metadata.gold_doc_id`.
- `BENCHMARK_STAGE`: `all`, `index`, or `query`.
- `VECTOR_DB_BACKEND`: `chroma` or `lancedb`.
- `LANCEDB_PATH`: LanceDB storage path, default `.lancedb`.

Semantic chunking:

- `SEMANTIC_BREAKPOINT_TYPE`: `percentile`, `standard_deviation`, or `interquartile`.
- `SEMANTIC_BREAKPOINT_AMOUNT`: positive integer threshold amount.
- Semantic chunking is controlled by embeddings and breakpoint settings, not fixed chunk sizes or overlap. Semantic configs store `chunk_size` and `chunk_overlap` as `null` in JSON reports and omit `cs/co` from config names.

RAG system adapters:

- `RAG_SYSTEM_ADAPTER`: `internal` keeps the built-in chunk/retrieve/generate pipeline; `http` evaluates an external RAG service through JSON HTTP POST.
- `RAG_HTTP_ENDPOINT_URL`: required for `RAG_SYSTEM_ADAPTER=http`. The framework sends `{question, metadata, ground_truth, config}` and expects a JSON object.
- `RAG_HTTP_ANSWER_FIELD`, `RAG_HTTP_CONTEXTS_FIELD`, `RAG_HTTP_METADATA_FIELD`, `RAG_HTTP_TIMINGS_FIELD`: dotted response field paths. Defaults are `answer`, `contexts`, `metadata`, and `timings`.
- `RAG_HTTP_TIMEOUT_SECONDS`: request timeout, default `60`.
- `RAG_HTTP_HEADERS`: optional JSON object of static request headers.
- `RAG_HTTP_AUTH_HEADER`, `RAG_HTTP_AUTH_VALUE`: optional single auth header without committing secrets to source.
- `BENCHMARK_STAGE=index` is only supported by the internal adapter, because black-box HTTP systems own their own indexing lifecycle.

Answer post-processing:

- `LLM_ANSWER_STRIP_MODE`: `full`, `tags_only`, or `off`.
- `LLM_ANSWER_VALUE_FALLBACK`: enables concise value extraction fallback.

Validation notes:

- Unknown datasets fail early against `benchmark.dataset_adapters.REGISTRY`; unknown prompt templates fail early against `benchmark.prompt_templates.BUILTIN_TEMPLATES`.
- `RAG_SYSTEM_ADAPTER` is validated against the RAG adapter registry in `benchmark/adapters/__init__.py`; built-ins are `internal` and `http`. The HTTP adapter also requires `RAG_HTTP_ENDPOINT_URL` and a positive timeout.
- `VECTOR_DB_BACKEND` is validated against `available_vector_store_backends()` from `benchmark/retrieval.py`; built-ins are `chroma` and `lancedb`. This keeps backend selection explicit at config load time while allowing registered extensions.
- `BENCHMARK_STAGE=index` is rejected with `RETRIEVAL_MODE=direct` and with non-internal RAG adapters.
- Chunk overlaps must be smaller than chunk sizes for non-semantic chunking strategies. Semantic chunking ignores both values.
- Positive integer checks are enforced for sample size, token limits, chunk sizes, top-k, and semantic threshold.
- Dataset samples are normalized through `BenchmarkSample`, `normalize_sample()`, and `normalize_samples()` in `benchmark/dataset.py`. Malformed rows surface at the dataset boundary with source-indexed errors.

Adapter registries:

- Dataset adapters live in `benchmark/dataset_adapters.py` and register themselves in `REGISTRY` with short names such as `t2-ragbench`, `ragbench`, `squad`, `ragas-wikiqa`, and `ragperf-wikipedia-nq`.
- RAG-system adapters live under `benchmark/adapters/`. `register_rag_adapter()` is the extension seam and `get_rag_adapter()` dispatches through the registry: `internal` returns `None`, while `http` returns `HttpRagAdapter.from_config(config)`.
- Prompt templates are a separate registry in `benchmark/prompt_templates/__init__.py`; config validation checks names before benchmark execution.

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

Resource monitor variables:

- `BENCHMARK_RESOURCE_MONITOR`: `true`/`false`, default `false`. When enabled, records per-config resource traces for paper-style utilization plots.
- `BENCHMARK_RESOURCE_MONITOR_INTERVAL_SECONDS`: sampling interval in seconds, default `1.0`.
- `BENCHMARK_RESOURCE_MONITOR_GPU_INDEX`: GPU index passed to `nvidia-smi --id`, default `0`.
