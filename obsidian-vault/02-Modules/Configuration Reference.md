# Configuration Reference

Source: [config.py](../config.py)

`BenchmarkConfig` is the central immutable configuration object. `get_all_combinations()` loads `.env` for runtime secrets/service endpoints and, when `BENCHMARK_CONFIG_FILE` or an explicit path is set, expands a JSON/YAML manifest into one config per combination. If no manifest is configured, it preserves the legacy `.env` matrix behavior.


Experiment manifests:

- `BENCHMARK_CONFIG_FILE=experiments/full-grid-example.yaml python main.py` runs the normal benchmark entrypoint from YAML without requiring ClearML.
- `python -m benchmark.worker plan <manifest>` loads the same JSON/YAML manifests through `benchmark.orchestration.matrix` for resumable worker runs.
- Manifests can override `dataset`, `settings`, and `matrix` values while `.env` still supplies provider URLs, API keys, and defaults for omitted fields.
- Common matrix aliases are `llm_models`, `embedding_models`, `chunking_strategies`, `chunk_sizes`, `chunk_overlaps`, `prompt_templates`, and `reranker_models`; direct `BenchmarkConfig` field names such as `retrieval_top_k`, `retrieval_strategy`, and `retrieval_use_hyde` are also accepted. Dataset lists under `dataset.name`, `dataset.subset`, or `dataset.sample_size` are treated as matrix axes.
- `semantic` configs are deduplicated across chunk-size/overlap axes because those fields do not affect semantic chunking.
- `retrieval_top_k` is included in config names when it differs from the default `5`, preventing result-file collisions during top-k sweeps.
- Example: [full-grid-example.yaml](../experiments/full-grid-example.yaml).

YAML-first workflow:

- Put non-secret benchmark workflow values in `experiments/*.yaml`: models, embeddings, chunking, retrieval, dataset, evaluator, vector backend, stage, prompt templates, reranker, and adapter settings.
- Keep `.env` for `BENCHMARK_CONFIG_FILE`, provider base URLs, API keys, ClearML credentials, LangFuse credentials, and other machine-local service endpoints.
- `EXPERIMENT_MANIFEST` is accepted as a compatibility alias for `BENCHMARK_CONFIG_FILE`.

Legacy `.env` variables, still supported when no YAML manifest is set:

- `LLM_MODELS`: comma-separated generator models. Prefix with provider when needed, using [[Providers and Models]] parsing.
- `EMBEDDING_MODELS`: comma-separated embedding models.
- `CHUNK_SIZES`, `CHUNK_OVERLAPS`, `CHUNKING_STRATEGIES`
  - `semantic` ignores `CHUNK_SIZES` and `CHUNK_OVERLAPS`; it contributes one config per remaining grid combination instead of multiplying across the size/overlap grid.
- `RETRIEVAL_TOP_K`
- `MAX_NEW_TOKENS`
- `DATASET_NAME`, `DATASET_SUBSET`, `DATASET_SAMPLE_SIZE`
- `DATASET_PATH`, `DATASET_QUESTION_FIELD`, `DATASET_GROUND_TRUTH_FIELD`, `DATASET_CONTEXT_FIELD`, `DATASET_METADATA_FIELD` for local `jsonl`/`csv` datasets.
- `RAGAS_ENABLED`, `CUSTOM_METRICS_ENABLED` can disable model-based evaluation blocks for smoke tests or answer-only runs.
- `EVAL_CRITIC_LLM`, `EVAL_CRITIC_EMBEDDING`, `EVAL_CRITIC_MAX_TOKENS`
- `PROMPT_TEMPLATES`
- `RERANKER_MODELS`, `RERANKER_TOP_K`

Provider URLs and keys stay in `.env` even in YAML-first runs:

- Shared defaults: `OLLAMA_BASE_URL`, `OLLAMA_API_KEY`, `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_API_KEY`
- Generator overrides: `LLM_OLLAMA_BASE_URL`, `LLM_OLLAMA_API_KEY`, `LLM_OPENAI_COMPAT_BASE_URL`, `LLM_OPENAI_COMPAT_API_KEY`
- Critic overrides: `EVAL_CRITIC_OLLAMA_BASE_URL`, `EVAL_CRITIC_OLLAMA_API_KEY`, `EVAL_CRITIC_OPENAI_COMPAT_BASE_URL`, `EVAL_CRITIC_OPENAI_COMPAT_API_KEY`
- Embedding overrides: `EMBEDDING_OLLAMA_BASE_URL`, `EMBEDDING_OLLAMA_API_KEY`

API cost tracking:

- Generator calls now preserve provider token usage as `input_tokens`, `output_tokens`, and `total_tokens` per sample. `token_count` remains the backward-compatible output-token count.
- `LLM_MODEL_PRICING_USD_PER_1M`: optional JSON object mapping model names to USD prices per one million tokens, for example `{"deepseek-v4-pro": {"input": 0.0, "output": 0.0}}`. Replace the zeroes with the current provider pricing. Aliases `input_per_1m`/`output_per_1m` and `prompt`/`completion` are accepted.
- `MODEL_PRICING_USD_PER_1M`: compatibility alias for the same JSON.
- `LLM_INPUT_COST_PER_1M_USD` and `LLM_OUTPUT_COST_PER_1M_USD`: fallback pricing for single-model runs when no per-model JSON entry is configured.
- Cost outputs are `estimated_cost_usd` per sample plus `total_estimated_cost_usd` and `avg_estimated_cost_per_answer_usd` per config in JSON, CSV, and MLflow. If no pricing is configured, token counts are still recorded and cost fields stay empty.
- The current estimate covers answer-generation calls. RAGAS critic calls and HyDE query-expansion calls are separate paid LLM calls if configured against API models.

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
- `MLFLOW_CLASSIC_RETRIEVER_METRICS_ENABLED`: `true` by default. When retrieved and gold document IDs exist, `benchmark/tracking.py` logs MLflow `precision_at_k`, `recall_at_k`, and `ndcg_at_k` in the active config child run.
- `MLFLOW_GENAI_JUDGES_ENABLED`: `false` by default. When enabled, `benchmark/tracking.py` replays stored per-sample answers and contexts through `mlflow.genai.evaluate()` with RAG judges.
- `MLFLOW_GENAI_JUDGE_MODEL`: judge model identifier for MLflow RAG judges, default `openai:/gpt-4o-mini`. Requires the provider credentials expected by MLflow.

Semantic chunking:

- `SEMANTIC_BREAKPOINT_TYPE`: `percentile`, `standard_deviation`, or `interquartile`.
- `SEMANTIC_BREAKPOINT_AMOUNT`: positive integer threshold amount.
- Semantic chunking is controlled by embeddings and breakpoint settings, not fixed chunk sizes or overlap. Semantic configs store `chunk_size` and `chunk_overlap` as `null` in JSON reports and omit `cs/co` from config names.

RAG system adapters:

- `RAG_SYSTEM_ADAPTER`: `internal` keeps the built-in chunk/retrieve/generate pipeline; `http` evaluates an external RAG service through JSON HTTP POST.
- `RAG_ADAPTER_MODULES`: optional comma-separated Python module list imported before adapter validation. Use this for native adapter plugins that call `register_rag_adapter()`.
- `RAG_HTTP_ENDPOINT_URL`: required for `RAG_SYSTEM_ADAPTER=http`. The framework sends `{question, metadata, ground_truth, config}` and expects a JSON object.
- `RAG_HTTP_ANSWER_FIELD`, `RAG_HTTP_CONTEXTS_FIELD`, `RAG_HTTP_METADATA_FIELD`, `RAG_HTTP_TIMINGS_FIELD`: dotted response field paths. Defaults are `answer`, `contexts`, `metadata`, and `timings`.
- `RAG_HTTP_TIMEOUT_SECONDS`: request timeout, default `60`.
- `RAG_HTTP_HEADERS`: optional JSON object of static request headers.
- `RAG_HTTP_AUTH_HEADER`, `RAG_HTTP_AUTH_VALUE`: optional single auth header without committing secrets to source.
- `examples/http_rag_server.py` and `examples/sample_dataset.jsonl` provide a minimal smoke-test endpoint and dataset for HTTP adapter onboarding.
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

- Dataset adapters live in `benchmark/dataset_adapters.py` and register themselves in `REGISTRY` with short names such as `jsonl`, `csv`, `t2-ragbench`, `ragbench`, `squad`, `ragas-wikiqa`, and `ragperf-wikipedia-nq`.
- RAG-system adapters live under `benchmark/adapters/`. `register_rag_adapter()` is the extension seam and `get_rag_adapter()` dispatches through the registry: `internal` returns `None`, while `http` returns `HttpRagAdapter.from_config(config)`.
- Prompt templates are a separate registry in `benchmark/prompt_templates/__init__.py`; config validation checks names before benchmark execution.

Dataset notes:

- `jsonl` and `csv` load local files through `DATASET_PATH`. Field mapping is controlled by `DATASET_QUESTION_FIELD`, `DATASET_GROUND_TRUTH_FIELD`, `DATASET_CONTEXT_FIELD`, and `DATASET_METADATA_FIELD`. JSONL rows must be objects; CSV metadata can be blank, plain text, or a JSON object string.
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

## ClearML Parameters

`benchmark.clearml_task` exposes the first `.env` or manifest-derived `BenchmarkConfig` as ClearML Hyperparameters and applies Web UI overrides before running the existing worker core. Common editable fields include `llm_model`, `llm_provider`, `embedding_model`, `chunk_size`, `chunk_overlap`, `chunking_strategy`, `retrieval_top_k`, `prompt_template`, `reranker_model`, dataset settings, vector backend settings, and HTTP adapter field mappings.

Secret-bearing fields are intentionally excluded: API keys, auth header names/values, and raw HTTP headers. Keep those configured in the ClearML Agent runtime environment. A prefixed `llm_model` such as `openai:Qwen/Qwen3-32B-AWQ` updates both model and provider; an unprefixed model name preserves the separately configured `llm_provider`.


Enterprise RAG Blueprint adapter notes:

- `examples.enterprise_rag_plugin.adapter` registers `RAG_SYSTEM_ADAPTER=enterprise_rag`.
- On `prepare()`, the adapter replaces the Blueprint Chroma collection named by `INDEX_NAME` under `VECTORDB_DIR` with the YAML-selected benchmark dataset corpus. No separately pre-indexed `Enterprise_RAG_Blueprint/test_documents` corpus is required or mixed into evaluation.
- The adapter accepts framework-built `chunker`, `embedder`, and `llm` components. Blueprint still owns its Chroma retriever and PromptKey prompt selection.
- Adapter component capability overrides are applied on a copied `BenchmarkConfig`, because the config dataclass is frozen; runtime injection must not mutate the original config.
