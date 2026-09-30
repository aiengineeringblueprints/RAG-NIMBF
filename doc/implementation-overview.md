# Implementation Overview

What is implemented in this framework and how it works internally. Companion to
`doc/experiment-todos.md` (what to run) and `doc/00-README.md` (docs index).

---

## 1. Big picture

The framework runs RAG pipelines and external RAG systems against datasets, evaluates
the results, and reports to `results/runN/` + MLflow. There are three layers:

```
main.py  /  python -m benchmark.worker        (thin CLIs)
        │
benchmark/orchestration/
  ├── matrix.py    manifest → BenchmarkConfig matrix (cartesian product)
  ├── worker.py    resumable experiment loop (ExperimentWorker)
  └── runner.py    single-config execution (run_single_benchmark)
        │
benchmark/  (the actual machinery)
  ├── adapters/    RAG system adapters (internal, http, mcp, ragflow, plugins)
  ├── chunking / embedding / vectorstore / retrieval / reranker / generation
  ├── parsing/     document-parser benchmarking (OCR family)
  ├── evaluation / custom_metrics / gold_retrieval_metrics
  ├── reporting/   + tracking.py (MLflow), tracing, checkpoint, reproducibility
  └── dataset.py + dataset_adapters.py
```

Entry points:

- `main.py` — requires an experiment manifest (`argv`, `BENCHMARK_CONFIG_FILE`, or
  `EXPERIMENT_MANIFEST`); sets up tracing + MLflow and delegates to `ExperimentWorker`.
- `python -m benchmark.worker plan <manifest>` — expands the matrix, prints a JSON plan.
- `python -m benchmark.worker run <manifest>` — full resumable run
  (`--run-dir`, `--keep-going`, `--no-resume`, `--no-reports`, `--no-mlflow`).
  Without a manifest it falls back to the legacy `.env` matrix.

---

## 2. Configuration (`config.py`)

- `BenchmarkConfig` (config.py:36) — frozen dataclass with ~110 fields: LLM/embedding
  models + per-role Ollama/OpenAI-compatible URLs and API keys (generator, critic,
  embedding), dataset knobs (name/subset/sample_size/path/corpus_path), metric toggles,
  prompt template, reranker, answer stripping, retrieval knobs (similarity/MMR, HyDE,
  multihop, thresholds), chunking (strategy/size/overlap, semantic breakpoints),
  `vector_db_backend` (chroma|lancedb), `benchmark_stage` (all|index|query|retrieve|parsing),
  RAG adapter selection (`rag_system_adapter`, `rag_adapter_accepts`), parser adapter
  fields + `corpus_parser`, MCP fields (transport, result_mode, execution_mode, fairness),
  managed-RAG lifecycle fields, and LLM-performance knobs.
- `name` property (config.py:194) — deterministic run key embedding all non-default
  dimensions. Managed/MCP adapters get a sha256 hash of their effective surface so
  resume keys never collide.
- Validation: `validate_benchmark_config()` (config.py:341) does cross-field checks per
  stage/adapter (e.g. HTTP adapter can't index; `chunk_overlap < chunk_size`;
  `corpus_parser` requires `jsonl-shared` + corpus path).
- Two loading paths:
  - **Manifest (preferred)**: `benchmark/orchestration/matrix.py` —
    `load_experiment_spec()` reads YAML/JSON, `build_configs_from_spec()` (matrix.py:55)
    applies `dataset:` + `settings:` (validated against config field names) as the base,
    then cartesian-products the `matrix:` axes with legacy-name normalization and type
    coercion. Deduplicates configs and runs MCP fairness validation
    (`_validate_mcp_comparison_fairness`, matrix.py:113): RAG-vs-MCP matrices must agree
    on dataset/corpus/top-k/generator.
  - **Legacy env grid**: `get_all_combinations()` / `get_env_combinations()`
    (config.py:507/540) parse `LLM_MODELS`, `CHUNK_SIZES`, etc. from `.env`.

---

## 3. Orchestration

### ExperimentWorker (`benchmark/orchestration/worker.py:110`)

- `WorkerOptions` (worker.py:42): run_dir, resume, keep_going, dry_run, write_reports,
  log_mlflow.
- **Resumability**: `ProgressStore` (worker.py:53) keeps a JSON ledger
  `run_dir/progress.json` keyed by `config.name` with statuses
  `running/completed/failed` + timestamps, errors, result paths. A config is only
  skipped when marked `completed` **and** its result file exists.
- Run flow: pick the next `results/runN` dir → reject mixing `parsing` and RAG stages in
  one run → load the dataset **once** (`_load_data_once`, worker.py:279, including an
  optional `corpus_parser` ingestion) → write a reproducibility bundle +
  `worker_manifest.json` → open an MLflow parent run → loop over configs:
  - skip if completed,
  - `parsing` stage → `run_parsing_benchmark()`,
  - otherwise wrap with an optional `ResourceMonitor` and call
    `run_single_benchmark()`;
  - on failure mark `failed` and either raise or continue (`--keep-going`).
- After the loop: parsing leaderboard (parsing runs) or `generate_report()` +
  aggregate MLflow logging.

### Single-config runner (`benchmark/orchestration/runner.py`)

`run_single_benchmark()` (runner.py:127) guarantees adapter cleanup even on failure.
The core (`_run_single_benchmark_impl`, runner.py:167):

1. Resolve the adapter (`get_rag_adapter`), inject components + `prepare_adapter()`
   (chunk → embed → index; timed as `adapter_prepare`).
2. `benchmark_stage=index` returns early with an index-only result.
3. Per sample: checkpoint lookup (`CheckpointStore`) → `retrieve_adapter()` (stage
   `retrieve`) or `generate_adapter()`; stage timings folded in via
   `adapter_stage_timings()`; QA log streamed to `configs/<name>_qa.json`; each sample
   checkpointed to `<name>_checkpoint.json`.
4. Evaluation: RAGAS (`evaluate_results()`), skipped for retrieve-stage; custom metrics
   (`compute_custom_metrics()`) or gold-doc retrieval metrics
   (`compute_gold_doc_retrieval_metrics()`) for retrieve-stage /
   `custom_retrieval_metrics_mode=gold_doc`.
5. Aggregate into `BenchmarkResultExtended` (`compute_stats`, energy/cost estimates,
   adapter aggregate metrics).

---

## 4. RAG system adapters (`benchmark/adapters/`)

Contracts in `base.py`:

- `RagSystemOutput` (base.py:13) — normalized answer + contexts + metadata + timings +
  token usage + diagnostics. Everything downstream consumes this shape.
- `AdapterCapabilities` (base.py:37) — explicit feature flags (ingestion, retrieval,
  generation, chunk/embedding config, reranking, references, token usage, cleanup).
- `RagSystemAdapter` protocol (base.py:170) — black-box `answer()` plus optional
  component injection; `ManagedRagSystemAdapter` (base.py:208) — full
  `capabilities/prepare/retrieve/generate/cleanup` lifecycle.
- `RetrievalResult` / `AdapterGenerationResult` with legacy conversions.

**Registry** (`adapters/__init__.py`): `RAG_ADAPTER_REGISTRY` +
`register_rag_adapter(name, factory)`. Built-ins: `internal`, `http`, `mcp`,
`ragflow` (alias `optimaiserag`). New integrations register here — nothing is wired
into `main.py`.

**Lifecycle helpers** (`lifecycle.py`): capability checks (`require_adapter_capabilities`),
`prepare_adapter`, `retrieve_adapter`, `generate_adapter` (falls back to legacy
`answer()`), `cleanup_adapter`, timing/metric extraction.

**Component injection** (`components.py`): `ComponentBundle` (chunker, embedder,
retriever_factory, reranker, llm, prompt) is built from config and handed to adapters
that accept it. `resolve_injection_slots()` (components.py:137) intersects what the
adapter supports with the user intent `rag_adapter_accepts` (can only narrow, never
force). The retriever factory defers vector-store construction until the corpus exists.

Concrete adapters:

- **internal** (`internal.py:29`) — managed wrapper around the built-in pipeline:
  `prepare()` chunks the corpus and builds the vector store; `retrieve()` does optional
  HyDE expansion, similarity/MMR or multihop retrieval, then reranking; `generate()`
  builds the prompt from the template and calls `generate_answer` with strip mode +
  value fallback.
- **http** (`http.py:58`) — answer-only black box: JSON POST via urllib, configurable
  answer/contexts/metadata/timings field paths, auth header, token/cost extraction.
  Capabilities: generation only.
- **mcp** (`mcp.py:365`) — MCP tool backend over a `PersistentMcpSession` (dedicated
  asyncio thread, mcp.py:185), stdio or streamable-HTTP transport. `result_mode=context`
  (tool supplies evidence to the configured generator) vs. `answer` (tool is the full QA
  system); fixed or agentic multi-round tool-loop mode; retries/backoff; redacted
  argument logging; stage timings + aggregate metrics.
- **ragflow** (`ragflow.py`) — managed RAGFlow integration: `RagflowClient`
  (ragflow.py:283) handles dataset/document upload, parsing, polling, retrieval, chat
  management; deterministic resource naming + corpus digest for reuse; full
  prepare/retrieve/generate/cleanup lifecycle.

---

## 5. Built-in pipeline stages

- **Chunking** (`chunking.py`): `STRATEGY_MAP` (chunking.py:26) maps
  `recursive|character|token|markdown|text|transformers` to LangChain splitters;
  plus `paragraph` (one chunk per document) and `semantic` (`SemanticChunker` with
  breakpoint type/amount, requires embeddings). `chunk_documents()` (chunking.py:66)
  sanitizes metadata and drops chunks < 50 chars; MLflow-traced.
- **Embedding** (`embedding.py`): `get_embedding_model()` — `ollama` (optional Bearer
  key) or `huggingface`.
- **Vector store** (implemented in `retrieval.py`, not a package):
  `VectorStoreBackend` protocol (retrieval.py:179) with registered
  `ChromaVectorStoreBackend` (persistent `.chroma/`) and `LanceDBVectorStoreBackend`
  (retrieval.py:188); plugin registration via `register_vector_store_backend()`.
  `build_vector_store()` (retrieval.py:308) caches stores in-memory by cache key and
  reuses on-disk collections when the doc count matches.
- **Index identity**: `index_cache_key()` (retrieval.py:31) content-addresses the index
  from embedding model/provider, chunk params, dataset identity, and a
  `corpus_fingerprint` (retrieval.py:70) — identical corpora reuse the same index.
- **Retrieval** (`retrieval.py`): `retrieve()` (retrieval.py:388) with `similarity` and
  `mmr` (fetch_k, mmr_lambda); `expand_query_with_hyde()` (retrieval.py:431);
  `retrieve_multihop()` (retrieval.py:485) iteratively generates follow-up queries with
  the LLM.
- **Reranking** (`reranker.py`): `CrossEncoderReranker` (sentence-transformers
  cross-encoder); `get_reranker()` returns `None` without a model.
- **Generation** (`generation.py`): `get_llm()` delegates to
  `benchmark/providers.py` (`get_chat_model`, provider dispatch for
  ollama/openai-compatible ids, wrapped by content-as-string, RAGAS-wrapping, and
  token-counting wrappers). `_call_with_streaming()` (generation.py:328) measures true
  TTFT and collects usage metadata + `reasoning_content`. Answer post-processing
  (`_postprocess_answer`, generation.py:425): think-tag stripping (modes
  `full|tags_only|off`), thinking-heuristic detection, concise fallback extraction,
  `finqa` final-value extraction, percentage normalization. Prompt templates live in
  `benchmark/prompt_templates/` (`concise`, `detailed`, `finqa`).

---

## 6. Parsing / OCR benchmarking (`benchmark/parsing/`, `ocr_datasets.py`)

Implements benchmarking of *document parsers* (OCR models) as a first-class stage
(`benchmark_stage=parsing`):

- **Contracts** (`parsing/base.py`): `ParsedPage` (page_number, markdown, optional
  blocks), `ParseResult` (base.py:19 — document_id, pages, parser name/version,
  total_seconds, raw response, diagnostics; `.markdown` joins pages),
  `DocumentParser` protocol (`parse(document, config) -> ParseResult`; input documents
  carry per-page `image_base64`/text).
- **Registry** (`parsing/__init__.py`): `PARSER_ADAPTER_REGISTRY` +
  `register_parser_adapter()`; built-in `http` (`HttpParserAdapter`, http.py:22 —
  per-page HTTP calls with configurable endpoint/model/prompt/headers). Plugin modules
  take precedence over the registry (`load_parser_class`, plugin.py:11).
  `get_corpus_parser()` (parsing/__init__.py:114) resolves the `corpus_parser` knob so
  RAG corpora themselves can be built through a parser adapter.
- **Alignment** (`parsing/alignment.py`): paragraph splitting + `quick_match()`
  (alignment.py:325) — Levenshtein cost matrix with fuzzy sub/superset matching,
  merge/truncation handling (vendored OmniDocBench-style matching).
- **Text metrics** (`parsing/metrics.py`): `cer`, `wer`, `normalized_edit_distance`,
  per-page `PageTextMetrics`, computed raw and over aligned pairs.
- **Table metrics** (`parsing/tables.py`, `teds.py`, `table_metrics.py`): markdown table
  extraction → HTML conversion → TEDS tree-edit-distance score (full and
  structure-only), aggregated per page and overall.
- **Driver** (`parsing/evaluation.py`): `run_parsing_benchmark()` (evaluation.py:36)
  parses each document, aligns against ground truth, computes text+table metrics per
  config cell; `generate_parsing_leaderboard()` (evaluation.py:253) produces a
  cross-parser leaderboard.
- **Ground-truth datasets** (`ocr_datasets.py`): loaders for OmniDocBench,
  DP-Bench (MIT), olmOCR-Bench (Apache-2.0; only rows with `expected_text` are scored).
  All are local-file datasets: point `DATASET_PATH` at the export, nothing downloads.

---

## 7. Evaluation

- **RAGAS** (`evaluation.py`): `evaluate_results()` (evaluation.py:36) builds
  `SingleTurnSample`s, resolves a separate critic LLM (`eval_critic_llm` + per-role
  URLs) and critic embeddings through the provider factory (with token accounting),
  and runs `ragas.evaluate` with `RunConfig(timeout=600, max_workers=1)`.
  **Enabled by default:** `faithfulness`, `context_recall`, `SemanticSimilarity`.
  `answer_relevancy`, `answer_correctness`, `context_precision` are imported but
  currently not enabled. Returns means, per-sample scores, valid-sample counts, critic
  token usage, error text.
- **Custom metrics** (`custom_metrics.py`): non-RAGAS suite — IR heuristics
  (embedding-similarity relevance, `hit_at_k`, `ndcg_at_k`, `recall_at_k`,
  `context_relevance`), NLG metrics (`rouge_l`, `bleu`, `meteor_score`, `bert_score`
  via roberta-large), refusal detection. Driver `compute_custom_metrics()`
  (custom_metrics.py:408).
- **Gold-doc retrieval metrics** (`gold_retrieval_metrics.py`):
  `compute_gold_doc_retrieval_metrics()` (gold_retrieval_metrics.py:207) — Hit@k /
  NDCG@k / Recall@k against `gold_doc_id` metadata, plus gold-title sets and span
  relevance for HotpotQA-style supporting facts.
- **System metrics** (`metrics.py`): GPU sampling via nvidia-smi/NVML, host RAPL energy,
  energy-cost estimation.

---

## 8. Reporting, tracking, infrastructure

- **Reporting** (`benchmark/reporting/`): data models (`models.py`: `PerSampleResult`,
  `StatSummary`, `BenchmarkResultExtended`, `compute_stats`); exports to JSON/CSV/Markdown
  (`exports.py`); normalized multi-metric rankings + insights (`analysis.py`);
  matplotlib/Plotly visualization incl. publication-style chart sets (`visualization.py`,
  `paper_charts.py`, `ngen_ai_charts.py`, `resource_charts.py`); rich terminal report
  (`terminal.py`); cross-run scanning (`run_tracker.py`). Entry: `generate_report()`.
- **MLflow** (`tracking.py`): `setup_mlflow()`, nested per-config runs via
  `log_benchmark_run()` (flattened RAGAS stats, per-sample CSV artifacts, MLflow GenAI
  RAG judges), `log_parsing_run()`, aggregate artifact + plot logging, `log_genai_eval()`.
- **Tracing** (`tracing.py`): MLflow autolog/OTEL (optional OTLP exporter) + optional
  Langfuse; wired into both CLIs.
- **Checkpointing** (`checkpoint.py`): `CheckpointStore` — per-config JSON checkpoint
  of every sample (contexts, metadata, gold doc id, generation result) enabling
  per-question resume; `generation_result_from_record()` rebuilds results.
- **Reproducibility** (`reproducibility.py`): `build_reproducibility_manifest()`
  (reproducibility.py:102) — git commit, redacted env vars, config dicts, pip packages;
  written into the run dir.
- **Resource monitoring** (`resource_monitor.py`): `ResourceMonitor` background sampler
  (CPU, host memory, NVML GPU, disk rates) writing a CSV trace + stage markers;
  env-toggled.
- **Costing** (`costing.py`): env-driven model pricing map, per-answer USD estimates.
- **LLM performance** (`llm_performance.py`): standalone latency/throughput benchmark —
  TTFT/TPS percentiles over `call_counts` with warmup; results cached by performance key.
- **ClearML** (`clearml_task.py`): converts `BenchmarkConfig` ↔ ClearML parameters and
  runs benchmarks inside a ClearML agent task.

---

## 9. Datasets (`dataset.py`, `dataset_adapters.py`)

- `DatasetAdapter` (dataset_adapters.py:23) describes each source: HF id, question /
  ground-truth keys, context builder, preferred split, shared-corpus flag, license.
  Registry with `register`/`get_adapter`/`resolve_adapter` (dataset_adapters.py:60–72);
  a generic HF adapter is available via `DATASET_HF_ID`.
- Built-ins: `jsonl`, `jsonl-shared` (shared corpus — required for `corpus_parser`),
  `csv`, `t2-ragbench`, `ragbench`, `squad`, `ragas-wikiqa`,
  `ragperf-wikipedia-nq`, `hf-generic`, `multihop-generic`, and the OCR GT sets
  `omnidocbench`, `dp-bench`, `olmocr-bench`.
- `dataset.py`:
  - `load_benchmark_data()` (dataset.py:95) — per-sample QA datasets from HF or local
    JSONL/CSV.
  - `load_corpus_and_questions()` (dataset.py:176) — shared-corpus datasets; the corpus
    can optionally be ingested through a parser adapter (`load_corpus_via_parser`,
    dataset.py:433).
  - Helpers for chroma-safe metadata and stable doc IDs; special loader for
    ragperf-wikipedia-nq (dataset.py:653).

---

## 10. Known gaps (relevant when interpreting results)

- RAGAS/critic calls are slow and cost money — smoke with
  `RAGAS_ENABLED=false CUSTOM_METRICS_ENABLED=false` + tiny `DATASET_SAMPLE_SIZE`.
- Custom metrics and the reporting system have **no unit tests**; validate by hand on
  fixture data before trusting them (see `doc/04-testing-gaps.md`).
- Only three RAGAS metrics are enabled by default; the rest are one-line changes in
  `evaluation.py`.
- `benchmark_stage=index` works only for the internal adapter.
- MCP comparisons run fairness validation by default; disable only intentionally
  (`MCP_ENFORCE_FAIRNESS=false`).
- `mlflow.db` (~750 MB) sits in the repo root — back up before destructive MLflow
  scripts; `results/runN` dirs are append-only.
