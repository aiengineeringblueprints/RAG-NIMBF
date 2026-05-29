# Benchmark Pipeline

Source entrypoint: [main.py](../main.py)

`run_all_benchmarks()` loads all environment-derived configurations from `get_all_combinations()`, loads the selected dataset, creates `results/runN/`, then runs each config sequentially with `run_single_benchmark()`. `BENCHMARK_STAGE` can run the full flow (`all`), index only (`index`), or query an already-built vector store (`query`).


`benchmark.orchestration` adds a resumable worker layer around the same single-config core. `benchmark.worker` can plan a matrix from `.env` or a JSON/YAML manifest, run configs sequentially on one machine, write `progress.json`, skip completed results on resume, and generate normal reports for executed configs.

`run_single_benchmark()` stages:

1. Prepare per-config QA logging under `results/runN/configs/`.
2. Select a RAG system adapter. `RAG_SYSTEM_ADAPTER=internal` uses the built-in pipeline; `http` calls an external JSON RAG endpoint as a black-box system.
3. For the internal adapter, if `retrieval_mode == "retrieval"`, build chunks and a vector store.
4. For the internal adapter, if `retrieval_mode == "direct"`, skip vector store work and use each sample's supplied context.
5. For the HTTP adapter, skip internal chunking/retrieval/generation and normalize the endpoint response into answer, contexts, metadata, and timings.
6. Run RAGAS evaluation through [[Evaluation and Metrics]].
7. Compute custom metrics.
8. Build `BenchmarkResultExtended` with aggregate stats and per-sample results.
9. Save per-config JSON immediately.
10. Log the config as a nested MLflow child run under the benchmark parent run.
11. Optionally log MLflow classic retriever metrics and MLflow GenAI RAG judges.
12. Generate aggregated reports.

Stage timing is captured into each result JSON/CSV under `stage_timings` and
`stage_*_seconds` fields. Timed stages include data loading, chunking, indexing,
model loading, retrieval, optional HyDE/reranking, generation, RAGAS evaluation,
custom metrics, and total runtime.

Important implementation details:

- `_content_fingerprint()` hashes corpus/sample content for vector-store cache stability.
- `_get_bert_model()` caches SentenceTransformer BERTScore models in-process.
- `_next_run_dir()` allocates monotonic `results/runN/` folders.
- `_save_config_result()` writes each config result as soon as it finishes, so partial runs survive later failures.
- `VECTOR_DB_BACKEND` selects `chroma` or `lancedb`. Chroma persists under `.chroma/`; LanceDB persists under `LANCEDB_PATH`, default `.lancedb/`.
- External HTTP RAG systems are integrated through `benchmark/adapters/http.py`; they are evaluated with the same metric, reporting, and MLflow path as the internal pipeline.
- MLflow tracking is hierarchical: `main.py` and `benchmark.worker` open one parent run for the full benchmark invocation, and `log_benchmark_run()` creates one nested child run per `BenchmarkConfig`. Aggregate artifacts are logged as nested summary runs when a parent is active.
- Per-sample results carry `retrieved_doc_ids` and `ground_truth_doc_ids` when metadata is available, enabling MLflow `precision_at_k`, `recall_at_k`, and `ndcg_at_k` logging in addition to the existing custom gold-document metrics.

Adapter and validation seams:

- `benchmark/adapters/__init__.py` owns the RAG-system adapter registry. `register_rag_adapter()` adds adapter factories, and `get_rag_adapter()` returns `None` for the internal pipeline or an adapter instance for external systems such as `http`.
- `benchmark/adapters/base.py` defines the black-box RAG contract. Adapters normalize their output into `RagSystemOutput`, including answer text, retrieved contexts, metadata, timings, token stats, optional raw response fields, and `answer_valid`.
- Dataset samples are validated by `BenchmarkSample`, `normalize_sample()`, and `normalize_samples()` in `benchmark/dataset.py`. The public flow remains plain dictionaries with `question`, `ground_truth`, `context`, and `metadata`, but malformed samples now fail at the dataset boundary with source-indexed errors.
- Answer validity is currently checked at generation/adapter output time. Empty internal generations and empty HTTP answers are marked invalid and carried into per-sample reports.
- RAGAS evaluation and custom metrics preserve per-sample validity counts where available, which is the main reporting surface for skipped or invalid metric samples.

Related notes:

- [[Configuration Reference]]
- [[Chunking and Retrieval]]
- [[Reporting and Tracking]]
- [[Operations Runbook]]

## ClearML Entrypoint

`benchmark.clearml_task` wraps the resumable worker for ClearML Agent execution. It builds the same `BenchmarkConfig` objects as `benchmark.worker`, connects non-secret config fields to ClearML Hyperparameters, applies cloned-task UI overrides, and runs one selected config through `ExperimentWorker`. This leaves the benchmark core, result files, reports, and MLflow logging path unchanged while adding ClearML clone/enqueue execution.
