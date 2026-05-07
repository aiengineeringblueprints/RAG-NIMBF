# Benchmark Pipeline

Source entrypoint: [main.py](../main.py)

`run_all_benchmarks()` loads all environment-derived configurations from `get_all_combinations()`, loads the selected dataset, creates `results/runN/`, then runs each config sequentially with `run_single_benchmark()`. `BENCHMARK_STAGE` can run the full flow (`all`), index only (`index`), or query an already-built vector store (`query`).

`run_single_benchmark()` stages:

1. Prepare per-config QA logging under `results/runN/configs/`.
2. If `retrieval_mode == "retrieval"`, build chunks and Chroma vector store.
3. If `retrieval_mode == "direct"`, skip vector store work and use each sample's supplied context.
4. Create generator LLM and optional reranker.
5. For each sample, retrieve context, generate an answer, and append the streamed answer log.
6. Run RAGAS evaluation through [[Evaluation and Metrics]].
7. Compute custom metrics.
8. Build `BenchmarkResultExtended` with aggregate stats and per-sample results.
9. Save per-config JSON immediately.
10. Log benchmark, genai eval payload, and plots to MLflow.
11. Generate aggregated reports.

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

Related notes:

- [[Configuration Reference]]
- [[Chunking and Retrieval]]
- [[Reporting and Tracking]]
- [[Operations Runbook]]
