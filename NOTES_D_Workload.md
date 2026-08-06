# Notes — Workload Generator (Design D)

This file documents the RAGPerf-style concurrent workload generator that
ships under `benchmark/workload/`. The generator issues a configurable
mix of Query / Insert / Update / Remove operations against a live RAG
pipeline (vector store + generator) under open-loop arrival and bounded
concurrency, replacing the static sequential query loop when enabled.

Inspired by RAGPerf §3.2 (arXiv:2603.10765v1).

## Layout

| Module | Role |
| --- | --- |
| `benchmark/workload/__init__.py` | Public re-exports. |
| `benchmark/workload/distributions.py` | `uniform_sample` + `zipfian_sample`. |
| `benchmark/workload/operations.py` | `QueryOp` / `InsertOp` / `UpdateOp` / `RemoveOp` dataclasses. |
| `benchmark/workload/generator.py` | `WorkloadConfig` + `WorkloadGenerator`. Builds the deterministic op stream with Poisson arrivals and the configured mix. |
| `benchmark/workload/runner.py` | `WorkloadRunner` + `WorkloadSummary` + `OpRecord`. Executes ops against the live vector store + generator with a `ThreadPoolExecutor`. |
| `benchmark/workload/report.py` | `summarize`, `write_summary_json`, `mlflow_metrics`. |
| `benchmark/workload/integration.py` | `run_workload_phase` glue layer wired into `main.py`. Builds a query callable around the existing retrieve + generate path. |
| `tests/test_workload.py` | Distribution, generator, and runner tests with a fake vector store. |

Vector-store adapter helpers were added to `benchmark/retrieval.py`:
`workload_insert_documents`, `workload_update_document`,
`workload_remove_document`. All three serialise on the existing
`_chroma_lock` because ChromaDB is not thread-safe for concurrent writes
to a single collection (its Rust bindings race on the persistent index).
Reads also acquire the lock defensively — Chroma tolerates concurrent
reads but coupling with concurrent writes was occasionally flaky in CI.

## YAML schema

Set `workload_enabled: true` in the experiment `settings:` block. The
mode replaces the sequential query loop in `main.py` for the `all` and
`query` stages of the internal RAG pipeline.

```yaml
settings:
  workload_enabled: true

  # Op probabilities (renormalised at generation time).
  workload_op_mix: {query: 0.7, insert: 0.15, update: 0.1, remove: 0.05}
  # Equivalent flat form also accepted:
  # workload_op_mix_query: 0.7
  # workload_op_mix_insert: 0.15
  # workload_op_mix_update: 0.1
  # workload_op_mix_remove: 0.05

  workload_distribution: zipfian   # uniform | zipfian
  workload_zipf_theta: 0.8         # > 0; higher = more skew
  workload_target_qps: 5           # Poisson offered rate
  workload_concurrency: 8          # max in-flight ops (ThreadPoolExecutor)
  workload_total_ops: 500          # length of the run
```

Equivalent environment variables (`WORKLOAD_ENABLED`,
`WORKLOAD_OP_MIX_QUERY`, etc.) are also honoured, mirroring the rest of
the framework's `.env`-driven matrix.

Defaults are conservative: `workload_enabled: false`, `target_qps=5`,
`concurrency=8`, `total_ops=500`, mix `0.7 / 0.15 / 0.1 / 0.05`,
distribution `uniform`.

## Per-op semantics

* **Query** — pulls a question from the pool (uniform or Zipf-skewed over
  question indices), runs the full retrieve -> generate path, and
  records latency, outcome, answer, contexts, and ground truth for the
  Ragas eval pass.
* **Insert** — embeds a fresh `Document` (LangChain) and adds it to the
  index. By default the generator synthesises a short placeholder
  document; pass `insert_pool=[{"doc_id": ..., "text": ...}, ...]` to
  `WorkloadGenerator` to draw from curated content.
* **Update** — picks an existing `doc_id`, replaces its body with new
  text, and re-indexes. Implemented as delete-all-chunks-then-insert-one
  inside the write lock to avoid partial states.
* **Remove** — picks a `doc_id`, looks up every chunk whose metadata
  carries it via `collection.get(where={"doc_id": ...})`, and deletes
  those IDs. Returns the number of chunks removed.

## How dynamic ground-truth synth would plug in

The Update op invalidates the existing ground truth for any Query that
later targets the same `doc_id`. For now the runner:

1. Records every Update op as `outcome="stale-check"` in `OpRecord`.
2. Appends its `doc_id` to `WorkloadSummary.stale_check_doc_ids`.
3. Persists the list in `workload_summary.json` and emits
   `workload_stale_check_rate` to MLflow.

A downstream **dynamic ground-truth synth agent** (out of scope for this
worktree) can:

* Read `results/runN/workload_summary.json`.
* Filter to `stale_check_doc_ids` that a subsequent Query actually
  retrieved, then re-synthesise ground truth via the configured critic
  LLM.
* Augment the Ragas eval pass with the new `(question, answer,
  synthesised_ground_truth)` triples.

The seam in code is `benchmark/workload/runner.py:WorkloadSummary`
plus the `OpRecord.outcome == "stale-check"` flag — both are stable.

## Concurrency safety notes

* **ChromaDB**: writes are serialised on `benchmark.retrieval._chroma_lock`.
  Concurrent inserts/updates/removes will not corrupt the index. Reads
  also acquire the lock for now; if profiling shows contention, split
  into a separate read lock.
* **LanceDB**: the LanceDB adapter in `benchmark/retrieval.py` does not
  currently expose thread-safe mutation helpers. Workload mode is
  **Chroma-only** for now; running it against LanceDB will raise when
  the Update / Remove helpers try to call `collection.get(where=...)`.
  Extending LanceDB is left for future work.
* **Generator LLM**: the LangChain chat model returned by `get_llm` is
  treated as thread-safe (most providers' HTTP clients are). If a
  provider misbehaves under concurrency, set `workload_concurrency: 1`
  to serialise without disabling the workload entirely.
* **Open-loop arrival**: the runner honours each op's `arrive_at`
  offset. When the offered rate exceeds the pipeline's capacity, the
  queue depth grows and observed QPS falls short of `target_qps` —
  exactly the headroom signal RAGPerf §3.2 wants to surface.

## MLflow logging

All workload metrics are emitted with the `workload_` prefix on the
existing nested MLflow run:

* `workload_total_submitted`, `workload_total_completed`,
  `workload_total_errors`, `workload_total_stale_check`
* `workload_error_rate`, `workload_stale_check_rate`
* `workload_offered_qps`, `workload_observed_qps`, `workload_wall_seconds`
* Per op kind: `workload_<kind>_count`, `workload_<kind>_mean_seconds`,
  `workload_<kind>_p50_seconds`, `workload_<kind>_p95_seconds`,
  `workload_<kind>_p99_seconds`, `workload_<kind>_max_seconds`
* Queue depth: `workload_queue_depth_mean`, `..._p50`, `..._p95`,
  `..._p99`, `..._max`

A full `workload_summary.json` is written to `results/runN/` alongside
the rest of the run artefacts.

## Limitations

* **External RAG adapters**: workload mode is opt-in for the internal
  pipeline only. External adapters (`http`, `mcp`, managed) are skipped
  because their mutation surface is provider-specific; supporting them
  would require adapter-specific `insert`/`update`/`remove` contracts.
* **Insert pool**: when no pool is supplied, inserts use placeholder
  text. This is enough to exercise embedding + indexing throughput but
  does not stress retrieval recall. Curate a pool for realistic runs.
* **Zipf cache**: `zipfian_sample` rebuilds the CDF table on every call.
  Fine for corpus sizes in the low thousands; for very large pools,
  cache the CDF outside the sampler.
* **No closed-loop mode**: arrivals are open-loop only. A closed-loop
  (think-time) variant is not implemented.

## Tests

```
pytest tests/test_workload.py
```

Covers uniform + Zipfian sampling properties, op-mix honouring, seed
reproducibility, arrival-pattern scaling, and runner behaviour for each
op kind including the stale-check flag and error capture path.
