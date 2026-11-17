# Ground-truth expectations on MLflow traces

Date: 2026-11-17

## Summary
Every per-sample trace of a tracing-enabled run (issue #42) now carries the
sample's ground truth as MLflow expectations: the reference answer and, where
the dataset has them, the gold documents. Traces from a run can therefore be
scored later (e.g. `mlflow.genai.evaluate` on `mlflow.search_traces(run_id=...)`)
without re-running the pipeline. Issue #43, parent #40.

## Changes

### Expectations per trace (`benchmark/tracing.py`)
- `sample_expectations(sample)` maps a benchmark sample onto MLflow's reserved
  expectation names:
  - `expected_response` — `ground_truth` (skipped when empty).
  - `expected_retrieved_context` — `[{"doc_uri", "content"}]` for a single
    `metadata.gold_doc_id` (content = the sample context, which is that gold
    document); ids only (`[{"doc_uri"}, ...]`) for `metadata.gold_doc_ids`
    (multi-hop, whose context mixes in distractors). `doc_uri` matches the
    RETRIEVER span document `id`.
- `log_sample_expectations(samples_by_trace, source_id=...)` flushes async
  trace logging first (an expectation needs the persisted trace), then writes
  the expectations with source `CODE` / `dataset:<dataset_name>`. Write
  failures log a warning and never fail the benchmark.

### Runner (`benchmark/orchestration/runner.py`)
- The per-sample loop collects `(trace_id, sample)` pairs; after the loop
  `log_sample_expectations` replaces the bare `flush_traces()` call. Ground
  truth is attached only after the sample finished, so adapters never see it.
  Checkpoint-resumed (cached) samples get expectations too.

## Behaviour changes and migration
None for runs with `mlflow_tracing_enabled: false`. Tracing-enabled runs write
up to two extra assessments per trace into the tracking store.

## Verification
- New tests in `tests/test_mlflow_tracing.py`:
  `test_every_trace_carries_the_reference_answer`,
  `test_gold_context_and_doc_ids_become_expected_retrieved_context`,
  `test_run_traces_are_scoreable_without_rerunning` (run-linked
  `search_traces` returns inputs, outputs, RETRIEVER documents, expectations).
- Full suite on the DGX Spark (MLflow 3.18): 964 passed, 3 failed. The same 3
  fail on `dev` without this change (961 passed): `test_hybrid_defaults`
  (Spark `.env` sets `mmr`), `test_worker_pipeline_runs_parsed_corpus_unchanged`
  (Spark `.env` points `MLFLOW_TRACKING_URI` at the `mlruns` file store, which
  MLflow 3.18 rejects), and an order-dependent
  `test_internal_and_stubbed_external_runs_are_structurally_identical`
  (passes alone).
- Smoke run on the Spark: `experiments/mlflow-tracing-smoke.yaml`,
  `--run-dir results/run-expectations-smoke` (3 SQuAD questions, similarity +
  hybrid). Both config runs tagged `linked_3`; `search_traces(run_id=...)`
  returned 3 traces each, every one with question input, response output,
  exactly one RETRIEVER span, `expected_response` and a single-document
  `expected_retrieved_context` with gold text; the gold `doc_uri` appeared
  among the retrieved document ids in all 6 traces.

## Follow-ups
- Gold annotations derived only from `supporting_facts` / `supporting_contexts`
  titles (no `gold_doc_ids`) are not attached; multi-hop loaders already
  resolve them into `gold_doc_ids`.
- Scoring the stored traces with MLflow GenAI judges is the next #40 slice.
