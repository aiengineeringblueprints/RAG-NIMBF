# MLflow DB cleanup on the Spark

Date: 2026-11-17

## Summary
One-off cleanup of the Spark's `mlflow.db` after the MLflow 3.18 upgrade
(#41): the file had grown to 6.6 GB (2.9 GB free pages left by the schema
migration), the UI was cluttered with failed/stuck runs and heavy aggregate
traces. Follow-up code changes are tracked in #47–#50 (parent #40).

## Changes

### Spark `mlflow.db` (data only, no code)
- Backup: `mlflow.db.backup-20261117-pre-cleanup` (compact copy via `VACUUM INTO`, 3.67 GB).
- Runs (45 in `FAILED`/`RUNNING` before):
  - 33 childless runs without metrics deleted and hard-deleted with `mlflow gc` (incl. stale parent `benchmark_enterprise-rag-blueprint-rag-gold-run7_run7`, which had no children).
  - 4 stuck `RUNNING` runs set to `KILLED` (2 parents with children, 2 config runs with partial metrics from June/July).
  - 8 `FAILED` parent runs with children kept so the hierarchy stays intact.
- Traces: 1,112 aggregate traces deleted (`custom_metrics`, `ragas_evaluation`,
  `chunk_documents`, `build_vector_store`; 1.2–2.6 MB per span, no analytical
  value beyond the run metrics). 117,953 per-sample `retrieve` /
  `generate_answer` traces kept.
- `VACUUM`: 6.6 GB → 1.96 GB.

### Docs
- `doc/results-and-mlflow.md`: new "MLflow DB maintenance" section (backup, `mlflow gc`, `VACUUM`, closing stuck runs).

## Behaviour changes and migration
None for code. On the Spark, the deleted runs and traces are gone from the UI;
restore from `mlflow.db.backup-20261117-pre-cleanup` if one is needed.
`mlruns/` artifacts of the 33 hard-deleted runs were removed by `mlflow gc`.

## Verification
- `pragma quick_check` → ok; `mlflow.search_runs` returns all 376 remaining runs.
- Final state: 364 FINISHED, 8 FAILED (parents), 4 KILLED; only `retrieve` and
  `generate_answer` spans remain.

## Follow-ups
- #47 one MLflow experiment per manifest (everything currently lands in `RAG-Benchmark`).
- #48 one span-tree trace per sample with token usage (Tokens column is empty today; Enterprise RAG adapter drops Blueprint usage).
- #49 stop emitting aggregate traces so the DB doesn't fill up again.
- #50 evaluation datasets + `mlflow.genai.evaluate` on per-sample traces (Datasets / Evaluation runs views).
- The laptop's `mlflow.db` was not touched.
