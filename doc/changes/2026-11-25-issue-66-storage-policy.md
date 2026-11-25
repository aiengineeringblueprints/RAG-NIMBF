# Single durable source of truth: MLflow record, temporary run dirs, safe cleanup

Date: 2026-11-25

## Summary

Implements the storage policy from issue #66: MLflow is the canonical durable
record for completed runs (parameters, metrics, provenance, canonical
per-sample artifacts); `results/runN/` is demoted to a temporary working area
for checkpointing/resumability; Git stores code and manifests only. Adds a
safe, opt-in cleanup CLI with dry-run default, removes duplicate full
per-sample payloads from aggregate MLflow runs, enables the `results/` Git
ignore rule, and makes the run output root configurable outside the checkout.

## Changes

- **Artifact deduplication** (`benchmark/tracking.py`):
  `log_aggregate_artifacts_to_mlflow` no longer uploads
  `benchmark_*.samples.json.gz` (full per-sample sidecars) to the aggregate
  run — that data lives canonically on each config's MLflow run. It accepts a
  new `config_run_ids` mapping and logs `reports/config_run_links.json` so
  aggregate runs link to the canonical config runs.
- **Run-ID provenance** (`benchmark/orchestration/worker.py`):
  `ProgressStore.mark_completed` takes `mlflow_run_id`; the worker records it
  per config in `progress.json` and passes `config_run_ids` to the aggregate
  logging call.
- **Safe cleanup** (new `benchmark/cleanup.py`):
  `verify_config_run` checks a run is `FINISHED` with parameters and
  artifacts; `plan_cleanup`/`cleanup_run_dir` select only configs whose
  progress record is `completed` with a verified MLflow run; dry-run is the
  default (`--execute` required to delete; `--prune-samples` for sidecars).
  CLI: `python -m benchmark.cleanup results/runN [--execute] [--prune-samples]`.
  `progress.json` and sweep summaries are never removed.
- **Configurable output root** (`benchmark/orchestration/worker.py`):
  `_next_run_dir` honours the new `BENCHMARK_RESULTS_DIR` env var so scratch
  output can live outside the Git checkout (`--run-dir` still overrides).
- **Git policy** (`.gitignore`): the previously commented-out `results/` rule
  is now active — generated results are never committed.
- **Docs** (`doc/results-and-mlflow.md`): new "Storage policy (issue #66)"
  section covering the canonical location per artifact class, backup
  implications (Git ignore ≠ disk reclaimed; both `mlflow.db` and the
  artifact store need backups), cleanup workflow, and the
  tracing-is-opt-in note.

## Behaviour changes and migration

- Aggregate MLflow runs written by new code contain fewer artifacts: the
  `*.samples.json.gz` sidecars are gone (retrievable from each config run's
  `data/` and `lineage/` artifacts). Old MLflow runs remain readable; nothing
  historical is deleted or rewritten.
- New worker runs record `mlflow_run_id` in `progress.json`; existing
  `progress.json` files without it simply never qualify for cleanup (safe
  default).
- `results/` is now ignored — anyone who intentionally versioned result files
  must use `git add -f` (not recommended) or move them.
- Set `BENCHMARK_RESULTS_DIR` to keep run output out of the checkout.

## Verification

- New `tests/test_storage_cleanup.py` (16 tests): aggregate dedup and run-ID
  linking, upload verification (finished/params/artifacts, missing runs),
  cleanup planning/execution, failed-config and unverified-upload protection,
  dry-run default, samples-sidecar pruning, `BENCHMARK_RESULTS_DIR`, and a
  `.gitignore` regression test.
- `tests/test_worker_loop.py` fakes updated for the new kwarg; suite green
  except `test_worker_run_one_cell_manifest_produces_artifacts_and_report`,
  which also fails on unmodified `dev` (pre-existing MLflow/sqlite issue,
  unrelated).

## Follow-ups

- The pre-existing `test_worker_run_one_cell_manifest_produces_artifacts_and_report`
  failure on `dev` (duplicate SQLite index) needs its own fix.
- MLflow tracing storage footprint on large sweeps could be measured once
  tracing-heavy sweeps exist.

## Review fixes (a61ce683, PR #73 review round 1)

- `ProgressStore.is_completed`: completed record with `mlflow_run_id` stays
  completed after local cleanup (no re-runs on resume).
- `verify_config_run`: requires `data/per_sample_results.csv` specifically.
- Per-sample CSV: added `contexts_json` column; aggregate report uploads get
  `samples_file` keys stripped so downloaded copies load cleanly.
- `--prune-samples`: only when `progress.json` exists and every completed
  config is verified; per-config `configs/<name>.samples.json.gz` added to
  cleanup suffixes.
- Worker seeds `config_run_ids` from the ledger (complete
  `config_run_links.json` on resume).
- `BENCHMARK_RESULTS_DIR` became the `BenchmarkConfig.run_results_dir` knob
  (env fallback + tests per AGENTS.md).
- Untracked `results/` from Git index (`git rm -r --cached`, 148 files);
  updated `doc/00-README.md` index entry and the storage-policy guide.

Verification: full suite 1170 passed; only the 3 known pre-existing flaky
failures (mlflow sqlite index race) remain, reproduced on plain dev too.
