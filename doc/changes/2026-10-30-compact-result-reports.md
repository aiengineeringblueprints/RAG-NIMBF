# Compact benchmark result reports

Date: 2026-10-30

## Summary
Result reports grew to 124 MB (20 configs × 500 questions), above GitHub's
100 MB file limit. Retrieved contexts made up ~70 % of the size and repeat
across samples and configs. Reports are now split into a small summary JSON
and a gzipped per-sample sidecar with deduplicated contexts; a script shrinks
existing reports losslessly.

## Changes
- **Report layout** — `write_json_report` writes `benchmark_<ts>.json` (and
  `configs/<name>.json`) without `per_sample`, plus
  `<stem>.samples.json.gz`: `{"format": 1, "contexts": [...], "results":
  [[sample, ...], ...]}` where samples reference contexts by index
  (`context_ids`). The summary names its sidecar in `samples_file`.
  File: `benchmark/reporting/exports.py`.
- **Reader** — `load_json_report(path)` returns the old inline schema for both
  layouts. `scripts/compare_runs.py`, `scripts/backfill_mlflow.py` use it; the
  aggregate MLflow run also logs the sidecar (`benchmark/tracking.py`).
- **Migration tool** — `scripts/compact_results.py [roots] [--dry-run]`
  rewrites legacy reports via `compact_json_report`, which reloads the result,
  compares it with the original and restores the original text on mismatch.

## Behaviour changes and migration
Code reading `benchmark_*.json` directly no longer finds `per_sample`; use
`load_json_report`. Keep the `.samples.json.gz` next to its JSON when copying
results. Run `python scripts/compact_results.py` once per machine that holds
old results; compaction changes the file format, not the recorded data.

## Verification
- `tests/test_report_exports.py`: round-trip through the sidecar, 100 samples
  with a shared 10 KB context stay < 30 KB, legacy reports load unchanged,
  compaction is lossless and idempotent, results without samples survive.
- `tests/test_worker_loop.py` reads per-config reports via `load_json_report`.
- Full suite: 756 passed.
- Real data (copies): run95 28.8 MB → 27 KB + 1.06 MB, run93 8.4 MB → 0.44 MB,
  reloaded report identical to the original. Measured on the Spark:
  deepseek-v4-flash/run2 124 MB → ~7 MB.

## Follow-ups
- `configs/*_qa.json` and `*_checkpoint.json` keep their own format.
