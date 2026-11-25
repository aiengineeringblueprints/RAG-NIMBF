# Experiment lineage and paired run comparisons

Date: 2026-11-25

## Summary
Runs now record which experiment they belong to, which version of the system
under test they measured, which question set and corpus they used, and how
they were scored. A new `python -m benchmark.lineage compare` command pairs a
baseline and a candidate run by stable sample IDs and writes a Markdown/JSON
comparison with deltas, confidence intervals and an explicit comparability
status (issue #65).

## Changes

### Experiment metadata and system provenance
- `config.py`: new fields `experiment_metadata_json` and `system_provenance_json`
  (env `EXPERIMENT_METADATA_JSON`, `SYSTEM_PROVENANCE_JSON`), validated with
  unknown keys rejected so no secrets slip in.
- `benchmark/orchestration/matrix.py`: top-level manifest blocks `experiment:`
  and `system_provenance:` map onto those fields; dicts are coerced to JSON.
- `benchmark/lineage/provenance.py`: `SystemProvenance`, git capture
  (commit, branch, credential-free remote, dirty flag, SHA-256 diff hash; never
  the diff), `resolve_system_provenance` (adapter hook > manifest > not_available).
- Adapters: `InternalRagAdapter` captures this repository; the Blueprint
  adapters `enterprise_rag` / `enterprise_rag_mcp` capture the Blueprint
  checkout via `blueprint_provenance(config)`.
- `benchmark/reproducibility.py`: manifest gains `experiments`; `_run_git`
  takes a `cwd` and tolerates a missing git binary.

### Dataset identity
- `benchmark/lineage/dataset.py`: stable `sample_id` (hash of question, ground
  truth, occurrence), optional `category`, ordered question-set hash,
  order-independent corpus fingerprint, dataset identity incl. sampling seed.
- `benchmark/dataset.py`: sampling seed `42` is now the shared constant
  `SAMPLING_SEED` (unchanged value).

### Run records
- `benchmark/reporting/models.py`, `exports.py`: `PerSampleResult.sample_id` /
  `category`, `BenchmarkResultExtended.lineage`; both written to JSON reports.
- `benchmark/orchestration/runner.py`: builds the lineage once per config run.
- `benchmark/tracking.py`: `experiment.*`, `system.*`, `dataset.*`,
  `scorer.fingerprint` tags and `lineage/lineage.json` + `lineage/samples.json`
  artifacts (no question text).

### Comparison
- `benchmark/lineage/runs.py`: load a run from a report (`path[#config]`) or
  an MLflow run ID; per-sample metrics reuse `benchmark.regression.metrics`.
- `benchmark/lineage/compare.py`, `__main__.py`: status `paired` /
  `partially_paired` / `not_comparable` with reasons, warnings, per-metric
  baseline/candidate/delta/direction/n/missing counts/bootstrap CI,
  per-question deltas, category summary; optional MLflow logging.

## Behaviour changes and migration
None breaking. Reports and MLflow runs gain fields; old reports still load
(sample IDs are re-derived, absent lineage shows as unknown/not_available).
MLflow runs logged before this change cannot be compared by run ID — use their
JSON reports.

## Verification
New tests: `tests/test_lineage_provenance.py`, `test_lineage_dataset.py`,
`test_lineage_report.py`, `test_lineage_mlflow.py`, `test_lineage_compare.py`,
plus `TestExperimentLineageConfig` in `tests/test_config.py`. Full suite on the
DGX Spark: 1126 passed.

## Follow-ups
- No dashboard and no automatic baseline promotion (non-goals of #65).
- HTTP/MCP/managed systems only get provenance when the manifest declares it
  or a plugin implements `system_provenance(config)`.
