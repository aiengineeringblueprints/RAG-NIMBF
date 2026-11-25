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
  order-independent corpus fingerprint, dataset identity incl. adapter,
  source, version (local file hash), sampling method and seed (only for
  seeded shuffles); `SampleIdentity` (`sample_id`, `question_id`, `category`).
- `benchmark/dataset.py`: sampling seed `42` is now the shared constant
  `SAMPLING_SEED` (unchanged value) plus `sampling_method()`, which names how
  each loader subsamples (`none`, `head`, `seeded_shuffle`, `unknown`).
  Loaders stamp `metadata.dataset_rows` (source row count) and, for
  HuggingFace, `metadata.dataset_split` (resolved split) on every sample, so a
  sample size equal to the source size is recorded as `none`, not as a seeded
  shuffle, and default-resolved splits are recorded.

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
- `benchmark/lineage/compare.py`: status `paired` / `partially_paired` /
  `unverified` / `not_comparable` with reasons, warnings, per-metric
  baseline/candidate/delta/direction/n/question clusters/missing counts/
  bootstrap CI clustered by question, per-question deltas, category summary,
  per-side error counts (invalid answers, evaluation error). A scorer or
  corpus mismatch blocks the comparison; an unknown scorer or corpus identity
  on one or both sides makes it `unverified`; a candidate without, or with another,
  `baseline_run_id` is flagged.
- `benchmark/lineage/report.py` (Markdown/JSON output),
  `benchmark/lineage/publish.py` (MLflow run linked to the input run IDs),
  `__main__.py` (CLI).
- `benchmark/lineage/experiment.py`: scorer identity includes critic max
  tokens, thinking control, MLflow scorers and the evaluator implementation
  (hash of the framework's evaluator modules + scorer library versions);
  `candidate_id` requires a non-empty `baseline_run_id`.
- `benchmark/lineage/json_object.py`: shared validator for the two JSON
  lineage knobs. `benchmark.reproducibility.run_git` and
  `benchmark.regression.metrics.RETRIEVAL_PREFIXES` are now public.
- `benchmark/lineage/record.py`: framework commit in the lineage record;
  `run_dir/reproducibility/lineage/<config>.json` holds lineage + sample IDs.
- `benchmark/adapters/base.py`, `doc/Benchmark_Your_RAG.md`: optional
  `system_provenance(config)` adapter hook documented.

## Behaviour changes and migration
Reports and MLflow runs gain fields; old reports still load (sample IDs are
re-derived, absent lineage shows as unknown/not_available) and compare as
`unverified` against new runs. Manifests that set `candidate_id` must also set a
non-empty `baseline_run_id`. Loaded samples carry two extra metadata keys
(`dataset_rows`, `dataset_split`).
MLflow runs logged before this change cannot be compared by run ID — use their
JSON reports.

## Verification
New tests: `tests/test_lineage_provenance.py`, `test_lineage_dataset.py`,
`test_lineage_report.py`, `test_lineage_mlflow.py`, `test_lineage_compare.py`,
plus `TestExperimentLineageConfig` in `tests/test_config.py`. Full suite on the
DGX Spark: 1147 passed.

## Follow-ups
- No dashboard and no automatic baseline promotion (non-goals of #65).
- HTTP/MCP/managed systems only get provenance when the manifest declares it
  or a plugin implements `system_provenance(config)`.
- Not yet covered from #65: post-run MLflow scorer duration/failures/coverage
  in the comparison; dataset-native source IDs (sample IDs are content
  hashes); HuggingFace dataset revisions are not pinned, so their `version`
  is unknown; lineage is rebuilt per config (corpus hash + git calls) and
  could be cached per run.
