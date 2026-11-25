# Regression benchmark: PR #68 review fixes

Date: 2026-11-25

## Summary
Addresses the review on PR #68 (issue #64): `compare` now refuses runs
scored with a different judge or dataset and split files that no longer
match their manifest; the `current best` is mandatory; conflicting
answerable labels no longer distort abstention; the binary test counts
repeated questions once; the criteria are a typed, validated policy with
their bootstrap settings; the module is split into a package.

## Changes

### `benchmark/regression/` (was `benchmark/regression.py`)
- Split into `question_set.py` (freeze, sample keys, categories, split hash
  check), `metrics.py` (per-sample metrics, run summary), `criteria.py`
  (`RegressionCriteria`, `Guardrail`), `compare.py` (`Paired`, verdict),
  `runs.py` (run loading, config check), `report.py`, `cli.py`. Public
  imports from `benchmark.regression` and `python -m benchmark.regression`
  are unchanged.
- `compare` reads each run's `reproducibility/manifest.json` and refuses
  runs that differ in dataset settings, judge (`eval_critic_llm`,
  `eval_critic_embedding`), BERTScore model, retrieval-metric mode or RAGAS
  settings (`check_run_configs`, status `invalid`). Runs without a single
  recorded config are compared with a warning (`unverified`). The report
  lists which system settings the candidate changes against each reference.
- `compare` checks the split file's sha256 against the manifest before
  comparing; `--best` is required.
- Questions with conflicting answerable labels (TechQA duplicates) are
  `ambiguous` and count for neither `abstained` nor `false_refusal`
  (`n_ambiguous` in the summary). Derived at read time, so the frozen v1
  hashes are unchanged.
- 0/1 metrics: exact sign test over questions (one vote per question, by
  the sign of its mean change) instead of a row-level McNemar that counted
  both TechQA copies of a question.
- `criteria.yaml` is loaded into `RegressionCriteria`: unknown keys and
  guardrail limits fail; `n_boot` and `seed` moved from CLI flags into the
  file and are recorded in the report.

### Docs and manifests
- ADR [006](../adr/006-regression-criteria-are-an-analysis-policy.md):
  criteria are an analysis policy, not a `BenchmarkConfig` knob.
- `criteria.yaml`: threshold rationale (only `min_improvement` is derived
  from data; guardrails are judgment calls to recalibrate after a repeated
  baseline run).
- `baseline.yaml` and the guide: model digests at freeze time
  (`nomic-embed-text:latest` = `0a109f422b47`) and the re-run rule after
  model updates.
- Guide `doc/blueprint-regression-benchmark.md`: compare refusal rules,
  ambiguous rows (dev 16 rows / 8 questions, holdout 4 / 2), threshold
  rationale, model versions, follow-up issues #69–#71.

## Behaviour changes and migration
- `compare --n-boot/--seed` removed: set `n_boot` / `seed` in the criteria
  file.
- `compare` without `--best` now fails.
- `compare` exits 1 on a modified split file or mismatched evaluation
  setup.
- `abstained` / `false_refusal` exclude ambiguous rows: on v1 dev 444
  answerable and 40 unanswerable rows are scored (was 452 / 48).

## Verification
- `tests/test_regression_benchmark.py`: 37 tests on the DGX Spark (new:
  config check invalid/ok/unverified, CLI refusal on judge mismatch and on a
  modified split, `--best` required, ambiguous labels, sign test counts
  repeated questions once, criteria validation and bootstrap settings).
- `compare` on run8 (refinement on vs off, hybrid alpha 0.5): reject;
  `bert_score_f1` −0.056 (CI −0.098..−0.016), false refusals +0.063 on 444
  rows (sign test p < 0.001); `config_check` `unverified` (matrix run).

## Follow-ups
- #69 reference runs on dev + holdout confirmation + noise recalibration.
- #70 hybrid candidate.
- #71 Blueprint token usage for the cost dimension.
