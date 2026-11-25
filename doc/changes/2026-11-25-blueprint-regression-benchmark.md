# Fixed regression benchmark for iterative Blueprint evaluation

Date: 2026-11-25

## Summary
Adds a bounded, repeatable evaluation for Enterprise RAG Blueprint changes
(issue #64): a versioned TechQA question set with a separate holdout split, an
immutable reference baseline plus current-best and one-change candidate
manifests, and a `benchmark.regression` module that compares runs per
evaluation dimension with paired confidence intervals and an explicit
accept / reject / inconclusive verdict. Full sweeps are no longer needed to
judge a single change.

## Changes

### Question set (`datasets/blueprint_regression/v1/`)
- `manifest.json` (committed): source, split sizes, categories and sha256 of
  `dev.jsonl` (500) and `holdout.jsonl` (200). The split files are rebuilt
  deterministically with `freeze` and gitignored (~30 MB, mostly documents).
- `dev` = the first 500 rows of RAGBench TechQA `train` after
  `shuffle(seed=42)` — the same questions as runs 7–9. `holdout` = the next
  rows whose question is not in `dev`.
- Each row stores `metadata.category` (`answerable` / `unanswerable` from
  `all_relevant_sentence_keys`); gold doc IDs are re-derived on load.

### `benchmark/regression/`
- `freeze_question_set()` → dev/holdout split + manifest; `sample_keys()`
  identifies samples by (question, reference answer, occurrence), because
  TechQA repeats every question with a different reference.
- `sample_metrics()` / `summarize_run()`: per-dimension metrics
  (correctness, faithfulness, retrieval, abstention, latency, cost); missing
  data is reported as `unavailable` with a reason, never as 0. Retrieval
  metrics only on answerable questions with gold labels; abstention via
  `benchmark.answer_match.declines`.
- `compare_runs()`: paired deltas, 95% CI from a cluster bootstrap over
  questions (TechQA's duplicate questions are resampled together), exact
  McNemar for 0/1 metrics, verdict from criteria (primary metric CI, `min_improvement`,
  guardrails as absolute or percentage worsening; a breached guardrail
  rejects even without a primary metric; a fallback primary metric is
  flagged as `primary_fallback`). Refuses runs that cover
  different questions or not the full frozen split.
- CLI `python -m benchmark.regression freeze | verify | compare`; `compare`
  reads report JSONs or `.samples.json.gz` sidecars and writes
  `comparison.json` + `report.md`. `freeze` aborts if the HF split is not
  larger than the request (the loader skips the shuffle then) and, on a
  directory holding only `manifest.json`, rebuilds the splits only if their
  hashes match.

### Manifests (`experiments/blueprint-regression/`)
- `baseline.yaml` (immutable reference: recursive 1000/200, similarity,
  top-k 12, `qwen3.8-27b`, `nomic-embed-text:latest`), `current-best.yaml`
  (run7 winner), `candidate-hybrid.yaml` (baseline with hybrid retrieval),
  `smoke.yaml` (5 dev questions, RAGAS off), `criteria.yaml`.

### Docs
- New guide `doc/blueprint-regression-benchmark.md` (protocol, metric/label/
  judge per dimension, improvement criteria, holdout rule); linked from
  `doc/00-README.md` and `doc/enterprise-rag-blueprint-results.md`.
- `.gitignore`: split files of `datasets/blueprint_regression/*/`.

## Behaviour changes and migration
None. New module, manifests and docs only. Before a run on a new machine,
rebuild the split files (`freeze` command in the guide) and `verify`.

## Verification
- `tests/test_regression_benchmark.py`: 27 tests (freeze, summary, compare,
  cluster bootstrap, verdict incl. guardrail-only reject and fallback flag,
  CLI freeze/verify/compare incl. missing split files, manifest one-change
  rule, criteria metric names); full suite on the DGX Spark: 1065 passed.
- `freeze` on the Spark twice: identical hashes; `dev` sample keys equal
  run8's 500 samples exactly.
- Smoke run (`smoke.yaml`) on the Spark: completed; `bert_score_f1` 0.576,
  `relevant_sentence_recall` 0.914, mean latency 1.52 s; faithfulness,
  abstention and cost correctly `unavailable`.
- `compare` on run8 cells: query refinement on vs off (hybrid alpha 0.5):
  `bert_score_f1` −0.056 (CI −0.098..−0.016), false refusals +0.066
  (McNemar p < 0.001) → reject.

## Follow-ups
- Run `baseline.yaml` and `current-best.yaml` (full, RAGAS on) on `dev`, then
  `candidate-hybrid.yaml`; create follow-up issues per candidate from the
  evidence (refinement off, chunking, top-k, reranker, refusal prompt).
- Blueprint adapter reports no token usage → cost dimension unavailable.
- Served model weights are tagged, not digest-pinned; record them per run.
- `current-best.yaml` (run7 winner) was selected on the dev questions; confirm
  it on holdout before using it as the best reference.
- PR review follow-up: see
  [2026-11-25-regression-review-fixes](2026-11-25-regression-review-fixes.md).
