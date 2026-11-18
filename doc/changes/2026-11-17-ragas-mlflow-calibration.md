# Calibration: direct RAGAS vs. MLflow scorers

Date: 2026-11-17

## Summary
Adds the tooling and the experiment for issue #46 (parent #40): a manifest
that judges one config with both direct RAGAS and the post-run MLflow scorers
using the same local critic, a report that pairs both evaluators sample by
sample, and the measured result with a decision
([doc 14](../14-ragas-vs-mlflow-calibration.md)).

## Changes

### Judge runtime
- `benchmark/mlflow_scorers.py`: `score_run_traces` logs the wall time of the
  judge step as `mlflow_judge_seconds` (`RUNTIME_METRIC`) on the config run.
  The returned dict still holds only `mlflow_judge_<name>_mean`. Needed to
  compare with the direct RAGAS stage (`stage_ragas_eval_seconds`).
- It also tags the run with the configured scorer list
  (`mlflow_scorer_names`, `SCORER_NAMES_TAG`), so the report can show a
  scorer that failed on every sample instead of silently dropping it.

### Calibration report
- New `benchmark/reporting/judge_calibration.py`
  (`python -m benchmark.reporting.judge_calibration`):
  - `pair_stats(direct, judge)`: paired samples, means, mean Δ
    (judge − direct), mean |Δ|, Pearson, Spearman (`None` when undefined, e.g.
    a constant yes-judge), failure rate per side.
  - `judge_scores(run_id, names, n_samples)`: reads the run-linked traces and
    places each scorer's newest (non-invalidated) assessment at the trace's
    `benchmark.sample_index`. `ASSESSMENT_NAMES` maps config names to
    assessment names (RAGAS wrappers use the metric class name, e.g.
    `Faithfulness`; `retrieval_relevance` uses the per-span
    `retrieval_relevance/precision`). yes/no → 1/0; errors → missing.
  - `METRIC_PAIRS`: direct RAGAS metric ↔ MLflow scorer (same-math RAGAS
    wrappers, custom `answer_correctness`, and the binary built-in proxies).
  - `build_report(config_json, run_id)` + `render_markdown` + CLI writing
    Markdown and JSON. A pair is reported when direct RAGAS scored the metric
    and the scorer was configured on the run (or produced any score); a
    scorer with 100 % failure shows up with its failure rate.

### Manifest and docs
- `experiments/ragas-mlflow-calibration.yaml`: 50 SQuAD questions (seed-42
  slice), internal pipeline, generator `openai:qwen3.8-27b`, critic
  `ollama:gemma3:12b` (temperature 0) for both paths, direct RAGAS
  (5 metrics), tracing, and 10 MLflow scorers.
- `doc/14-ragas-vs-mlflow-calibration.md`: results and decision.
- `doc/results-and-mlflow.md`: `mlflow_judge_seconds`, calibration CLI.
- `doc/00-README.md`: index entry for doc 14.

### Review follow-ups (PR #55)
- `_sample_index` takes the first root span that carries
  `benchmark.sample_index` instead of giving up on the first root span.
- `--out pass.2` writes `pass.2.md` / `pass.2.json` (only a `.md`/`.json`
  suffix is replaced).
- Doc 14 §6: re-score whole scorers, not subsets (newest assessment wins).
- `mlflow.search_traces(max_results=None)` already pages through all
  traces (MLflow 3.18), so large runs are not truncated; no change.
- `results/run100/`: the calibration run (Spark), both report passes, and
  `rescore_pass2.py` used for the throttled re-score.

## Behaviour changes and migration
A scored config run carries one extra metric, `mlflow_judge_seconds`, and
one extra tag, `mlflow_scorer_names`. No
config changes; the default evaluator stays direct RAGAS.

## Verification
- New tests: `tests/test_judge_calibration.py` (pair statistics, reading
  assessments from traces, report pairing, 100 %-failure scorer, CLI) and
  `test_judge_wall_time_is_logged_on_the_config_run` in
  `tests/test_mlflow_scorers.py`. Full suite on the DGX Spark: 1020 passed.
- Calibration run on the Spark: `results/run100`, MLflow run
  `ac134264f76b4961bb1b6f877de6ce52`, 803 s worker run.
  - Pass 1 (MLflow default concurrency, 10 rows × 10 scorers): 38–100 %
    scorer failures from Ollama read/scorer timeouts; direct RAGAS 0 %.
  - Pass 2 (8 rows × 1 scorer, longer timeouts): 0 % failures. No pair
    meets mean |Δ| < 0.05 and Spearman > 0.9. Direct RAGAS 788 s
    (15.8 s/sample), MLflow scorers 1929 s (38.6 s/sample).
  - Decision: side by side, direct RAGAS stays the default (no ADR). Details
    in doc 14.

## Follow-ups
- Expose MLflow scorer concurrency/timeouts as manifest settings instead of
  `MLFLOW_GENAI_EVAL_*` env vars; the worker currently runs with the
  flooding defaults.
- A CLI to re-score an existing run's traces (today: a small script, see
  `results/run100/rescore_pass2.py`).
- Repeat on a harder dataset (e.g. RAGBench TechQA); SQuAD scores sit near
  the ceiling, which depresses the rank correlations.
- Doc 13 (`13-ragas-vs-mlflow-genai-judges.md`, research for #40) is not on
  `dev` yet; doc 14 cites its §6.5 criterion.
