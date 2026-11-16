# RAGAS metric selection and critic worker guidance

Date: 2026-11-16

## Summary
RAGAS with the critic LLM takes ~3 of ~3.5 h per TechQA config (issue #19).
A new `ragas_metrics` knob selects which RAGAS metrics run, so sweeps can skip
the expensive `context_precision` (one critic call per retrieved context).
`EVAL_CRITIC_MAX_WORKERS` 8 vs 16 was measured against the hosted critic and
documented with a recommendation in the new guide `doc/ragas-runtime.md`.

## Changes

### Metric selection knob
- `config.py`: `RAGAS_METRIC_NAMES` (the six metrics, in the previous order),
  `BenchmarkConfig.ragas_metrics` (default: full set), env fallback
  `RAGAS_METRICS` (comma-separated; blank = full set), `parse_ragas_metrics`.
  `validate_benchmark_config` rejects unknown and duplicate names and an empty
  selection while `ragas_enabled`.
- `benchmark/orchestration/matrix.py`: manifest `ragas_metrics` accepts a list
  or a comma string.
- `benchmark/evaluation.py`: `build_ragas_metrics(metric_names)` builds fresh
  instances of only the selected metrics; `evaluate_results(...,
  metric_names=...)` passes them to RAGAS. Unselected metrics are never built,
  so they cost no critic calls.
- `benchmark/orchestration/runner.py`: passes `config.ragas_metrics`; the
  console line lists the metrics that run.

### Skipped metrics stay absent, never 0
- Results already carry `None` for metrics without scores (MLflow skips them,
  JSON `null`, CSV empty cell).
- `benchmark/reporting/exports.py`: the markdown RAGAS table drops columns no
  config computed (was `N/A`); the section disappears when RAGAS was off.
- `benchmark/reporting/run_tracker.py`: new `composite_ragas_score`
  renormalises weights per row over the metrics a row has. Used by
  `plot_ranking` and `paper_charts.plot_composite_ranking`, which previously
  filled missing metrics with 0 and so penalised runs with fewer metrics.
- `benchmark/reporting/ngen_ai_charts.py`: no bar instead of a 0 bar for a
  metric without values; the three chart functions share
  `metric_mean_and_error` (PR #38 review).
- `benchmark/reporting/visualization.py` (PR #38 review): new
  `ragas_plot_matrix` drives the RAGAS comparison bars, radar and heatmap.
  Metrics no config ran are left out; a metric missing for one config is NaN:
  no bar, a gap in the radar, a grey `N/A` heatmap cell (was 0 / red).
- PR #38 re-review: the interactive LLM radar
  (`llm_comparison_radar_figure`, written by
  `_plot_llm_comparison_radar_html`) averages each metric only over configs
  that measured it and leaves a gap for metrics an LLM never measured (both
  were 0). `metric_mean_and_error` no longer drops genuine 0.0 scores; missing
  values are recognised by null only.

### Docs
- New `doc/ragas-runtime.md`: knob usage, per-metric cost, worker measurement
  and recommendation, `sample_size` guidance. Indexed in `doc/00-README.md`.
- `doc/env-vars.md`: `RAGAS_METRICS` row; `EVAL_CRITIC_MAX_WORKERS` row
  recommends 16 for the hosted critic.
- `doc/enterprise-rag-techqa-refusal-analysis.md`: next-step 6 updated with
  the results.

## Behaviour changes and migration
- Default unchanged: all six metrics run in the same order.
- `ragas_metrics: []` / `RAGAS_METRICS=` run the full set; use
  `ragas_enabled: false` to skip RAGAS.
- Put `ragas_metrics` under `settings:`. Under `matrix:` a list expands into
  one config per metric.
- Resume (`worker --run-dir`) matches configs by name, which does not include
  the metric selection; use a new run directory after changing it.
- Composite rankings across runs with different metric sets change: missing
  metrics no longer count as 0.
- NGEN paper charts now include genuine 0.0 RAGAS scores in bar means (they
  were filtered out as presumed placeholders). Old result files that stored
  a missing RAGAS score as 0 will now pull those means down.
- `evaluate_results` raises `ValueError` for unknown metric names, also when
  called with no questions.

## Verification
- Tests: `tests/test_config.py::TestRagasMetricSelection` (default, env
  fallback, blank env, manifest list/string, unknown, duplicate, empty),
  `tests/test_evaluation.py::TestEvaluateResultsMetricSelection`,
  `tests/test_run_path_uniformity.py::test_skipped_ragas_metrics_stay_absent_not_zero`,
  `tests/test_report_exports.py::test_markdown_report_omits_ragas_metrics_that_were_not_run`,
  new `tests/test_ragas_composite.py`, `tests/test_ragas_plot_data.py`
  (data helpers plus rendered output: bar heights, heatmap mask, interactive
  radar traces). Full suite: 884 passed.
- Critic replay, first 10 TechQA samples of run6, `deepseek-v4-flash`
  (details in `doc/ragas-runtime.md`):

  | Workers | Metrics | Wall time | Critic input tokens |
  |---|---|---|---|
  | 8 | all 6 | 224 s | 592 k |
  | 16 | all 6 | 184 s | 592 k |
  | 16 | without `context_precision` | 105 s | 332 k |

## Follow-ups
- Worker scaling measured once on 10 samples; confirm 16 (or try 24) on a
  full 150-question sweep and watch for 429/5xx retries.
- Run identity for resume does not include `ragas_metrics` (or other
  non-name settings).
- Decision: CSV/JSON keep columns/keys for skipped metrics as empty/`null`
  (stable schema for `run_tracker` scans and downstream tools; no false
  zeros). "Absent" in issue #19 is read as "no value", not "no key".
