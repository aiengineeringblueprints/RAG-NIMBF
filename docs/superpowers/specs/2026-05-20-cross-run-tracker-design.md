# Cross-Run Metric Tracker & Visualization

## Goal

CLI tool that scans all benchmark result JSONs in `results/`, aggregates metrics across runs, and generates one Seaborn plot per metric comparing all runs.

## Data Source

Scan `results/*/benchmark_*.json` files. Each JSON contains:
- Top-level: `timestamp`, `num_configs`, `dataset`, `system_info`
- `results[]` array with flat per-config entries:
  - Config params: `llm_model`, `chunk_size`, `chunk_overlap`, `chunking_strategy`, `prompt_template`, `embedding_model`, (optional) `retrieval_strategy`, `reranker_model`
  - RAGAS metrics: `ragas_faithfulness`, `ragas_answer_relevancy`, `ragas_answer_correctness`, `ragas_context_precision`, `ragas_context_recall` (nullable)
  - Performance: `avg_ttft_seconds`, `avg_tokens_per_second`, `total_time_seconds`
  - Optional: `custom_metric_means` dict with IR/NLG metrics

## Module

**File**: `benchmark/reporting/run_tracker.py`
**CLI**: `python -m benchmark.reporting.run_tracker`

### Functions

1. **`scan_all_results(results_dir: Path) -> pd.DataFrame`**
   - Glob `results/*/benchmark_*.json`
   - Parse each JSON, flatten all `results[]` entries
   - Add `run_id` from directory name
   - Return DataFrame with one row per config-result

2. **`plot_metric_over_runs(df, metric, output_dir) -> Path`**
   - X-axis: configs sorted by (run_id, config_name)
   - Y-axis: metric value
   - Hue: LLM model
   - Seaborn whitegrid, deep palette, 300 DPI PNG + SVG

3. **`plot_overview_grid(df, output_dir) -> Path`**
   - Subplot grid with key metrics

4. **`plot_ranking(df, output_dir) -> Path`**
   - Top-20 configs by composite score (reuse analysis.py weights)

5. **`main(results_dir, output_dir)`**
   - Orchestrates scan + all plots

### Metrics to Plot

**RAGAS**: faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall
**Performance**: avg_ttft_seconds, avg_tokens_per_second, total_time_seconds
**Custom** (if available): hit@1, ndcg@1, rouge_l, meteor, bert_score_f1, context_relevance

### Output

All plots to `results/cross_run_plots/`:
- `metric_faithfulness.png` (one per metric)
- `overview_grid.png`
- `ranking_top20.png`

## Dependencies

- `seaborn` (add to requirements.txt)
- `pandas`, `matplotlib`, `numpy` (existing)
