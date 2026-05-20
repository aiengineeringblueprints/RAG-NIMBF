# Reporting and Tracking

Sources:

- [benchmark/reporting/](../benchmark/reporting)
- [benchmark/tracking.py](../benchmark/tracking.py)
- [benchmark/tracing.py](../benchmark/tracing.py)

Reporting outputs:

- JSON report.
- CSV report.
- Markdown report.
- Terminal tables and insights.
- Static Matplotlib plots.
- Interactive Plotly HTML visualizations.

Cross-run paper/poster plots:

- `python -m benchmark.reporting.run_tracker` scans `results/*/benchmark_*.json` plus `results_summary.csv` custom metrics and writes plots to `results/cross_run_plots/`.
- Main paper/poster figures use the `paper_*` prefix and are intended for publication surfaces rather than exhaustive debugging.
- `paper_quality_ranking_top10.png` ranks configurations with a timing-free composite quality score: Faithfulness, Context Relevance, NDCG@5, Recall@5, Hit@1, and BERTScore F1. Metrics must have at least 50% cross-run coverage, and each ranked row needs at least 70% of available composite weight.
- `paper_quality_vs_load_proxy.png` plots quality against indexed chunk count as a computational/load proxy because TTFT is not available for most runs.
- `paper_faithfulness_by_llm_prompt.png` shows Faithfulness separately as a mean plus 95% CI dot plot by LLM and prompt template because this RAGAS metric has full cross-run coverage; low outliers are summarized in the caption area rather than stretching the axis.
- `paper_llm_metric_heatmap.png` replaces radar profiles with an LLM-by-metric heatmap and row-level `n` counts.
- `paper_retrieval_by_strategy.png` compares Hit/NDCG/Recall at k=1/3/5 by chunking strategy.
- `paper_timing_subset.png` keeps runtime analysis separate and labels it as the available timing subset; timing is not used in the main quality score.

Ranking/analysis:

- `compute_rankings()` normalizes selected metrics and ranks configurations.
- Terminal and Markdown reports include performance, RAGAS, custom metric, and insight summaries.

MLflow:

- `setup_mlflow()` configures local tracking.
- `log_benchmark_run()` logs aggregate metrics, config params, tags, and per-sample CSV.
- `log_genai_eval()` logs model-evaluation style records.
- `log_plots_to_mlflow()` stores generated plots as artifacts.

Tracing:

- `setup_tracing()` configures tracing endpoints when relevant environment variables are present.
- `setup_langfuse()` provides optional Langfuse integration.

Related notes:

- [[Operations Runbook]]
- [[Benchmark Pipeline]]
- [[Evaluation and Metrics]]

