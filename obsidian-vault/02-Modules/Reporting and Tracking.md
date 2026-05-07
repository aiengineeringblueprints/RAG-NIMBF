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

