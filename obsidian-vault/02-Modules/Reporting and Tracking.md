# Reporting and Tracking

Sources:

- [benchmark/reporting/](../benchmark/reporting)
- [benchmark/tracking.py](../benchmark/tracking.py)
- [benchmark/tracing.py](../benchmark/tracing.py)
- [benchmark/reproducibility.py](../benchmark/reproducibility.py)

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

- `setup_mlflow()` configures local tracking and can enable MLflow system metrics via `MLFLOW_ENABLE_SYSTEM_METRICS=true`.
- `log_benchmark_run()` logs aggregate metrics, config params, tags, per-sample CSV, and the reproducibility bundle.
- `log_genai_eval()` logs model-evaluation style records.
- `log_aggregate_artifacts_to_mlflow()` creates a run-level MLflow summary containing CSV tables, JSON/Markdown reports, PNG/HTML plots, and reproducibility artifacts.
- `log_plots_to_mlflow()` remains as a backward-compatible wrapper around aggregate artifact logging.

Reproducibility:

- Each deterministic run writes `results/runN/reproducibility/manifest.json` and `packages.txt`.
- The manifest captures UTC timestamp, Python/platform details, argv/cwd, git commit/branch/dirty status, selected benchmark environment variables with secrets redacted, and the full validated config grid.
- The same reproducibility directory is logged to every per-config MLflow run and the aggregate summary run under the `reproducibility` artifact path.
- MLflow runs also get searchable reproducibility tags including `git_commit`, `git_branch`, `git_dirty`, and `reproducibility_hash`.

Tracing:

- `setup_tracing()` configures tracing endpoints when relevant environment variables are present.
- `setup_langfuse()` provides optional Langfuse integration.

Related notes:

- [[Operations Runbook]]
- [[Benchmark Pipeline]]
- [[Evaluation and Metrics]]

