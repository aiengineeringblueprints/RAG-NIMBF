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
- `log_benchmark_run()` logs aggregate metrics, config params, tags, per-sample CSV, reproducibility bundle, optional classic retriever metrics, and optional MLflow GenAI RAG judges. It auto-nests under an active parent MLflow run.
- `log_genai_eval()` remains as a backward-compatible model-evaluation style logger, but the main worker path now logs per-config MLflow evaluations from inside `log_benchmark_run()`.
- `log_aggregate_artifacts_to_mlflow()` creates a run-level MLflow summary containing CSV tables, JSON/Markdown reports, PNG/HTML plots, and reproducibility artifacts; it nests when called under the benchmark parent run.
- `log_plots_to_mlflow()` remains as a backward-compatible wrapper around aggregate artifact logging.

Reproducibility:

- Each deterministic run writes `results/runN/reproducibility/manifest.json` and `packages.txt`.
- The manifest captures UTC timestamp, Python/platform details, argv/cwd, git commit/branch/dirty status, selected benchmark environment variables with secrets redacted, and the full validated config grid.
- The same reproducibility directory is logged to every per-config MLflow run and the aggregate summary run under the `reproducibility` artifact path.
- MLflow runs also get searchable reproducibility tags including `git_commit`, `git_branch`, `git_dirty`, and `reproducibility_hash`.
- `MLFLOW_GENAI_JUDGES_ENABLED=true` enables `RetrievalRelevance`, `RetrievalGroundedness`, and `RetrievalSufficiency` via `mlflow.genai.evaluate()`. The implementation replays stored contexts through a `RETRIEVER` span, so it does not rerun retrieval/generation.

Tracing:

- `setup_tracing()` configures tracing endpoints when relevant environment variables are present.
- `setup_langfuse()` provides optional Langfuse integration.

Related notes:

- [[Operations Runbook]]
- [[Benchmark Pipeline]]
- [[Evaluation and Metrics]]

Resource utilization figures:

- `BENCHMARK_RESOURCE_MONITOR=true` enables per-config resource tracing under `results/runN/resource_traces/`.
- Each traced config writes `<config>.csv` with sampled CPU, host memory, GPU utilization/memory, PCIe, disk I/O, and GPU power counters, plus `<config>_markers.csv` with stage start/end markers.
- `python -m benchmark.reporting.resource_charts --trace-dir results/runN/resource_traces --output Paper/NGEN-AI/figures/fig_resource_breakdown` generates PDF and PNG multi-panel resource breakdown figures for the NGEN-AI paper.

## ClearML Tracking

ClearML is an optional orchestration and UI-editing layer. `benchmark.clearml_task` initializes a ClearML Task, connects safe benchmark parameters for Web UI edits, executes one config through `ExperimentWorker`, reports scalar benchmark metrics to ClearML, and keeps MLflow logging enabled by default. MLflow remains the detailed run/artifact backend; ClearML owns cloning, queueing, remote execution, and high-level scalar visibility.
