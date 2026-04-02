# MLflow Metrics & RAG Evaluation

## MLflow Built-in Metrics

MLflow provides tracking infrastructure out of the box:

- **Automatic metrics** — `mlflow.autolog()` with sklearn, PyTorch, etc. auto-logs loss, accuracy, and training metrics.
- **Custom metrics** — `mlflow.log_metric("name", value)`, `mlflow.log_metrics({"metric1": val1, ...})`.
- **System metrics** — CPU/GPU utilization, memory, network, disk via `mlflow.enable_system_metrics_logging()`.
- **Parameters** — `mlflow.log_param()` / `mlflow.log_params()`.
- **Artifacts** — Files, plots, models via `mlflow.log_artifact()`.

## MLflow Tracing for RAG/LLM

MLflow Tracing automatically captures RAG pipeline details:

- **Retrieval metrics** — retrieved documents, relevance scores
- **Generation metrics** — token counts, latency, input/output text
- **Spans** — each step (retrieval, reranking, generation) as a traceable span

## RAG Quality Metrics

For actual RAG quality evaluation (faithfulness, answer relevance, context precision/recall), MLflow relies on integrations:

- **Ragas** — Primary framework for RAG evaluation metrics (currently used in this project)
- **MLflow's `mlflow.evaluate()`** — Built-in LLM-judged metrics like `toxicity`, `factualness`, `relevance` via `mlflow.metrics`

## Current Approach

Ragas for metrics + MLflow for tracking = correct pattern.

Potential additions to explore:
- `mlflow.enable_system_metrics_logging()` for GPU/CPU tracking during evaluation runs
- `mlflow.evaluate()` with built-in metrics as a complement to Ragas scores
