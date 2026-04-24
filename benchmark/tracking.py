"""MLflow experiment tracking for RAG benchmark runs."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import RunStatus

from benchmark.reporting.models import BenchmarkResultExtended

logger = logging.getLogger(__name__)


def setup_mlflow() -> str:
    """Configure MLflow experiment settings after tracing is initialized.

    Sets the experiment name to RAG-Benchmark. Called after setup_tracing()
    which handles the tracking URI.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # Try server connection, fall back to SQLite
    try:
        import requests
        requests.get(f"{tracking_uri}/api/2.0/mlflow/experiments/search", timeout=2)
    except Exception:
        tracking_uri = "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment("RAG-Benchmark")
    logger.info("MLflow experiment: RAG-Benchmark (URI: %s)", tracking_uri)
    return tracking_uri


def _flatten_ragas_stats(
    result: BenchmarkResultExtended,
) -> dict[str, float | None]:
    """Return flat ``{metric_stat: value}`` dict for RAGAS stat summaries."""
    flat: dict[str, float | None] = {}
    for key, stats in [
        ("ragas_faithfulness", result.ragas_faithfulness_stats),
        ("ragas_answer_relevancy", result.ragas_answer_relevancy_stats),
        ("ragas_answer_correctness", result.ragas_answer_correctness_stats),
        ("ragas_context_precision", result.ragas_context_precision_stats),
        ("ragas_context_recall", result.ragas_context_recall_stats),
    ]:
        if stats is None:
            flat[f"{key}_mean"] = None
            flat[f"{key}_std"] = None
            flat[f"{key}_median"] = None
            flat[f"{key}_min"] = None
            flat[f"{key}_max"] = None
        else:
            flat[f"{key}_mean"] = stats.mean
            flat[f"{key}_std"] = stats.std
            flat[f"{key}_median"] = stats.median
            flat[f"{key}_min"] = stats.min
            flat[f"{key}_max"] = stats.max
    return flat


def _make_run_name(result: BenchmarkResultExtended) -> str:
    """Generate a structured run name from config parameters."""
    llm = result.llm_model.split("/")[-1].replace(":", "_")
    retrieval = result.retrieval_strategy or "unknown"
    template = result.prompt_template
    return f"{llm}_{result.chunking_strategy}_cs{result.chunk_size}_co{result.chunk_overlap}_{retrieval}_{template}"


def _make_tags(result: BenchmarkResultExtended) -> dict[str, str]:
    """Build MLflow tags from benchmark result."""
    tags: dict[str, str] = {
        "llm_model": result.llm_model,
        "embedding_model": result.embedding_model,
        "chunking_strategy": result.chunking_strategy,
        "prompt_template": result.prompt_template,
    }
    if result.reranker_model:
        tags["reranker_model"] = result.reranker_model
    if result.retrieval_strategy:
        tags["retrieval_strategy"] = result.retrieval_strategy
    if result.retrieval_top_k is not None:
        tags["retrieval_top_k"] = str(result.retrieval_top_k)
    if result.dataset_name:
        tags["dataset_name"] = result.dataset_name
    return tags


def log_benchmark_run(result: BenchmarkResultExtended) -> None:
    """Log a single benchmark configuration as one MLflow run.

    Parameters
    ----------
    result:
        The fully aggregated benchmark result for one config.
    """
    experiment_name = "RAG-Benchmark"
    mlflow.set_experiment(experiment_name)

    tags = _make_tags(result)

    params: dict[str, Any] = {
        "chunk_size": result.chunk_size,
        "chunk_overlap": result.chunk_overlap,
        "num_chunks": result.num_chunks,
        "num_questions": result.num_questions,
    }
    if result.reranker_top_k is not None:
        params["reranker_top_k"] = result.reranker_top_k
    if result.retrieval_strategy:
        params["retrieval_strategy"] = result.retrieval_strategy
    if result.retrieval_top_k is not None:
        params["retrieval_top_k"] = result.retrieval_top_k
    if result.dataset_name:
        params["dataset_name"] = result.dataset_name
    if result.dataset_sample_size is not None:
        params["dataset_sample_size"] = result.dataset_sample_size

    metrics: dict[str, float] = {
        "avg_ttft_seconds": result.avg_ttft_seconds,
        "avg_tokens_per_second": result.avg_tokens_per_second,
        "total_time_seconds": result.total_time_seconds,
    }

    # Optional GPU metrics
    if result.avg_gpu_utilization_pct is not None:
        metrics["avg_gpu_utilization_pct"] = result.avg_gpu_utilization_pct
    if result.avg_gpu_memory_used_mb is not None:
        metrics["avg_gpu_memory_used_mb"] = result.avg_gpu_memory_used_mb

    # RAGAS mean metrics
    for key, value in [
        ("ragas_faithfulness", result.ragas_faithfulness),
        ("ragas_answer_relevancy", result.ragas_answer_relevancy),
        ("ragas_answer_correctness", result.ragas_answer_correctness),
        ("ragas_context_precision", result.ragas_context_precision),
        ("ragas_context_recall", result.ragas_context_recall),
    ]:
        if value is not None:
            metrics[key] = value

    # RAGAS detailed stats (mean, std, min, max, median)
    for stat_name, stat_value in _flatten_ragas_stats(result).items():
        if stat_value is not None:
            metrics[stat_name] = stat_value

    def _mlflow_safe(key: str) -> str:
        """Sanitize metric key for MLflow (no ``@`` allowed)."""
        return key.replace("@", "_at_")

    # Custom metrics (IR + NLG) means
    if result.custom_metric_means:
        for key, value in result.custom_metric_means.items():
            metrics[f"custom_{_mlflow_safe(key)}"] = value

    # Custom metrics stats
    if result.custom_stats:
        for key, stats in result.custom_stats.items():
            if stats is not None:
                safe = _mlflow_safe(key)
                metrics[f"custom_{safe}_mean"] = stats.mean
                metrics[f"custom_{safe}_std"] = stats.std
                metrics[f"custom_{safe}_median"] = stats.median
                metrics[f"custom_{safe}_min"] = stats.min
                metrics[f"custom_{safe}_max"] = stats.max

    # Latency / throughput stats
    for prefix, stats in [
        ("ttft", result.ttft_stats),
        ("tps", result.tps_stats),
        ("gpu_util", result.gpu_util_stats),
        ("gpu_mem", result.gpu_mem_stats),
    ]:
        if stats is not None:
            metrics[f"{prefix}_mean"] = stats.mean
            metrics[f"{prefix}_std"] = stats.std
            metrics[f"{prefix}_median"] = stats.median
            metrics[f"{prefix}_min"] = stats.min
            metrics[f"{prefix}_max"] = stats.max

    run_name = _make_run_name(result)
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Log per-sample results as a CSV artifact
        if result.per_sample:
            _log_per_sample_csv(result, run.info.run_id)

        if result.evaluation_error:
            mlflow.set_tag("evaluation_error", result.evaluation_error)

        logger.info(
            "Logged MLflow run %s for config '%s'", run.info.run_id, result.config_name
        )


def _log_per_sample_csv(result: BenchmarkResultExtended, run_id: str) -> None:
    """Write per-sample data to a CSV and log it as an artifact."""
    import csv
    import tempfile
    from pathlib import Path

    ragas_keys = [
        "faithfulness",
        # "answer_relevancy",
        # "answer_correctness",
        # "context_precision",
        # "context_recall",
    ]

    # Collect all custom metric keys across samples
    custom_keys: list[str] = []
    seen_custom: set[str] = set()
    for sample in result.per_sample:
        if sample.custom_scores:
            for k in sample.custom_scores:
                if k not in seen_custom:
                    custom_keys.append(k)
                    seen_custom.add(k)

    tmpdir = Path(tempfile.mkdtemp())
    csv_path = tmpdir / "per_sample_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "question",
            "ground_truth",
            "answer",
            "ttft_seconds",
            "total_seconds",
            "token_count",
            "tokens_per_second",
        ] + [f"ragas_{k}" for k in ragas_keys] + [f"custom_{k}" for k in custom_keys]
        writer.writerow(header)

        for sample in result.per_sample:
            row = [
                sample.question,
                sample.ground_truth,
                sample.answer,
                sample.ttft_seconds,
                sample.total_seconds,
                sample.token_count,
                sample.tokens_per_second,
            ] + [sample.ragas_scores.get(k, "") for k in ragas_keys] + [
                (sample.custom_scores or {}).get(k, "") for k in custom_keys
            ]
            writer.writerow(row)

    mlflow.log_artifact(str(csv_path), artifact_path="data")
    logger.debug("Logged per-sample CSV artifact to run %s", run_id)


def log_genai_eval(result: BenchmarkResultExtended) -> None:
    """Log a GenAI evaluation dataset for the MLflow eval-monitor dashboard.

    Creates an eval DataFrame from per-sample results and logs it via
    mlflow.evaluate() with built-in scorers. This is additive — it does
    not replace the RAGAS or custom metrics computation.
    """
    if not result.per_sample:
        logger.warning("No per-sample data — skipping GenAI eval logging")
        return

    try:
        import pandas as pd

        eval_data = pd.DataFrame({
            "inputs": [s.question for s in result.per_sample],
            "predictions": [s.answer for s in result.per_sample],
            "contexts": [list(s.contexts) for s in result.per_sample],
            "ground_truth": [s.ground_truth for s in result.per_sample],
        })

        with mlflow.start_run(run_name=_make_run_name(result) + "_eval", tags=_make_tags(result)):
            eval_result = mlflow.evaluate(
                data=eval_data,
                targets="ground_truth",
                predictions="predictions",
                evaluators="default",
            )
            logger.info(
                "GenAI eval logged: %s metrics", len(eval_result.metrics)
            )
    except Exception as e:
        logger.warning("GenAI eval logging failed (non-fatal): %s", e)


def log_plots_to_mlflow(results_dir: Path, run_name_prefix: str = "plots") -> None:
    """Log all generated plot HTML files from a results directory to MLflow.

    Scans the results_plots/ subdirectory for .html files and logs each
    as an MLflow artifact under the 'plots' artifact path.
    """
    plots_dir = results_dir / "results_plots"
    if not plots_dir.exists():
        logger.warning("No results_plots/ directory found at %s", results_dir)
        return

    html_files = list(plots_dir.glob("*.html"))
    if not html_files:
        logger.warning("No HTML plot files found in %s", plots_dir)
        return

    try:
        with mlflow.start_run(run_name=run_name_prefix, tags={"type": "plots"}) as run:
            for html_file in sorted(html_files):
                mlflow.log_artifact(str(html_file), artifact_path="plots")
            logger.info("Logged %d plot artifacts to MLflow run %s", len(html_files), run.info.run_id)
    except Exception as e:
        logger.warning("Failed to log plots to MLflow (non-fatal): %s", e)
