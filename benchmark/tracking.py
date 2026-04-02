"""MLflow experiment tracking for RAG benchmark runs."""
from __future__ import annotations

import logging
import os
from typing import Any

import mlflow
from mlflow.entities import RunStatus

from benchmark.reporting.models import BenchmarkResultExtended

logger = logging.getLogger(__name__)


def setup_mlflow() -> str:
    """Configure MLflow tracking URI and return it.

    Reads MLFLOW_TRACKING_URI from the environment (default: ``mlruns``
    inside the project root, i.e. a local file store).
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)
    return tracking_uri


def _flatten_ragas_stats(
    result: BenchmarkResultExtended,
) -> dict[str, float | None]:
    """Return flat ``{metric_stat: value}`` dict for RAGAS stat summaries."""
    flat: dict[str, float | None] = {}
    for key, stats in [
        ("ragas_faithfulness", result.ragas_faithfulness_stats),
        ("ragas_answer_relevancy", result.ragas_answer_relevancy_stats),
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


def log_benchmark_run(result: BenchmarkResultExtended) -> None:
    """Log a single benchmark configuration as one MLflow run.

    Parameters
    ----------
    result:
        The fully aggregated benchmark result for one config.
    """
    experiment_name = "RAG-Benchmark"
    mlflow.set_experiment(experiment_name)

    tags: dict[str, str] = {
        "llm_model": result.llm_model,
        "embedding_model": result.embedding_model,
        "chunking_strategy": result.chunking_strategy,
    }
    if result.reranker_model:
        tags["reranker_model"] = result.reranker_model

    params: dict[str, Any] = {
        "chunk_size": result.chunk_size,
        "chunk_overlap": result.chunk_overlap,
        "num_chunks": result.num_chunks,
        "num_questions": result.num_questions,
    }
    if result.reranker_top_k is not None:
        params["reranker_top_k"] = result.reranker_top_k

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
        ("ragas_context_precision", result.ragas_context_precision),
        ("ragas_context_recall", result.ragas_context_recall),
    ]:
        if value is not None:
            metrics[key] = value

    # RAGAS detailed stats (mean, std, min, max, median)
    for stat_name, stat_value in _flatten_ragas_stats(result).items():
        if stat_value is not None:
            metrics[stat_name] = stat_value

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

    with mlflow.start_run(run_name=result.config_name, tags=tags) as run:
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
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

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
        ] + [f"ragas_{k}" for k in ragas_keys]
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
            ] + [sample.ragas_scores.get(k, "") for k in ragas_keys]
            writer.writerow(row)

    mlflow.log_artifact(str(csv_path), artifact_path="data")
    logger.debug("Logged per-sample CSV artifact to run %s", run_id)
