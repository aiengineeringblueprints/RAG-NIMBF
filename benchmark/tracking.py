"""MLflow experiment tracking for RAG benchmark runs."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import mlflow

from benchmark.reporting.models import BenchmarkResultExtended

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


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

    if os.getenv("MLFLOW_ENABLE_SYSTEM_METRICS", "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        try:
            mlflow.enable_system_metrics_logging()
            logger.info("MLflow system metrics logging enabled")
        except Exception as exc:
            logger.warning("Could not enable MLflow system metrics logging: %s", exc)

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
        ("ragas_semantic_similarity", result.ragas_semantic_similarity_stats),
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
    if result.chunking_strategy == "semantic":
        chunk_label = "semantic"
    else:
        chunk_label = f"cs{result.chunk_size}_co{result.chunk_overlap}"
    return f"{llm}_{result.chunking_strategy}_{chunk_label}_{retrieval}_{template}"


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
    if result.vector_db_backend:
        tags["vector_db_backend"] = result.vector_db_backend
    return tags


def _reproducibility_tags(reproducibility_dir: Path | None) -> dict[str, str]:
    """Load searchable MLflow tags from a reproducibility manifest."""
    if reproducibility_dir is None:
        return {}
    manifest_path = reproducibility_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        logger.warning("Could not read reproducibility manifest: %s", exc)
        return {}

    git = manifest.get("git") or {}
    python_info = manifest.get("python") or {}
    tags: dict[str, str] = {
        "reproducibility_hash": hashlib.sha256(raw).hexdigest()[:16],
    }
    if git.get("commit"):
        tags["git_commit"] = str(git["commit"])
    if git.get("branch"):
        tags["git_branch"] = str(git["branch"])
    tags["git_dirty"] = str(bool(git.get("dirty"))).lower()
    if python_info.get("platform"):
        tags["python_platform"] = str(python_info["platform"])[:250]
    return tags


def log_benchmark_run(
    result: BenchmarkResultExtended,
    reproducibility_dir: Path | None = None,
    *,
    nested: bool | None = None,
) -> str | None:
    """Log a single benchmark configuration as one MLflow run.

    Parameters
    ----------
    result:
        The fully aggregated benchmark result for one config.
    reproducibility_dir:
        Optional directory containing manifest/package artifacts for reruns.
    nested:
        Whether to create this as a nested child run. ``None`` auto-nests
        when a parent MLflow run is already active.
    """
    experiment_name = "RAG-Benchmark"
    mlflow.set_experiment(experiment_name)

    tags = _make_tags(result)
    tags.update(_reproducibility_tags(reproducibility_dir))

    params: dict[str, Any] = {
        "chunk_size": result.chunk_size if result.chunk_size is not None else "n/a",
        "chunk_overlap": result.chunk_overlap if result.chunk_overlap is not None else "n/a",
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
    if result.vector_db_backend:
        params["vector_db_backend"] = result.vector_db_backend

    metrics: dict[str, float] = {
        "avg_ttft_seconds": result.avg_ttft_seconds,
        "avg_tokens_per_second": result.avg_tokens_per_second,
        "total_time_seconds": result.total_time_seconds,
        "total_input_tokens": result.total_input_tokens,
        "total_output_tokens": result.total_output_tokens,
        "total_tokens": result.total_tokens,
    }

    # Optional GPU metrics
    if result.avg_gpu_utilization_pct is not None:
        metrics["avg_gpu_utilization_pct"] = result.avg_gpu_utilization_pct
    if result.avg_gpu_memory_used_mb is not None:
        metrics["avg_gpu_memory_used_mb"] = result.avg_gpu_memory_used_mb
    if result.total_estimated_cost_usd is not None:
        metrics["total_estimated_cost_usd"] = result.total_estimated_cost_usd
    if result.avg_estimated_cost_per_answer_usd is not None:
        metrics["avg_estimated_cost_per_answer_usd"] = result.avg_estimated_cost_per_answer_usd

    if result.stage_timings:
        for key, value in result.stage_timings.items():
            metrics[f"stage_{key}_seconds"] = value

    # RAGAS mean metrics
    for key, value in [
        ("ragas_faithfulness", result.ragas_faithfulness),
        ("ragas_answer_relevancy", result.ragas_answer_relevancy),
        ("ragas_answer_correctness", result.ragas_answer_correctness),
        ("ragas_context_precision", result.ragas_context_precision),
        ("ragas_context_recall", result.ragas_context_recall),
        ("ragas_semantic_similarity", result.ragas_semantic_similarity),
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
    nested_run = bool(mlflow.active_run()) if nested is None else nested
    with mlflow.start_run(run_name=run_name, tags=tags, nested=nested_run) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Log per-sample results as a CSV artifact
        if result.per_sample:
            _log_per_sample_csv(result, run.info.run_id)

        if reproducibility_dir is not None and reproducibility_dir.exists():
            mlflow.log_artifacts(str(reproducibility_dir), artifact_path="reproducibility")

        if result.evaluation_error:
            mlflow.set_tag("evaluation_error", result.evaluation_error)

        _log_classic_retriever_metrics(result)
        _log_genai_rag_judges(result)

        logger.info(
            "Logged MLflow run %s for config '%s'", run.info.run_id, result.config_name
        )
        return run.info.run_id


def _log_per_sample_csv(result: BenchmarkResultExtended, run_id: str) -> None:
    """Write per-sample data to a CSV and log it as an artifact."""
    import csv
    import tempfile
    from pathlib import Path

    ragas_keys = [
        "faithfulness",
        "semantic_similarity",
        "context_recall",
        # "answer_relevancy",
        # "answer_correctness",
        # "context_precision",
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
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "estimated_cost_usd",
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
                sample.input_tokens,
                sample.output_tokens,
                sample.total_tokens,
                sample.estimated_cost_usd if sample.estimated_cost_usd is not None else "",
            ] + [sample.ragas_scores.get(k, "") for k in ragas_keys] + [
                (sample.custom_scores or {}).get(k, "") for k in custom_keys
            ]
            writer.writerow(row)

    mlflow.log_artifact(str(csv_path), artifact_path="data")
    logger.debug("Logged per-sample CSV artifact to run %s", run_id)


def _log_classic_retriever_metrics(result: BenchmarkResultExtended) -> None:
    """Log MLflow retrieval metrics when gold and retrieved doc IDs exist."""
    if not _env_flag("MLFLOW_CLASSIC_RETRIEVER_METRICS_ENABLED", True):
        mlflow.set_tag("mlflow_classic_retriever_metrics", "disabled")
        return
    rows = [
        {
            "query": sample.question,
            "retrieved_doc_ids": list(sample.retrieved_doc_ids),
            "ground_truth_doc_ids": list(sample.ground_truth_doc_ids),
        }
        for sample in result.per_sample
        if sample.retrieved_doc_ids and sample.ground_truth_doc_ids
    ]
    if not rows:
        mlflow.set_tag("mlflow_classic_retriever_metrics", "skipped_no_doc_ids")
        return

    try:
        import pandas as pd
        from mlflow.metrics import ndcg_at_k, precision_at_k, recall_at_k

        k = result.retrieval_top_k or max(len(row["retrieved_doc_ids"]) for row in rows)
        eval_result = mlflow.evaluate(
            data=pd.DataFrame(rows),
            predictions="retrieved_doc_ids",
            targets="ground_truth_doc_ids",
            extra_metrics=[
                precision_at_k(k),
                recall_at_k(k),
                ndcg_at_k(k),
            ],
        )
        mlflow.set_tag("mlflow_classic_retriever_metrics", "logged")
        logger.info(
            "Logged classic retriever metrics: %s metrics",
            len(eval_result.metrics),
        )
    except Exception as exc:
        mlflow.set_tag("mlflow_classic_retriever_metrics", "failed")
        logger.warning("Classic MLflow retriever metrics failed (non-fatal): %s", exc)


def _log_genai_rag_judges(result: BenchmarkResultExtended) -> None:
    """Replay per-sample RAG outputs through MLflow RAG judges.

    The benchmark has already generated answers. This function avoids another
    expensive RAG pass by replaying the stored answer and contexts while still
    creating a RETRIEVER span that MLflow's RAG judges can inspect.
    """
    if not _env_flag("MLFLOW_GENAI_JUDGES_ENABLED", False):
        mlflow.set_tag("mlflow_genai_judges", "disabled")
        return
    if not result.per_sample:
        mlflow.set_tag("mlflow_genai_judges", "skipped_no_samples")
        return

    judge_model = os.getenv("MLFLOW_GENAI_JUDGE_MODEL", "openai:/gpt-4o-mini")
    samples_by_query = {sample.question: sample for sample in result.per_sample}
    eval_dataset = [
        {
            "inputs": {"query": sample.question},
            "expectations": {"expected_facts": [sample.ground_truth]},
        }
        for sample in result.per_sample
        if sample.ground_truth
    ]
    if not eval_dataset:
        mlflow.set_tag("mlflow_genai_judges", "skipped_no_expectations")
        return

    try:
        from mlflow.entities import Document
        from mlflow.genai.scorers import (
            RetrievalGroundedness,
            RetrievalRelevance,
            RetrievalSufficiency,
        )

        def coerce_query(query: Any = None, **kwargs: Any) -> str:
            if isinstance(query, dict):
                query = query.get("query")
            if query is None:
                query = kwargs.get("query")
            return str(query)

        @mlflow.trace(span_type="RETRIEVER")
        def replay_retrieve_docs(query: Any = None, **kwargs: Any) -> list[Document]:
            actual_query = coerce_query(query, **kwargs)
            sample = samples_by_query[actual_query]
            doc_ids = sample.retrieved_doc_ids
            docs: list[Document] = []
            for index, context in enumerate(sample.contexts):
                doc_id = doc_ids[index] if index < len(doc_ids) else f"ctx_{index}"
                docs.append(
                    Document(
                        id=str(doc_id),
                        page_content=context,
                        metadata={"rank": index + 1},
                    )
                )
            return docs

        @mlflow.trace
        def replay_rag_app(query: Any = None, **kwargs: Any) -> dict[str, str]:
            actual_query = coerce_query(query, **kwargs)
            replay_retrieve_docs(actual_query)
            return {"response": samples_by_query[actual_query].answer}

        eval_result = mlflow.genai.evaluate(
            data=eval_dataset,
            predict_fn=replay_rag_app,
            scorers=[
                RetrievalRelevance(model=judge_model),
                RetrievalGroundedness(model=judge_model),
                RetrievalSufficiency(model=judge_model),
            ],
        )
        mlflow.set_tag("mlflow_genai_judges", "logged")
        mlflow.set_tag("mlflow_genai_judge_model", judge_model)
        logger.info("Logged MLflow GenAI RAG judges: %s", type(eval_result).__name__)
    except Exception as exc:
        mlflow.set_tag("mlflow_genai_judges", "failed")
        logger.warning("MLflow GenAI RAG judges failed (non-fatal): %s", exc)


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

        with mlflow.start_run(
            run_name=_make_run_name(result) + "_eval",
            tags=_make_tags(result),
            nested=bool(mlflow.active_run()),
        ):
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
    """Backward-compatible wrapper for aggregate artifact logging."""
    log_aggregate_artifacts_to_mlflow(results_dir, run_name=run_name_prefix)


def log_aggregate_artifacts_to_mlflow(
    results_dir: Path,
    *,
    run_name: str = "run_summary",
    reproducibility_dir: Path | None = None,
) -> None:
    """Log run-level tables, reports, plots, and reproducibility artifacts.

    This creates one MLflow run that is easy to open when comparing a full
    benchmark sweep: summary tables live under ``tables/``, plots under
    ``plots/``, reports under ``reports/``, and rerun metadata under
    ``reproducibility/``.
    """
    if not results_dir.exists():
        logger.warning("Results directory does not exist: %s", results_dir)
        return

    artifact_groups: list[tuple[Path, str]] = []
    for filename in ("results_summary.csv", "results_per_sample.csv"):
        path = results_dir / filename
        if path.exists():
            artifact_groups.append((path, "tables"))

    for path in sorted(results_dir.glob("benchmark_*.json")):
        artifact_groups.append((path, "reports"))
    for filename in ("report.md",):
        path = results_dir / filename
        if path.exists():
            artifact_groups.append((path, "reports"))

    plots_dir = results_dir / "results_plots"
    if plots_dir.exists():
        for pattern in ("*.html", "*.png", "*.jpg", "*.jpeg", "*.svg"):
            for path in sorted(plots_dir.glob(pattern)):
                artifact_groups.append((path, "plots"))

    repro_dir = reproducibility_dir or (results_dir / "reproducibility")

    try:
        tags = {"type": "aggregate", "results_dir": str(results_dir)}
        tags.update(_reproducibility_tags(repro_dir))
        with mlflow.start_run(
            run_name=run_name,
            tags=tags,
            nested=bool(mlflow.active_run()),
        ) as run:
            for path, artifact_path in artifact_groups:
                mlflow.log_artifact(str(path), artifact_path=artifact_path)
            if repro_dir.exists():
                mlflow.log_artifacts(str(repro_dir), artifact_path="reproducibility")
            mlflow.log_metric("num_logged_artifacts", float(len(artifact_groups)))
            logger.info(
                "Logged %d aggregate artifacts to MLflow run %s",
                len(artifact_groups),
                run.info.run_id,
            )
    except Exception as e:
        logger.warning("Failed to log aggregate artifacts to MLflow (non-fatal): %s", e)
