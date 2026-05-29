"""ClearML entrypoint for UI-editable benchmark tasks."""

from __future__ import annotations

import argparse
import math
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from rich.console import Console

from config import BenchmarkConfig, get_all_combinations
from benchmark.orchestration.worker import ExperimentWorker, WorkerOptions
from benchmark.providers import parse_model_id
from benchmark.tracking import setup_mlflow
from benchmark.tracing import setup_tracing
from benchmark.reporting.models import BenchmarkResultExtended


console = Console()

SECRET_CONFIG_FIELDS = {
    "ollama_api_key",
    "openai_compat_api_key",
    "llm_ollama_api_key",
    "llm_openai_compat_api_key",
    "eval_critic_ollama_api_key",
    "eval_critic_openai_compat_api_key",
    "embedding_ollama_api_key",
    "rag_http_headers",
    "rag_http_auth_header",
    "rag_http_auth_value",
}


def clearml_parameters_from_config(config: BenchmarkConfig) -> dict[str, Any]:
    """Return UI-editable config fields that are safe to publish to ClearML."""
    params: dict[str, Any] = {}
    for field in fields(BenchmarkConfig):
        if field.name in SECRET_CONFIG_FIELDS:
            continue
        params[field.name] = getattr(config, field.name)
    return params


def config_from_clearml_parameters(
    base_config: BenchmarkConfig,
    parameters: dict[str, Any],
) -> BenchmarkConfig:
    """Apply ClearML UI parameter values to a BenchmarkConfig."""
    allowed = {field.name for field in fields(BenchmarkConfig)}
    updates: dict[str, Any] = {}

    for key, raw_value in parameters.items():
        if key not in allowed or key in SECRET_CONFIG_FIELDS:
            continue
        updates[key] = _coerce_config_value(key, raw_value)

    if "llm_model" in updates and _has_provider_prefix(str(updates["llm_model"])):
        provider, model_name = parse_model_id(str(updates["llm_model"]))
        updates["llm_provider"] = provider
        updates["llm_model"] = model_name

    config = replace(base_config, **updates)
    if config.chunking_strategy == "semantic":
        config = replace(config, chunk_size=None, chunk_overlap=None)
    return config


def report_result_to_clearml(task: Any, result: BenchmarkResultExtended) -> None:
    """Log benchmark result metrics to the active ClearML task."""
    logger = task.get_logger()
    for name, value in _scalar_metrics(result).items():
        logger.report_scalar(
            title="RAG Benchmark",
            series=name,
            value=value,
            iteration=0,
        )


def run_clearml_task(
    *,
    manifest: Path | None,
    project_name: str,
    task_name: str,
    remote_queue: str | None,
    run_dir: Path | None,
    keep_going: bool,
    log_mlflow: bool,
) -> list[BenchmarkResultExtended]:
    """Initialize ClearML, apply UI overrides, and run one benchmark config."""
    try:
        from clearml import Task
    except ImportError as exc:
        raise RuntimeError(
            "ClearML support requires the optional 'clearml' package. "
            "Install project requirements before running this entrypoint."
        ) from exc

    base_configs = get_all_combinations(manifest) if manifest else get_all_combinations()
    base_config = base_configs[0]

    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        reuse_last_task_id=False,
    )
    connected_params = task.connect(clearml_parameters_from_config(base_config))
    config = config_from_clearml_parameters(base_config, connected_params)

    if remote_queue:
        task.execute_remotely(queue_name=remote_queue, exit_process=True)

    tracking_uri = setup_tracing()
    console.print(f"[dim]MLflow tracking: {tracking_uri}[/dim]")
    if log_mlflow:
        setup_mlflow()

    worker = ExperimentWorker(
        [config],
        WorkerOptions(
            run_dir=run_dir,
            resume=False,
            keep_going=keep_going,
            dry_run=False,
            experiment_name=task_name,
            write_reports=True,
            log_mlflow=log_mlflow,
        ),
    )
    results = worker.run()
    if results:
        report_result_to_clearml(task, results[0])
    task.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single RAG benchmark configuration as a ClearML task. "
            "Clone the task in the ClearML UI, edit hyperparameters, and "
            "enqueue it for a clearml-agent."
        ),
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        help="Optional JSON/YAML experiment manifest. The first expanded config is used.",
    )
    parser.add_argument("--project-name", default="RAG Benchmarking")
    parser.add_argument("--task-name", default="rag_benchmark_clearml")
    parser.add_argument(
        "--remote-queue",
        default=None,
        help=(
            "When set, enqueue this task to the named ClearML queue and exit "
            "locally. Usually omitted when cloning/enqueuing from the Web UI."
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    run_clearml_task(
        manifest=Path(args.manifest) if args.manifest else None,
        project_name=args.project_name,
        task_name=args.task_name,
        remote_queue=args.remote_queue,
        run_dir=args.run_dir,
        keep_going=args.keep_going,
        log_mlflow=not args.no_mlflow,
    )


def _coerce_config_value(key: str, value: Any) -> Any:
    if _is_empty_none(value):
        return None
    if key in {
        "chunk_size",
        "chunk_overlap",
        "retrieval_top_k",
        "retrieval_fetch_k",
        "max_new_tokens",
        "eval_critic_max_tokens",
        "dataset_sample_size",
        "reranker_top_k",
        "semantic_breakpoint_amount",
    }:
        return int(value)
    if key in {"rag_http_timeout_seconds", "retrieval_mmr_lambda"}:
        return float(value)
    if key in {"retrieval_use_hyde", "llm_answer_value_fallback"}:
        return _to_bool(value)
    return str(value) if value is not None else None


def _is_empty_none(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "none", "null"}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _has_provider_prefix(value: str) -> bool:
    prefix, separator, _ = value.partition(":")
    return bool(separator and prefix in {"ollama", "openai", "huggingface"})


def _scalar_metrics(result: BenchmarkResultExtended) -> dict[str, float]:
    metrics: dict[str, float | int | None] = {
        "num_chunks": result.num_chunks,
        "num_questions": result.num_questions,
        "avg_ttft_seconds": result.avg_ttft_seconds,
        "avg_tokens_per_second": result.avg_tokens_per_second,
        "avg_gpu_utilization_pct": result.avg_gpu_utilization_pct,
        "avg_gpu_memory_used_mb": result.avg_gpu_memory_used_mb,
        "ragas_faithfulness": result.ragas_faithfulness,
        "ragas_answer_relevancy": result.ragas_answer_relevancy,
        "ragas_answer_correctness": result.ragas_answer_correctness,
        "ragas_context_precision": result.ragas_context_precision,
        "ragas_context_recall": result.ragas_context_recall,
        "ragas_semantic_similarity": result.ragas_semantic_similarity,
        "total_time_seconds": result.total_time_seconds,
    }
    if result.custom_metric_means:
        metrics.update(result.custom_metric_means)
    if result.stage_timings:
        metrics.update({f"stage_{k}_seconds": v for k, v in result.stage_timings.items()})

    clean: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            clean[key] = float(value)
    return clean


if __name__ == "__main__":
    main()
