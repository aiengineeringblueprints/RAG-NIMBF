from __future__ import annotations

import math
import platform
import statistics
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PerSampleResult:
    question: str
    ground_truth: str
    answer: str
    contexts: tuple[str, ...]
    ttft_seconds: float
    total_seconds: float
    token_count: int
    tokens_per_second: float
    gpu_usage: dict[str, float] | None
    ragas_scores: dict[str, float | None]
    custom_scores: dict[str, float | None] = None
    answer_valid: bool = True


@dataclass(frozen=True)
class StatSummary:
    mean: float
    std: float
    min: float
    max: float
    median: float
    count: int


@dataclass(frozen=True)
class BenchmarkResultExtended:
    config_name: str
    llm_model: str
    embedding_model: str
    prompt_template: str
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    num_chunks: int
    num_questions: int
    avg_ttft_seconds: float
    avg_tokens_per_second: float
    avg_gpu_utilization_pct: float | None
    avg_gpu_memory_used_mb: float | None
    ragas_faithfulness: float | None
    ragas_answer_relevancy: float | None
    ragas_answer_correctness: float | None
    ragas_context_precision: float | None
    ragas_context_recall: float | None
    ragas_semantic_similarity: float | None
    total_time_seconds: float
    per_sample: tuple[PerSampleResult, ...]
    ttft_stats: StatSummary | None
    tps_stats: StatSummary | None
    gpu_util_stats: StatSummary | None
    gpu_mem_stats: StatSummary | None
    ragas_faithfulness_stats: StatSummary | None
    ragas_answer_relevancy_stats: StatSummary | None
    ragas_answer_correctness_stats: StatSummary | None
    ragas_context_precision_stats: StatSummary | None
    ragas_context_recall_stats: StatSummary | None
    ragas_semantic_similarity_stats: StatSummary | None
    evaluation_error: str | None = None
    ragas_valid_sample_counts: dict[str, int] | None = None
    custom_metric_means: dict[str, float] | None = None
    custom_stats: dict[str, StatSummary | None] | None = None
    reranker_model: str | None = None
    reranker_top_k: int | None = None
    retrieval_strategy: str | None = None
    retrieval_top_k: int | None = None
    dataset_name: str | None = None
    dataset_sample_size: int | None = None
    stage_timings: dict[str, float] | None = None
    vector_db_backend: str | None = None


@dataclass(frozen=True)
class BenchmarkRun:
    timestamp: str
    dataset_name: str
    dataset_subset: str
    dataset_sample_size: int
    system_info: dict[str, Any]
    results: tuple[BenchmarkResultExtended, ...]


def compute_stats(values: list[float]) -> StatSummary | None:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return None
    return StatSummary(
        mean=statistics.mean(clean),
        std=statistics.stdev(clean) if len(clean) > 1 else 0.0,
        min=min(clean),
        max=max(clean),
        median=statistics.median(clean),
        count=len(clean),
    )


def collect_system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "os": platform.system(),
        "os_version": platform.release(),
        "platform": platform.platform(),
    }
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            info["gpu"] = parts[0]
            info["gpu_memory_total_mb"] = parts[1] if len(parts) > 1 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return info
