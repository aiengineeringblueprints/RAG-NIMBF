"""Per-stage wall-clock latency breakdown for RAG benchmark runs.

Inspired by RAGPerf (arXiv:2603.10765v1) §3.4. Records one latency sample
per call so per-question stages (retrieval, reranking, generation) can be
summarised with mean / p50 / p95, while corpus-level stages (chunking,
indexing, evaluation) accumulate a single sample per run.

Uses only ``time.perf_counter`` (monotonic, high-resolution) and the
standard library. Sub-microsecond overhead per call.
"""
from __future__ import annotations

import json
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator


CANONICAL_STAGES: tuple[str, ...] = (
    "chunking",
    "embedding",
    "indexing",
    "retrieval",
    "reranking",
    "generation",
    "evaluation",
    "total",
)

# Percentile helper that does not depend on numpy.
def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


@dataclass
class StageStats:
    """Aggregated summary for a single stage."""

    count: int
    total_s: float
    mean_s: float
    p50_s: float
    p95_s: float
    max_s: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "total_s": self.total_s,
            "mean_s": self.mean_s,
            "p50_s": self.p50_s,
            "p95_s": self.p95_s,
            "max_s": self.max_s,
        }


@dataclass
class StageTimings:
    """Accumulates per-call latency samples for each canonical stage."""

    samples: dict[str, list[float]] = field(default_factory=dict)
    # Optional counts (chunks, docs, questions) used to derive throughput.
    counts: dict[str, int] = field(default_factory=dict)

    def record(self, stage: str, seconds: float) -> None:
        if seconds < 0:
            return
        bucket = self.samples.get(stage)
        if bucket is None:
            bucket = []
            self.samples[stage] = bucket
        bucket.append(seconds)

    def set_count(self, key: str, value: int) -> None:
        if value < 0:
            return
        self.counts[key] = value

    def stages(self) -> list[str]:
        return [s for s in CANONICAL_STAGES if s in self.samples]

    def stats(self, stage: str) -> StageStats | None:
        bucket = self.samples.get(stage)
        if not bucket:
            return None
        return StageStats(
            count=len(bucket),
            total_s=sum(bucket),
            mean_s=statistics.fmean(bucket),
            p50_s=_percentile(bucket, 50.0),
            p95_s=_percentile(bucket, 95.0),
            max_s=max(bucket),
        )

    def summary(self) -> dict[str, dict[str, float | int]]:
        out: dict[str, dict[str, float | int]] = {}
        for stage in self.stages():
            stats = self.stats(stage)
            if stats is not None:
                out[stage] = stats.to_dict()
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": self.summary(),
            "counts": dict(self.counts),
            "throughput": self.throughput(),
        }

    def throughput(self) -> dict[str, float]:
        """Derive throughput metrics from recorded samples and counts."""
        out: dict[str, float] = {}
        indexing = self.stats("indexing")
        num_chunks = self.counts.get("chunks")
        if indexing and num_chunks:
            out["embed_index_chunks_s"] = num_chunks / indexing.total_s if indexing.total_s > 0 else 0.0
        retrieval = self.stats("retrieval")
        if retrieval and retrieval.total_s > 0:
            out["retrieve_qps"] = retrieval.count / retrieval.total_s
        generation = self.stats("generation")
        if generation and generation.total_s > 0:
            out["generate_qps"] = generation.count / generation.total_s
        return out

    def mlflow_metrics(self) -> dict[str, float]:
        """Flat ``lat_<stage>_*`` metrics for MLflow logging."""
        metrics: dict[str, float] = {}
        for stage, stats in self.summary().items():
            metrics[f"lat_{stage}_total_s"] = stats["total_s"]
            metrics[f"lat_{stage}_mean_s"] = stats["mean_s"]
            metrics[f"lat_{stage}_p50_s"] = stats["p50_s"]
            metrics[f"lat_{stage}_p95_s"] = stats["p95_s"]
            metrics[f"lat_{stage}_max_s"] = stats["max_s"]
            metrics[f"lat_{stage}_count"] = float(stats["count"])
        for key, value in self.throughput().items():
            metrics[f"throughput_{key}"] = value
        return metrics

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


@contextmanager
def stage_timer(
    recorder: StageTimings | None,
    stage: str,
) -> Iterator[None]:
    """Time a block and push one sample into ``recorder``."""
    if recorder is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        recorder.record(stage, time.perf_counter() - started)


def timed(
    recorder: StageTimings | None,
    stage: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that records one sample per call of the wrapped function."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                if recorder is not None:
                    recorder.record(stage, time.perf_counter() - started)

        wrapper.__wrapped__ = func  # type: ignore[attr-defined]
        return wrapper

    return decorator
