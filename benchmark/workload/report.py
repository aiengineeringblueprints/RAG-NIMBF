"""Aggregate a ``WorkloadSummary`` into JSON / MLflow-friendly metrics."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from benchmark.workload.runner import WorkloadSummary


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    # Linear interpolation between closest ranks.
    k = (len(sorted_vals) - 1) * pct
    floor = math.floor(k)
    ceil = math.ceil(k)
    if floor == ceil:
        return sorted_vals[int(k)]
    return sorted_vals[floor] + (k - floor) * (
        sorted_vals[ceil] - sorted_vals[floor]
    )


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def summarize(summary: WorkloadSummary) -> dict[str, Any]:
    """Build the canonical summary dict (op counts, percentiles, throughput)."""
    per_kind_latency: dict[str, dict[str, float]] = {}
    for kind, latencies in summary.latencies_by_kind.items():
        per_kind_latency[kind] = _stats(latencies)

    queue_depth_samples = summary.queue_depth_samples or [0]
    error_rate = (
        summary.total_errors / summary.total_submitted
        if summary.total_submitted
        else 0.0
    )
    stale_rate = (
        summary.total_stale_check / summary.total_submitted
        if summary.total_submitted
        else 0.0
    )

    return {
        "total_submitted": summary.total_submitted,
        "total_completed": summary.total_completed,
        "total_errors": summary.total_errors,
        "total_stale_check": summary.total_stale_check,
        "error_rate": error_rate,
        "stale_check_rate": stale_rate,
        "counts_by_kind": dict(summary.counts_by_kind),
        "latency_by_kind": per_kind_latency,
        "queue_depth": _stats([float(v) for v in queue_depth_samples]),
        "queue_depth_over_time": list(summary.queue_depth_samples),
        "offered_qps": summary.offered_qps,
        "observed_qps": summary.observed_qps,
        "wall_seconds": summary.wall_seconds,
        "stale_check_doc_ids": list(summary.stale_check_doc_ids),
    }


def write_summary_json(
    summary: WorkloadSummary,
    out_path: Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist the summary as a stable JSON file. Returns the file path."""
    payload = summarize(summary)
    if extra:
        payload["extra"] = extra
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def mlflow_metrics(summary: WorkloadSummary) -> dict[str, float]:
    """Flatten the summary into ``workload_``-prefixed MLflow metrics.

    Latency percentiles are emitted per op kind under
    ``workload_<kind>_p95_seconds`` etc. Counts and rates are unprefixed
    under ``workload_`` as well.
    """
    agg = summarize(summary)
    flat: dict[str, float] = {
        "workload_total_submitted": float(agg["total_submitted"]),
        "workload_total_completed": float(agg["total_completed"]),
        "workload_total_errors": float(agg["total_errors"]),
        "workload_total_stale_check": float(agg["total_stale_check"]),
        "workload_error_rate": float(agg["error_rate"]),
        "workload_stale_check_rate": float(agg["stale_check_rate"]),
        "workload_offered_qps": float(agg["offered_qps"]),
        "workload_observed_qps": float(agg["observed_qps"]),
        "workload_wall_seconds": float(agg["wall_seconds"]),
    }
    for kind, stats_dict in agg["latency_by_kind"].items():
        for stat_name, value in stats_dict.items():
            if stat_name == "count":
                flat[f"workload_{kind}_count"] = float(value)
            else:
                flat[f"workload_{kind}_{stat_name}_seconds"] = float(value)
    queue = agg["queue_depth"]
    for stat_name, value in queue.items():
        if stat_name == "count":
            continue
        flat[f"workload_queue_depth_{stat_name}"] = float(value)
    return flat
