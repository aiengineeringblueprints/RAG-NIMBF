"""Load benchmark results from the filesystem for the Streamlit dashboard.

This module is intentionally dependency-light (stdlib + pandas only) so the
parsing logic can be unit-tested without a Streamlit session.

Data layout it reads from ``results/``:

- ``results/runN/benchmark_<timestamp>.json``  — aggregate JSON written once a
  sweep completes (``benchmark.reporting.exports.save_json_report``).
- ``results/runN/configs/<config>_qa.json``    — per-config QA log written
  immediately after each config finishes, i.e. the *live* source while a
  worker is still running.
- ``results/runN/progress.json``                — worker resume ledger with
  per-config status (``running`` / ``completed`` / ``failed``).
- ``results/runN/worker_manifest.json``         — config manifest + experiment
  name.
- ``results/runN/stage_timings/<config>.json``  — per-stage latency recorder.
- ``results/runN/llm_performance/<config>.json``— LLM load-test metrics.
- ``results/runN/resource_traces/<config>.csv`` — GPU/CPU resource trace.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

RUN_RE = re.compile(r"^run(\d+)$")

# Canonical RAGAS keys that appear flat on a config dict in the aggregate JSON.
RAGAS_KEYS = [
    "ragas_faithfulness",
    "ragas_answer_relevancy",
    "ragas_answer_correctness",
    "ragas_context_precision",
    "ragas_context_recall",
    "ragas_semantic_similarity",
]

# Legacy runs stored custom metrics as flat keys directly on the config dict.
LEGACY_FLAT_CUSTOM = [
    "hit@1", "hit@3", "hit@5",
    "ndcg@1", "ndcg@3", "ndcg@5",
    "recall@1", "recall@3", "recall@5",
    "context_relevance", "vec_dist_q_gt", "vec_dist_q_answer",
    "rouge_l", "bleu", "meteor",
    "bert_score_precision", "bert_score_recall", "bert_score_f1",
    "trace_utilization", "trace_completeness",
]

QUALITY_METRICS = [
    "ragas_faithfulness",
    "ragas_answer_relevancy",
    "ragas_answer_correctness",
    "ragas_context_precision",
    "ragas_context_recall",
    "ragas_semantic_similarity",
    "custom_context_relevance",
    "custom_hit_at_1",
    "custom_ndcg_at_1",
    "custom_rouge_l",
    "custom_bleu",
    "custom_meteor",
    "custom_bert_score_f1",
    "custom_trace_utilization",
    "custom_trace_completeness",
]

PERFORMANCE_METRICS = [
    "avg_ttft_seconds",
    "avg_tokens_per_second",
    "total_time_seconds",
    "total_input_tokens",
    "total_output_tokens",
    "total_tokens",
    "avg_gpu_utilization_pct",
    "avg_gpu_memory_used_mb",
    "avg_gpu_power_w",
    "energy_kwh",
    "estimated_energy_cost_usd",
    "total_estimated_cost_usd",
    "avg_estimated_cost_per_answer_usd",
]

# Human-readable labels for metric keys (German UI).
METRIC_LABELS: dict[str, str] = {
    "ragas_faithfulness": "Faithfulness",
    "ragas_answer_relevancy": "Answer Relevancy",
    "ragas_answer_correctness": "Answer Correctness",
    "ragas_context_precision": "Context Precision",
    "ragas_context_recall": "Context Recall",
    "ragas_semantic_similarity": "Semantic Similarity",
    "custom_context_relevance": "Context Relevance",
    "custom_hit_at_1": "Hit@1",
    "custom_hit_at_3": "Hit@3",
    "custom_hit_at_5": "Hit@5",
    "custom_ndcg_at_1": "NDCG@1",
    "custom_ndcg_at_3": "NDCG@3",
    "custom_ndcg_at_5": "NDCG@5",
    "custom_recall_at_1": "Recall@1",
    "custom_recall_at_3": "Recall@3",
    "custom_recall_at_5": "Recall@5",
    "custom_rouge_l": "ROUGE-L",
    "custom_bleu": "BLEU",
    "custom_meteor": "METEOR",
    "custom_bert_score_precision": "BERTScore Precision",
    "custom_bert_score_recall": "BERTScore Recall",
    "custom_bert_score_f1": "BERTScore F1",
    "custom_trace_utilization": "TRACe Utilization",
    "custom_trace_completeness": "TRACe Completeness",
    "custom_vec_dist_q_gt": "Vec-Dist Q→GT",
    "custom_vec_dist_q_answer": "Vec-Dist Q→Answer",
    "avg_ttft_seconds": "Ø TTFT (s)",
    "avg_tokens_per_second": "Ø Throughput (tok/s)",
    "total_time_seconds": "Gesamtzeit (s)",
    "total_input_tokens": "Input-Tokens",
    "total_output_tokens": "Output-Tokens",
    "total_tokens": "Gesamt-Tokens",
    "avg_gpu_utilization_pct": "Ø GPU-Auslastung (%)",
    "avg_gpu_memory_used_mb": "Ø GPU-Speicher (MB)",
    "avg_gpu_power_w": "Ø GPU-Leistung (W)",
    "energy_kwh": "Energie (kWh)",
    "estimated_energy_cost_usd": "Energiekosten (USD)",
    "total_estimated_cost_usd": "Geschätzte Kosten (USD)",
    "avg_estimated_cost_per_answer_usd": "Ø Kosten/Answer (USD)",
}


def metric_label(key: str) -> str:
    return METRIC_LABELS.get(key, key)


def _safe_name(config_name: str) -> str:
    return config_name.replace(":", "_").replace("/", "_")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _read_json_list(path: Path) -> list[dict] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _aggregate_path(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("benchmark_*.json"))
    return candidates[-1] if candidates else None


@dataclass(frozen=True)
class RunInfo:
    """Lightweight metadata about one results/runN directory."""

    run_dir: Path
    run_name: str
    status: str  # completed | running | failed | empty
    timestamp: str | None = None
    dataset_name: str | None = None
    dataset_subset: str | None = None
    dataset_sample_size: int | None = None
    num_configs: int = 0
    num_done: int = 0
    num_failed: int = 0
    system_info: dict[str, Any] | None = None
    progress: dict[str, Any] | None = None
    aggregate_path: Path | None = None


@dataclass
class LiveConfig:
    """Per-config status from progress.json plus any on-disk artifacts."""

    name: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    qa_log_path: Path | None = None
    stage_timings_path: Path | None = None
    llm_performance_path: Path | None = None


@dataclass
class RunData:
    """Everything the dashboard needs about one run."""

    info: RunInfo
    configs: list[dict[str, Any]] = field(default_factory=list)
    live_configs: list[LiveConfig] = field(default_factory=list)
    qa_logs: dict[str, list[dict]] = field(default_factory=dict)
    manifest: dict[str, Any] | None = None


def scan_runs(results_dir: Path) -> list[RunInfo]:
    """Scan results/ for run directories, newest first."""
    if not results_dir.is_dir():
        return []
    runs: list[RunInfo] = []
    for child in results_dir.iterdir():
        if not (child.is_dir() and RUN_RE.match(child.name)):
            continue
        runs.append(_inspect_run(child))
    runs.sort(key=lambda r: int(RUN_RE.match(r.run_name).group(1)), reverse=True)
    return runs


def _inspect_run(run_dir: Path) -> RunInfo:
    aggregate = _aggregate_path(run_dir)
    progress = _read_json(run_dir / "progress.json")
    manifest = _read_json(run_dir / "worker_manifest.json")

    status = "empty"
    timestamp: str | None = None
    dataset_name: str | None = None
    dataset_subset: str | None = None
    dataset_sample_size: int | None = None
    num_configs = 0
    num_done = 0
    num_failed = 0
    system_info: dict[str, Any] | None = None

    if aggregate is not None:
        data = _read_json(aggregate)
        status = "completed"
        if data:
            timestamp = data.get("timestamp")
            dataset = data.get("dataset") or {}
            dataset_name = dataset.get("name")
            dataset_subset = dataset.get("subset")
            dataset_sample_size = dataset.get("sample_size")
            num_configs = data.get("num_configs", 0)
            system_info = data.get("system_info")
        # All config results exist, so num_done equals num_configs.
        num_done = num_configs

    if progress:
        configs = progress.get("configs") or {}
        num_configs = max(num_configs, len(configs))
        num_done = sum(1 for c in configs.values() if c.get("status") == "completed")
        num_failed = sum(1 for c in configs.values() if c.get("status") == "failed")
        running = sum(1 for c in configs.values() if c.get("status") == "running")
        if aggregate is None and (running or num_done or num_failed):
            status = "running"
        elif aggregate is None and (num_failed and not num_done and not running):
            status = "failed"

    if manifest:
        dataset_name = dataset_name or manifest.get("dataset_name")
        if not dataset_name:
            exp = manifest.get("experiment_name")
            dataset_name = dataset_name or (exp if exp else None)

    if timestamp is None:
        try:
            timestamp = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime(
                "%Y%m%d_%H%M%S"
            )
        except OSError:
            timestamp = None

    return RunInfo(
        run_dir=run_dir,
        run_name=run_dir.name,
        status=status,
        timestamp=timestamp,
        dataset_name=dataset_name,
        dataset_subset=dataset_subset,
        dataset_sample_size=dataset_sample_size,
        num_configs=num_configs,
        num_done=num_done,
        num_failed=num_failed,
        system_info=system_info,
        progress=progress,
        aggregate_path=aggregate,
    )


def load_run(run_dir: Path) -> RunData:
    """Load the full run: aggregate configs, live progress and QA logs."""
    info = _inspect_run(run_dir)
    configs: list[dict[str, Any]] = []
    if info.aggregate_path is not None:
        data = _read_json(info.aggregate_path) or {}
        configs = list(data.get("results", []) or [])

    live_configs = _load_live_configs(run_dir)
    qa_logs: dict[str, list[dict]] = {}
    for live in live_configs:
        if live.qa_log_path and live.qa_log_path.exists():
            qa = _read_json_list(live.qa_log_path)
            if qa:
                qa_logs[live.name] = qa

    manifest = _read_json(run_dir / "worker_manifest.json")

    # If no aggregate exists yet, synthesize config dicts from QA logs so the
    # dashboard already shows partial results while the worker is running.
    if not configs and live_configs:
        for live in live_configs:
            if live.qa_log_path and live.qa_log_path.exists():
                configs.append(_config_from_qa_log(live, run_dir))

    return RunData(
        info=info,
        configs=configs,
        live_configs=live_configs,
        qa_logs=qa_logs,
        manifest=manifest,
    )


def _load_live_configs(run_dir: Path) -> list[LiveConfig]:
    progress = _read_json(run_dir / "progress.json") or {}
    ledger = progress.get("configs") or {}
    configs_dir = run_dir / "configs"

    live: list[LiveConfig] = []
    for name, record in ledger.items():
        status = record.get("status", "unknown")
        live.append(
            LiveConfig(
                name=name,
                status=status,
                started_at=record.get("started_at"),
                completed_at=record.get("completed_at"),
                error=record.get("error"),
                qa_log_path=configs_dir / f"{_safe_name(name)}_qa.json",
                stage_timings_path=(
                    run_dir / "stage_timings" / f"{_safe_name(name)}.json"
                ),
                llm_performance_path=(
                    run_dir / "llm_performance" / f"{_safe_name(name)}.json"
                ),
            )
        )

    # Include QA logs not present in the ledger (e.g. plain main.py runs).
    known = {c.name for c in live}
    if configs_dir.is_dir():
        for qa_path in sorted(configs_dir.glob("*_qa.json")):
            name = qa_path.name[: -len("_qa.json")]
            if name in known:
                continue
            live.append(
                LiveConfig(
                    name=name,
                    status="completed",
                    qa_log_path=qa_path,
                )
            )
    live.sort(key=lambda c: c.name)
    return live


def _config_from_qa_log(live: LiveConfig, run_dir: Path) -> dict[str, Any]:
    qa = _read_json_list(live.qa_log_path) or []
    total_seconds = sum(float(s.get("total_seconds") or 0) for s in qa)
    output_tokens = sum(int(s.get("output_tokens") or s.get("token_count") or 0) for s in qa)
    input_tokens = sum(int(s.get("input_tokens") or 0) for s in qa)
    per_sample = []
    for s in qa:
        row = dict(s)
        row.setdefault("ragas_scores", {})
        row.setdefault("custom_scores", {})
        per_sample.append(row)
    return {
        "config_name": live.name,
        "llm_model": None,
        "embedding_model": None,
        "prompt_template": None,
        "chunking_strategy": None,
        "chunk_size": None,
        "chunk_overlap": None,
        "num_questions": len(qa),
        "num_chunks": 0,
        "avg_ttft_seconds": _mean(qa, "ttft_seconds"),
        "avg_tokens_per_second": _mean(qa, "tokens_per_second"),
        "total_time_seconds": total_seconds,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "per_sample": per_sample,
        "stats": {},
        "stage_timings": _read_json(live.stage_timings_path) if live.stage_timings_path else None,
        "_live": True,
    }


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(values) / len(values) if values else None


def config_metrics(config: dict[str, Any]) -> dict[str, float | None]:
    """Normalize every metric the dashboard knows about into one flat map."""
    metrics: dict[str, float | None] = {}

    for key in RAGAS_KEYS:
        metrics[key] = _as_float(config.get(key))

    cmm = config.get("custom_metric_means") or {}
    if isinstance(cmm, dict):
        for k, v in cmm.items():
            metrics[_custom_key(k)] = _as_float(v)

    for k in LEGACY_FLAT_CUSTOM:
        if k in config:
            metrics.setdefault(_custom_key(k), _as_float(config.get(k)))

    # Newer runs may already use custom_ prefixed flat keys.
    for k, v in config.items():
        if k.startswith("custom_") and not k.endswith("_stats"):
            metrics.setdefault(k, _as_float(v))

    for key in PERFORMANCE_METRICS:
        metrics.setdefault(key, _as_float(config.get(key)))

    llm_perf = config.get("llm_performance_metrics") or {}
    if isinstance(llm_perf, dict):
        for k, v in llm_perf.items():
            metrics[f"llm_perf_{k}"] = _as_float(v)

    return metrics


def _custom_key(legacy: str) -> str:
    return "custom_" + legacy.replace("@", "_at_")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_config(config: dict[str, Any]) -> dict[str, Any]:
    """One table row per config with identity + headline metrics."""
    return {
        "config_name": config.get("config_name"),
        "llm_model": config.get("llm_model"),
        "embedding_model": config.get("embedding_model"),
        "prompt_template": config.get("prompt_template"),
        "chunking_strategy": config.get("chunking_strategy"),
        "chunk_size": config.get("chunk_size"),
        "chunk_overlap": config.get("chunk_overlap"),
        "retrieval_top_k": config.get("retrieval_top_k"),
        "num_questions": config.get("num_questions"),
        "num_chunks": config.get("num_chunks"),
        "dataset": config.get("dataset_name"),
        "vector_db_backend": config.get("vector_db_backend"),
        "total_time_seconds": config.get("total_time_seconds"),
        "avg_ttft_seconds": config.get("avg_ttft_seconds"),
        "avg_tokens_per_second": config.get("avg_tokens_per_second"),
        "total_tokens": config.get("total_tokens"),
        "ragas_faithfulness": config.get("ragas_faithfulness"),
        "ragas_answer_correctness": config.get("ragas_answer_correctness"),
        "ragas_context_recall": config.get("ragas_context_recall"),
        "custom_metric_means": config.get("custom_metric_means"),
    }


def per_sample_metric_values(
    config: dict[str, Any], metric: str
) -> list[float]:
    """Extract per-sample values for a metric to build box/CI plots.

    Handles ragas_*, custom_* (from per-sample custom_scores) and flat
    latency fields on each sample.
    """
    per_sample = config.get("per_sample") or []
    values: list[float] = []
    for sample in per_sample:
        value: Any = None
        if metric.startswith("ragas_"):
            scores = sample.get("ragas_scores") or {}
            value = scores.get(metric) or scores.get(metric[len("ragas_") :])
        elif metric.startswith("custom_"):
            scores = sample.get("custom_scores") or {}
            raw_key = metric[len("custom_") :]
            value = scores.get(metric) or scores.get(raw_key)
        else:
            value = sample.get(metric)
        f = _as_float(value)
        if f is not None:
            values.append(f)
    return values