"""Load a run's lineage and per-sample metrics from a report or MLflow (#65).

A run is referenced either by a JSON report path (optionally
``report.json#<config_name>`` when it holds several configs) or by an MLflow
run ID. Both resolve to the same ``RunRecord``, so a comparison does not care
where a run was stored. Reports written before lineage existed still load:
sample IDs are re-derived from content and every absent lineage field is
``unknown`` rather than a guess.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark.lineage.dataset import assign_sample_ids
from benchmark.regression.metrics import _RETRIEVAL_PREFIXES, sample_metrics

LINEAGE_ARTIFACT_DIR = "lineage"
LINEAGE_FILE = "lineage.json"
SAMPLES_FILE = "samples.json"


@dataclass(frozen=True)
class RunRecord:
    ref: str
    config_name: str | None
    lineage: dict[str, Any]
    # One entry per sample: sample_id, category, metrics (name -> value|None).
    samples: list[dict[str, Any]] = field(default_factory=list)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return None if math.isnan(value) else float(value)


def _abstention_category(category: str | None) -> str:
    return category if category in ("answerable", "unanswerable") else "unlabeled"


def per_sample_metrics(sample: Mapping[str, Any]) -> dict[str, float | None]:
    """Existing per-sample metrics (``benchmark.regression``) plus any other
    numeric RAGAS / non-retrieval custom score the run recorded."""
    metrics = sample_metrics(sample, _abstention_category(sample.get("category")))
    for name, value in (sample.get("ragas_scores") or {}).items():
        metrics.setdefault(name, _number(value))
    for name, value in (sample.get("custom_scores") or {}).items():
        if not name.startswith(_RETRIEVAL_PREFIXES):
            metrics.setdefault(name, _number(value))
    return metrics


def lineage_samples(per_sample: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Text-free per-sample records: ID, category, metrics."""
    derived = assign_sample_ids(per_sample)
    return [
        {
            "sample_id": sample.get("sample_id") or fallback,
            "category": sample.get("category"),
            "metrics": per_sample_metrics(sample),
        }
        for sample, fallback in zip(per_sample, derived)
    ]


def scorer_fingerprint(scorer: Mapping[str, Any] | None) -> str | None:
    if not scorer:
        return None
    return hashlib.sha256(json.dumps(scorer, sort_keys=True, default=str).encode()).hexdigest()


def _select_result(results: list[dict[str, Any]], config_name: str | None, ref: str):
    if config_name is not None:
        for result in results:
            if result.get("config_name") == config_name:
                return result
        raise ValueError(f"{ref}: no config named {config_name!r}")
    if len(results) != 1:
        names = ", ".join(str(r.get("config_name")) for r in results)
        raise ValueError(f"{ref} holds {len(results)} configs; pick one with '#<config_name>': {names}")
    return results[0]


def load_report_record(path: str | Path, config_name: str | None = None) -> RunRecord:
    from benchmark.reporting.exports import load_json_report

    report = load_json_report(Path(path))
    result = _select_result(report.get("results") or [], config_name, str(path))
    lineage = dict(result.get("lineage") or {})
    # Older reports: at least the dataset name is known from the run itself.
    lineage.setdefault("dataset", {}).setdefault("name", report.get("dataset_name"))
    return RunRecord(
        ref=str(path),
        config_name=result.get("config_name"),
        lineage=lineage,
        samples=lineage_samples(result.get("per_sample") or []),
    )


def load_mlflow_record(run_id: str) -> RunRecord:
    import mlflow

    base = f"runs:/{run_id}/{LINEAGE_ARTIFACT_DIR}"
    try:
        lineage = mlflow.artifacts.load_dict(f"{base}/{LINEAGE_FILE}")
        samples = mlflow.artifacts.load_dict(f"{base}/{SAMPLES_FILE}")["samples"]
    except Exception as exc:
        raise ValueError(
            f"MLflow run {run_id} has no lineage artifacts (logged before #65?); "
            "compare its JSON report instead"
        ) from exc
    config_name = mlflow.get_run(run_id).data.tags.get("config_name")
    return RunRecord(ref=run_id, config_name=config_name, lineage=lineage, samples=samples)


def load_run_record(ref: str) -> RunRecord:
    """Resolve a report path (``path[#config]``) or an MLflow run ID."""
    path_part, _, config_name = ref.partition("#")
    if Path(path_part).is_file():
        return load_report_record(path_part, config_name or None)
    return load_mlflow_record(ref)


# MLflow caps tag values at 8000 characters; free-text fields stay well below.
_TAG_LIMIT = 5000
_SYSTEM_TAGS = {
    "system_id": "system.id",
    "system_name": "system.name",
    "version_source": "system.version_source",
    "version": "system.version",
    "branch": "system.branch",
    "dirty": "system.dirty",
    "diff_hash": "system.diff_hash",
    "status": "system.status",
}
_DATASET_TAGS = (
    "name",
    "subset",
    "split",
    "sample_size",
    "sampling_seed",
    "num_samples",
    "question_set_hash",
    "source_sha256",
    "corpus_fingerprint",
)


def lineage_tags(lineage: Mapping[str, Any] | None, config_name: str | None = None) -> dict[str, str]:
    """Searchable MLflow tags: experiment.*, system.*, dataset.*, scorer.*.

    Hashes and identifiers only; no question text, no diffs, no secrets.
    """
    if not lineage:
        return {}
    tags: dict[str, str] = {}
    if config_name:
        tags["config_name"] = config_name
    for key, value in (lineage.get("experiment") or {}).items():
        tags[f"experiment.{key}"] = str(value)[:_TAG_LIMIT]
    system = lineage.get("system") or {}
    for key, tag in _SYSTEM_TAGS.items():
        if system.get(key) is not None:
            tags[tag] = str(system[key])
    dataset = lineage.get("dataset") or {}
    for key in _DATASET_TAGS:
        if dataset.get(key) is not None:
            tags[f"dataset.{key}"] = str(dataset[key])
    scorer = lineage.get("scorer") or {}
    if fingerprint := scorer_fingerprint(scorer):
        tags["scorer.fingerprint"] = fingerprint
    if scorer.get("eval_critic_llm"):
        tags["scorer.critic_llm"] = str(scorer["eval_critic_llm"])
    return tags
