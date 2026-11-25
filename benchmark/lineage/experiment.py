"""Experiment metadata and scorer identity (issue #65)."""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

from benchmark.lineage.json_object import parse_json_object

EXPERIMENT_FIELDS = (
    "study_id",
    "hypothesis",
    "candidate_id",
    "baseline_run_id",
    "change_under_test",
    "change_summary",
    "issue_reference",
)

# Settings that decide how answers are *scored*. Two runs scored differently
# cannot be compared metric by metric, whatever system they measured.
SCORER_FIELDS = (
    "eval_critic_llm",
    "eval_critic_embedding",
    "eval_critic_temperature",
    "eval_critic_max_tokens",
    "eval_critic_thinking_control",
    "ragas_enabled",
    "ragas_metrics",
    "custom_metrics_enabled",
    "custom_metrics_bert_model",
    "custom_retrieval_metrics_mode",
    "mlflow_scorers",
)


def parse_experiment_metadata(value: str | Mapping[str, Any] | None) -> dict[str, str]:
    """Validate ``experiment_metadata_json``; unknown keys are rejected.

    A candidate is only meaningful against a baseline, so ``candidate_id``
    requires ``baseline_run_id``.
    """
    data = parse_json_object(
        value,
        setting="experiment_metadata_json",
        label="experiment metadata",
        allowed=EXPERIMENT_FIELDS,
    )
    if data is None:
        return {}
    parsed = {key: str(data[key]) for key in EXPERIMENT_FIELDS if data.get(key) is not None}
    if "candidate_id" in parsed and "baseline_run_id" not in parsed:
        raise ValueError(
            "experiment metadata: candidate_id requires baseline_run_id "
            "(the run the candidate is compared against)"
        )
    return parsed


def _plain(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


# Scoring code shipped with the framework: a change here changes the scores
# even when every setting stays the same.
_EVALUATOR_MODULES = (
    "answer_match.py",
    "custom_metrics.py",
    "evaluation.py",
    "gold_retrieval_metrics.py",
    "metrics.py",
    "mlflow_scorers.py",
)
# Third-party scorer libraries whose version moves the numbers.
_EVALUATOR_PACKAGES = ("ragas", "bert-score", "mlflow", "langchain-openai")
_BENCHMARK_DIR = Path(__file__).resolve().parents[1]


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def evaluator_implementation() -> dict[str, Any]:
    """Versions of the scoring code: framework evaluator sources + libraries."""
    digest = hashlib.sha256()
    for module in _EVALUATOR_MODULES:
        path = _BENCHMARK_DIR / module
        digest.update(module.encode())
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    return {
        "evaluator_source_sha256": digest.hexdigest(),
        "packages": {name: _package_version(name) for name in _EVALUATOR_PACKAGES},
    }


def scorer_identity(config: Any) -> dict[str, Any]:
    """The evaluator settings and implementation of a run (no URLs, no keys)."""
    identity: dict[str, Any] = {
        field: _plain(getattr(config, field, None)) for field in SCORER_FIELDS
    }
    identity["implementation"] = evaluator_implementation()
    return identity
