"""Experiment metadata and scorer identity (issue #65)."""

from __future__ import annotations

import json
from typing import Any, Mapping

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
    "ragas_enabled",
    "ragas_metrics",
    "custom_metrics_enabled",
    "custom_metrics_bert_model",
    "custom_retrieval_metrics_mode",
    "mlflow_scorers",
)


def parse_experiment_metadata(value: str | Mapping[str, Any] | None) -> dict[str, str]:
    """Validate ``experiment_metadata_json``; unknown keys are rejected."""
    if value is None or value == "":
        return {}
    data = json.loads(value) if isinstance(value, str) else dict(value)
    if not isinstance(data, dict):
        raise ValueError("experiment_metadata_json must contain a JSON object")
    unknown = sorted(set(data) - set(EXPERIMENT_FIELDS))
    if unknown:
        raise ValueError(
            f"Unknown experiment metadata field(s): {', '.join(unknown)}; "
            f"allowed: {', '.join(EXPERIMENT_FIELDS)}"
        )
    return {key: str(data[key]) for key in EXPERIMENT_FIELDS if data.get(key) is not None}


def _plain(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def scorer_identity(config: Any) -> dict[str, Any]:
    """The evaluator settings of a run (no URLs, no keys)."""
    return {field: _plain(getattr(config, field, None)) for field in SCORER_FIELDS}
