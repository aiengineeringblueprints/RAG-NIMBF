"""Publish a run comparison as its own MLflow run (issue #65)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from benchmark.lineage.runs import lineage_tags


def log_comparison_to_mlflow(result: Mapping[str, Any], json_path: Path, md_path: Path) -> str:
    """Log the comparison as its own MLflow run, linked to both inputs by tag."""
    import mlflow

    tags = {
        "run_type": "comparison",
        "comparison.status": result["status"],
        "comparison.baseline": str(result["baseline"]["ref"]),
        "comparison.candidate": str(result["candidate"]["ref"]),
    }
    tags.update(lineage_tags({"experiment": result["candidate"]["experiment"]}))
    for side in ("baseline", "candidate"):
        if result[side].get("mlflow_run_id"):
            tags[f"comparison.{side}_run_id"] = result[side]["mlflow_run_id"]
    with mlflow.start_run(run_name=f"compare:{json_path.parent.name}"[:250], tags=tags) as run:
        for name, m in result["metrics"].items():
            if m["delta"] is not None:
                mlflow.log_metric(f"delta.{name}", m["delta"])
        mlflow.log_artifact(str(json_path), artifact_path="comparison")
        mlflow.log_artifact(str(md_path), artifact_path="comparison")
        return run.info.run_id
