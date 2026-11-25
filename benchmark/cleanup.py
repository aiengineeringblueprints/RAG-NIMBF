"""Safe local cleanup of temporary run output (issue #66).

Storage policy: MLflow is the durable record for completed runs
(parameters, metrics, provenance, canonical per-sample artifacts); the
local ``results/runN/`` directory is a temporary working area kept only
for checkpointing and resumability.

This module verifies that a configuration's MLflow run finished and
carries its expected parameters and artifacts, and only then allows the
local per-config working files to be removed. Cleanup is opt-in: the
module-level default is a dry-run, and failed / interrupted /
incompletely logged runs are always protected.

CLI::

    python -m benchmark.cleanup results/runN            # dry-run plan
    python -m benchmark.cleanup results/runN --execute  # actually delete
    python -m benchmark.cleanup results/runN --execute --prune-samples
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow

logger = logging.getLogger(__name__)

#: Per-config local working files that duplicate data already in MLflow.
#: The samples sidecar duplicates the canonical per-sample CSV (which
#: includes contexts) on the config's MLflow run.
CONFIG_FILE_SUFFIXES = (".json", "_qa.json", "_checkpoint.json", ".samples.json.gz")


@dataclass(frozen=True)
class CleanupAction:
    """One config whose local working files may be removed."""

    config_name: str
    mlflow_run_id: str
    paths: tuple[Path, ...]
    verification: str = "verified"


@dataclass
class CleanupReport:
    actions: list[CleanupAction] = field(default_factory=list)
    removed: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def verify_config_run(
    run_id: str,
    client: mlflow.MlflowClient | None = None,
) -> tuple[bool, str]:
    """Check that an MLflow run finished and holds its expected data.

    A run counts as fully logged when it exists, its status is
    ``FINISHED``, at least one parameter was logged, and the canonical
    per-sample artifact ``data/per_sample_results.csv`` (which includes
    the retrieved contexts, issue #66) is present. Returns ``(ok,
    reason)``.
    """
    try:
        client = client or mlflow.MlflowClient()
        run = client.get_run(run_id)
    except Exception as exc:
        return False, f"run not found: {exc}"
    if run.info.status != "FINISHED":
        return False, f"status is {run.info.status}, not FINISHED"
    if not run.data.params:
        return False, "no parameters logged"
    try:
        data_artifacts = {
            getattr(item, "path", item)
            for item in client.list_artifacts(run_id, "data")
        }
    except Exception as exc:
        return False, f"artifact listing failed: {exc}"
    if "data/per_sample_results.csv" not in data_artifacts:
        return False, "canonical per-sample artifact data/per_sample_results.csv missing"
    return True, ""


def _config_working_paths(run_dir: Path, safe_name: str) -> tuple[Path, ...]:
    configs_dir = run_dir / "configs"
    return tuple(
        path
        for suffix in CONFIG_FILE_SUFFIXES
        if (path := configs_dir / f"{safe_name}{suffix}").exists()
    )


def plan_cleanup(
    run_dir: Path,
    client: mlflow.MlflowClient | None = None,
) -> list[CleanupAction]:
    """List configs whose local working files can safely be removed.

    A config is selected only when the progress ledger marks it
    ``completed`` with a recorded ``mlflow_run_id`` whose run passes
    :func:`verify_config_run`. Everything else (failed, interrupted,
    unverified, still-running) is protected.
    """
    from benchmark.orchestration.worker import ProgressStore

    progress_path = run_dir / "progress.json"
    if not progress_path.exists():
        return []
    ledger = ProgressStore(progress_path)
    actions: list[CleanupAction] = []
    for name, record in ledger.data.get("configs", {}).items():
        if record.get("status") != "completed":
            continue
        run_id = record.get("mlflow_run_id")
        if not run_id:
            continue
        ok, reason = verify_config_run(run_id, client)
        if not ok:
            logger.info(
                "Protecting config '%s': MLflow run %s not verified (%s)",
                name,
                run_id,
                reason,
            )
            continue
        paths = _config_working_paths(run_dir, _safe(name))
        if paths:
            actions.append(CleanupAction(name, run_id, paths))
    return actions


def _all_completed_configs_verified(
    run_dir: Path, client: mlflow.MlflowClient | None = None
) -> bool:
    """True only when ``progress.json`` exists and *every* completed
    config has a verified MLflow run.

    Guards ``--prune-samples``: if any completed config lacks a verified
    per-sample record in MLflow, its sidecars may be the only remaining
    copy of the retrieved contexts and must not be deleted (issue #66
    review).
    """
    ledger = run_dir / "progress.json"
    if not ledger.exists():
        return False
    try:
        data = json.loads(ledger.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", ledger, exc)
        return False
    completed = [
        rec
        for rec in (data.get("configs") or {}).values()
        if rec.get("status") == "completed"
    ]
    if not completed:
        return False
    try:
        client = client or mlflow.MlflowClient()
    except Exception as exc:
        logger.warning("MLflow client unavailable: %s", exc)
        return False
    return all(
        rec.get("mlflow_run_id")
        and verify_config_run(rec["mlflow_run_id"], client)[0]
        for rec in completed
    )


def cleanup_run_dir(
    run_dir: Path,
    *,
    execute: bool = False,
    prune_samples: bool = False,
    client: mlflow.MlflowClient | None = None,
) -> tuple[list[Path], list[CleanupAction]]:
    """Dry-run (default) or execute cleanup of a run directory.

    Removes per-config working files for verified configs. With
    ``prune_samples`` the sweep-level ``*.samples.json.gz`` sidecars are
    also removed (the full per-sample data lives on each config's MLflow
    run). Pruning only happens when ``progress.json`` exists and every
    completed config has a verified MLflow run. Never touches
    ``progress.json``, configs with verified MLflow records only, and
    the sweep summary report.
    """
    actions = plan_cleanup(run_dir, client)
    removed: list[Path] = []
    if execute:
        for action in actions:
            for path in action.paths:
                try:
                    path.unlink()
                    removed.append(path)
                except OSError as exc:
                    logger.warning("Could not remove %s: %s", path, exc)
    if prune_samples and execute:
        if _all_completed_configs_verified(run_dir, client):
            for path in run_dir.glob("benchmark_*.samples.json.gz"):
                try:
                    path.unlink()
                    removed.append(path)
                except OSError as exc:
                    logger.warning("Could not remove %s: %s", path, exc)
        else:
            logger.warning(
                "Skipping sample pruning: not every completed config has a "
                "verified MLflow record; sidecars may be the only copy of "
                "the retrieved contexts."
            )
    return removed, actions


def _safe(name: str) -> str:
    return name.replace(":", "_").replace("/", "_")


class CleanupTool:
    """Object-oriented wrapper, convenient for callers that plan first
    and execute later."""

    def __init__(self, run_dir: Path, prune_samples: bool = False):
        self.run_dir = Path(run_dir)
        self.prune_samples = prune_samples

    def plan(self) -> list[CleanupAction]:
        return plan_cleanup(self.run_dir)

    def execute(self) -> list[Path]:
        removed, _ = cleanup_run_dir(
            self.run_dir, execute=True, prune_samples=self.prune_samples
        )
        return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean up temporary local run output after verifying the "
            "MLflow record (dry-run by default)."
        ),
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files (default: dry-run plan only).",
    )
    parser.add_argument(
        "--prune-samples",
        action="store_true",
        help="Also remove sweep-level *.samples.json.gz sidecars.",
    )
    args = parser.parse_args()

    removed, actions = cleanup_run_dir(
        args.run_dir, execute=args.execute, prune_samples=args.prune_samples
    )
    mode = "EXECUTED" if args.execute else "DRY-RUN"
    print(f"[{mode}] {len(actions)} config(s) eligible for cleanup:")
    for action in actions:
        print(f"  {action.config_name} (mlflow run {action.mlflow_run_id})")
        for path in action.paths:
            marker = "removed" if path in removed else "would remove"
            print(f"    {marker}: {path}")
    if args.prune_samples:
        for path in sorted(args.run_dir.glob("benchmark_*.samples.json.gz")):
            marker = "removed" if path in removed else "would remove"
            print(f"  {marker}: {path}")
    if not args.execute:
        print("Dry-run only; pass --execute to delete the listed files.")


if __name__ == "__main__":
    main()
