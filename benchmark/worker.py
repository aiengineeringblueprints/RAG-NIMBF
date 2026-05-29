"""CLI for planning and running resumable benchmark experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rich.console import Console

from config import get_all_combinations
from benchmark.orchestration.matrix import (
    build_configs_from_spec,
    load_experiment_spec,
)
from benchmark.orchestration.worker import ExperimentWorker, WorkerOptions
from benchmark.tracking import setup_mlflow
from benchmark.tracing import setup_tracing


console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or run a resumable benchmark configuration matrix.",
    )
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument(
        "manifest",
        nargs="?",
        help=(
            "Optional JSON/YAML experiment manifest. Defaults to "
            "BENCHMARK_CONFIG_FILE, then the legacy .env matrix."
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--no-reports", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else None
    spec = load_experiment_spec(manifest_path) if manifest_path else None
    configs = build_configs_from_spec(spec) if spec else get_all_combinations()
    configured_manifest = os.getenv("BENCHMARK_CONFIG_FILE") or os.getenv(
        "EXPERIMENT_MANIFEST"
    )
    experiment_name = (
        spec.name
        if spec
        else Path(configured_manifest).stem
        if configured_manifest
        else "env-matrix"
    )

    options = WorkerOptions(
        run_dir=args.run_dir,
        resume=not args.no_resume,
        keep_going=args.keep_going,
        dry_run=args.command == "plan",
        experiment_name=experiment_name,
        write_reports=not args.no_reports,
        log_mlflow=not args.no_mlflow,
    )
    worker = ExperimentWorker(configs, options)

    if args.command == "plan":
        console.print(json.dumps(worker.plan(), indent=2))
        return

    tracking_uri = setup_tracing()
    console.print(f"[dim]MLflow tracking: {tracking_uri}[/dim]")
    if options.log_mlflow:
        setup_mlflow()
    worker.run()


if __name__ == "__main__":
    main()
