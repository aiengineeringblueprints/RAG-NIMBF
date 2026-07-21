"""Resumable single-machine benchmark worker."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
from rich.console import Console

from config import BenchmarkConfig
from benchmark.reporting import generate_report
from benchmark.reporting.exports import _result_to_dict
from benchmark.reporting.models import BenchmarkResultExtended
from benchmark.reproducibility import write_reproducibility_bundle
from benchmark.resource_monitor import (
    ResourceMonitor,
    enabled_from_env as resource_monitor_enabled,
    gpu_index_from_env as resource_monitor_gpu_index,
    interval_from_env as resource_monitor_interval,
)
from benchmark.tracking import (
    log_aggregate_artifacts_to_mlflow,
    log_benchmark_run,
)
from benchmark.orchestration.matrix import summarize_matrix


console = Console()


@dataclass(frozen=True)
class WorkerOptions:
    run_dir: Path | None = None
    resume: bool = True
    keep_going: bool = False
    dry_run: bool = False
    experiment_name: str = "benchmark-worker"
    write_reports: bool = True
    log_mlflow: bool = True


class ProgressStore:
    """Small JSON progress ledger for resume/retry."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {"configs": {}}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            self.data.setdefault("configs", {})

    def is_completed(
        self,
        config: BenchmarkConfig,
        result_path: Path | None = None,
    ) -> bool:
        record = self.data["configs"].get(config.name)
        if not (record and record.get("status") == "completed"):
            return False
        if result_path is not None:
            return result_path.exists()
        recorded_path = record.get("result_path")
        return Path(recorded_path).exists() if recorded_path else True

    def mark_running(self, config: BenchmarkConfig) -> None:
        self._set(config, {"status": "running", "started_at": _now()})

    def mark_completed(
        self,
        config: BenchmarkConfig,
        result_path: Path | None = None,
    ) -> None:
        patch: dict[str, Any] = {
            "status": "completed",
            "completed_at": _now(),
        }
        if result_path is not None:
            patch["result_path"] = str(result_path)
        self._set(config, patch)

    def mark_failed(self, config: BenchmarkConfig, error: str) -> None:
        self._set(
            config,
            {
                "status": "failed",
                "failed_at": _now(),
                "error": error,
            },
        )

    def _set(self, config: BenchmarkConfig, patch: dict[str, Any]) -> None:
        current = self.data["configs"].get(config.name, {})
        current.update(patch)
        self.data["configs"][config.name] = current
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")


class ExperimentWorker:
    """Run a concrete BenchmarkConfig list on one machine."""

    def __init__(self, configs: list[BenchmarkConfig], options: WorkerOptions):
        if not configs:
            raise ValueError("No benchmark configurations to run")
        self.configs = configs
        self.options = options

    def plan(self) -> dict[str, Any]:
        return summarize_matrix(self.configs)

    def run(self) -> list[BenchmarkResultExtended]:
        if self.options.dry_run:
            summary = self.plan()
            console.print(json.dumps(summary, indent=2))
            return []

        from main import run_single_benchmark

        run_dir = self.options.run_dir or _next_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        progress = ProgressStore(run_dir / "progress.json")

        data, corpus, load_data_seconds = _load_data_once(self.configs[0])
        reproducibility_dir = write_reproducibility_bundle(run_dir, self.configs)
        _write_worker_manifest(run_dir, self.options.experiment_name, self.configs)

        console.print(
            f"[bold]Worker running {len(self.configs)} configuration(s) "
            f"into {run_dir}[/bold]"
        )

        results: list[BenchmarkResultExtended] = []
        wall_start = time.perf_counter()

        parent_context = (
            mlflow.start_run(
                run_name=f"benchmark_{self.options.experiment_name}_{run_dir.name}",
                tags={
                    "type": "benchmark_parent",
                    "experiment_name": self.options.experiment_name,
                    "results_dir": str(run_dir),
                    "num_configs": str(len(self.configs)),
                },
            )
            if self.options.log_mlflow
            else _nullcontext()
        )
        with parent_context:
            for index, config in enumerate(self.configs, start=1):
                console.print(f"\n[bold cyan]Worker config {index}/{len(self.configs)}[/bold cyan]")

                if self.options.resume and progress.is_completed(config):
                    console.print(f"[dim]Skipping completed config: {config.name}[/dim]")
                    continue

                progress.mark_running(config)
                monitor = _resource_monitor_for(run_dir, config)
                try:
                    if monitor is None:
                        result = run_single_benchmark(
                            config,
                            data,
                            run_dir=run_dir,
                            corpus=corpus,
                            load_data_seconds=load_data_seconds,
                        )
                    else:
                        with monitor:
                            result = run_single_benchmark(
                                config,
                                data,
                                run_dir=run_dir,
                                corpus=corpus,
                                load_data_seconds=load_data_seconds,
                                resource_monitor=monitor,
                            )
                        console.print(f"[dim]  Resource trace: {monitor.trace_path}[/dim]")


                    progress.mark_completed(config)
                    if self.options.log_mlflow:
                        log_benchmark_run(
                            result,
                            reproducibility_dir=reproducibility_dir,
                            nested=True,
                        )
                    results.append(result)
                except Exception as exc:
                    progress.mark_failed(config, str(exc))
                    if not self.options.keep_going:
                        raise
                    console.print(f"[red]Config failed and worker is continuing: {exc}[/red]")

            if results and self.options.write_reports:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                generate_report(
                    results,
                    results_dir=run_dir,
                    timestamp=timestamp,
                    dataset_name=self.configs[0].dataset_name,
                    dataset_subset=self.configs[0].dataset_subset,
                    dataset_sample_size=self.configs[0].dataset_sample_size,
                    total_time=time.perf_counter() - wall_start,
                )
                if self.options.log_mlflow:
                    log_aggregate_artifacts_to_mlflow(
                        run_dir,
                        run_name=f"summary_{run_dir.name}_{timestamp}",
                        reproducibility_dir=reproducibility_dir,
                    )

        return results




def config_result_path(run_dir: Path, config: BenchmarkConfig) -> Path:
    safe_name = config.name.replace(":", "_").replace("/", "_")
    return run_dir / "configs" / f"{safe_name}.json"


def _load_data_once(config: BenchmarkConfig) -> tuple[list[dict], list[dict] | None, float]:
    from benchmark.dataset import load_benchmark_data, load_corpus_and_questions
    from benchmark.dataset_adapters import resolve_adapter
    from main import _stage_timer

    load_stage: dict[str, float] = {}
    with _stage_timer(load_stage, "load_data"):
        adapter = resolve_adapter(config.dataset_name)
        if adapter.has_shared_corpus:
            corpus, data = load_corpus_and_questions(
                dataset_name=config.dataset_name,
                subset=config.dataset_subset or None,
                sample_size=config.dataset_sample_size,
                dataset_path=config.dataset_path,
                question_field=config.dataset_question_field,
                ground_truth_field=config.dataset_ground_truth_field,
                context_field=config.dataset_context_field,
                metadata_field=config.dataset_metadata_field,
            )
        else:
            corpus = None
            data = load_benchmark_data(
                dataset_name=config.dataset_name,
                subset=config.dataset_subset or None,
                sample_size=config.dataset_sample_size,
                dataset_path=config.dataset_path,
                question_field=config.dataset_question_field,
                ground_truth_field=config.dataset_ground_truth_field,
                context_field=config.dataset_context_field,
                metadata_field=config.dataset_metadata_field,
            )
    return data, corpus, load_stage.get("load_data", 0.0)


def _resource_monitor_for(run_dir: Path, config: BenchmarkConfig) -> ResourceMonitor | None:
    if not resource_monitor_enabled():
        return None
    safe_name = config.name.replace(":", "_").replace("/", "_")
    trace_dir = run_dir / "resource_traces"
    return ResourceMonitor(
        trace_dir / f"{safe_name}.csv",
        trace_dir / f"{safe_name}_markers.csv",
        interval_seconds=resource_monitor_interval(),
        gpu_index=resource_monitor_gpu_index(),
    )


def _next_run_dir(base: Path = Path("results")) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    max_run = 0
    for child in base.iterdir():
        if child.is_dir() and child.name.startswith("run"):
            try:
                max_run = max(max_run, int(child.name[3:]))
            except ValueError:
                pass
    return base / f"run{max_run + 1}"


def _write_worker_manifest(
    run_dir: Path,
    experiment_name: str,
    configs: list[BenchmarkConfig],
) -> None:
    manifest = {
        "experiment_name": experiment_name,
        "created_at": _now(),
        "num_configs": len(configs),
        "configs": [c.name for c in configs],
    }
    (run_dir / "worker_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False
