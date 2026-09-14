"""Worker-only orchestration loop (issue 04).

The worker owns the only orchestration loop; ``python main.py`` is a thin CLI
that requires a manifest and delegates to it. These tests drive the worker
through its own interface with a stub RAG adapter, a fake tracking backend,
and temp run directories.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from benchmark.adapters import (
    AdapterGenerationResult,
    AdapterCapabilities,
    PreparedTarget,
    RetrievalResult,
    RetrievedChunk,
    register_rag_adapter,
)
from benchmark.orchestration.matrix import (
    ExperimentSpec,
    build_configs_from_spec,
)
from benchmark.orchestration.worker import ExperimentWorker, WorkerOptions

DOC_A = "The Eiffel Tower is a wrought-iron lattice tower in Paris, France."
SAMPLES = [
    {
        "question": "Where is the Eiffel Tower?",
        "ground_truth": "Paris, France",
        "metadata": {},
    }
]

STUB_ADAPTER_NAME = "stub-worker-loop"


class _StubAdapter:
    """Minimal managed adapter returning a deterministic answer."""

    name = STUB_ADAPTER_NAME
    instances = 0

    def __init__(self, config: Any) -> None:
        _StubAdapter.instances += 1

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            ingestion=True,
            retrieval=True,
            generation=True,
            token_usage=True,
            cleanup=True,
        )

    def prepare(self, config, data, corpus=None) -> PreparedTarget:
        return PreparedTarget(target_id="stub-target", metadata={"chunk_count": 1})

    def retrieve(self, target, sample, config) -> RetrievalResult:
        return RetrievalResult(
            chunks=(RetrievedChunk(text=DOC_A, rank=1),),
            total_seconds=0.01,
        )

    def generate(self, target, sample, config, retrieval=None):
        retrieval = retrieval or self.retrieve(target, sample, config)
        return AdapterGenerationResult(
            answer=f"stub answer to: {sample['question']}",
            retrieval=retrieval,
            ttft_seconds=0.01,
            total_seconds=0.02,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )

    def cleanup(self, target, config) -> None:
        pass


register_rag_adapter(STUB_ADAPTER_NAME, _StubAdapter)


@pytest.fixture(autouse=True)
def _reset_stub_counter():
    _StubAdapter.instances = 0
    yield
    _StubAdapter.instances = 0


def _one_cell_configs() -> list:
    spec = ExperimentSpec(
        name="worker-loop",
        dataset={},
        settings={
            "rag_system_adapter": STUB_ADAPTER_NAME,
            "ragas_enabled": False,
            "custom_metrics_enabled": False,
        },
        matrix={},
    )
    return build_configs_from_spec(spec)


@pytest.fixture
def fake_tracking(monkeypatch):
    import benchmark.orchestration.worker as worker_module
    import mlflow

    calls = {"runs": [], "aggregate": []}

    def fake_log_run(result, reproducibility_dir=None, nested=None):
        calls["runs"].append(result.config_name)

    def fake_log_aggregate(run_dir, run_name=None, reproducibility_dir=None):
        calls["aggregate"].append(str(run_dir))

    @contextmanager
    def fake_start_run(*args, **kwargs):
        yield None

    monkeypatch.setattr(worker_module, "log_benchmark_run", fake_log_run)
    monkeypatch.setattr(
        worker_module, "log_aggregate_artifacts_to_mlflow", fake_log_aggregate
    )
    monkeypatch.setattr(mlflow, "start_run", fake_start_run)
    return calls


@pytest.fixture
def stubbed_data(monkeypatch):
    import benchmark.orchestration.worker as worker_module

    monkeypatch.setattr(
        worker_module, "_load_data_once", lambda config: (SAMPLES, None, 0.01)
    )


def test_main_without_manifest_exits_with_yaml_pointer(monkeypatch, capsys):
    import main

    monkeypatch.delenv("BENCHMARK_CONFIG_FILE", raising=False)
    monkeypatch.delenv("EXPERIMENT_MANIFEST", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        main.main([])

    assert excinfo.value.code == 2
    out = capsys.readouterr().out
    assert "BENCHMARK_CONFIG_FILE=experiments/<name>.yaml" in out


def test_main_with_manifest_delegates_to_worker(monkeypatch, tmp_path):
    import main

    monkeypatch.setattr(main, "setup_tracing", lambda: "file:/tmp/fake")
    monkeypatch.setattr(main, "setup_mlflow", lambda: None)

    manifest = tmp_path / "my_experiment.yaml"
    manifest.write_text(
        """
experiment_name: delegate-test
dataset:
  sample_size: 2
matrix:
  retrieval_top_k: [3]
""".strip(),
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    class FakeWorker:
        def __init__(self, configs, options):
            captured["configs"] = configs
            captured["options"] = options

        def run(self):
            captured["ran"] = True
            return []

    monkeypatch.setattr(main, "ExperimentWorker", FakeWorker)

    results = main.main([str(manifest)])

    assert results == []
    assert captured["ran"] is True
    assert len(captured["configs"]) == 1
    assert captured["options"].experiment_name == "my_experiment"


def test_worker_run_one_cell_manifest_produces_artifacts_and_report(
    tmp_path, fake_tracking, stubbed_data
):
    configs = _one_cell_configs()
    run_dir = tmp_path / "run1"
    worker = ExperimentWorker(
        configs,
        WorkerOptions(run_dir=run_dir, experiment_name="worker-loop"),
    )

    results = worker.run()

    assert len(results) == 1
    config = configs[0]
    safe_name = config.name.replace(":", "_").replace("/", "_")

    # Per-config artifacts from the benchmark core.
    assert (run_dir / "configs" / f"{safe_name}_qa.json").exists()
    assert (run_dir / "configs" / f"{safe_name}_checkpoint.json").exists()

    # Worker bookkeeping.
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["configs"][config.name]["status"] == "completed"
    assert (run_dir / "worker_manifest.json").exists()
    assert (run_dir / "reproducibility").exists()

    # Report out.
    reports = list(run_dir.glob("benchmark_*.json"))
    assert len(reports) == 1

    # Fake tracking backend saw the parent bookkeeping.
    assert fake_tracking["runs"] == [config.name]
    assert len(fake_tracking["aggregate"]) == 1


def test_worker_resume_skips_completed_configs(tmp_path, fake_tracking, stubbed_data):
    configs = _one_cell_configs()
    run_dir = tmp_path / "run1"

    first = ExperimentWorker(
        configs, WorkerOptions(run_dir=run_dir, experiment_name="worker-loop")
    )
    first.run()
    assert _StubAdapter.instances == 1

    second = ExperimentWorker(
        configs, WorkerOptions(run_dir=run_dir, experiment_name="worker-loop")
    )
    results = second.run()

    assert results == []
    assert _StubAdapter.instances == 1  # no adapter was re-prepared
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["configs"][configs[0].name]["status"] == "completed"
