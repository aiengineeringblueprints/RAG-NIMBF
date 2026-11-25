"""Storage policy tests for issue #66: MLflow as canonical record.

Covers: aggregate-run artifact deduplication, safe local cleanup with
dry-run and failure protection, MLflow upload verification, and the
configurable run output root.
"""

import json
from pathlib import Path

import pytest

from benchmark.cleanup import (
    CleanupTool,
    cleanup_run_dir,
    plan_cleanup,
    verify_config_run,
)
from benchmark.orchestration.worker import ProgressStore, _next_run_dir


class _FakeRunInfo:
    def __init__(self, status):
        self.status = status


class _FakeRun:
    def __init__(self, status, params, data_artifacts):
        self.info = _FakeRunInfo(status)
        self.data = type("D", (), {"params": params})()
        self._data_artifacts = data_artifacts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeClient:
    """Fake MlflowClient: maps run_id -> _FakeRun."""

    runs: dict = {}

    @classmethod
    def configure(cls, runs):
        cls.runs = runs

    def __init__(self, *args, **kwargs):
        pass

    def get_run(self, run_id):
        return self.runs[run_id]

    def list_artifacts(self, run_id, path=None):
        run = self.runs[run_id]
        if path == "data":
            return [
                type("F", (), {"path": p})() for p in run._data_artifacts
            ]
        return ["data"] if run._data_artifacts else []


@pytest.fixture(autouse=True)
def fake_client(monkeypatch):
    _FakeClient.configure({})
    monkeypatch.setattr(
        "benchmark.cleanup.mlflow.MlflowClient", _FakeClient
    )


def _make_run_dir(tmp_path: Path, completed=True, with_run_id=True) -> Path:
    run_dir = tmp_path / "run1"
    configs = run_dir / "configs"
    configs.mkdir(parents=True)
    safe = "test_config"
    (configs / f"{safe}.json").write_text("{}")
    (configs / f"{safe}_qa.json").write_text("[]")
    (configs / f"{safe}_checkpoint.json").write_text("{}")
    (configs / f"{safe}.samples.json.gz").write_bytes(b"")
    (run_dir / "benchmark_20260101_000000.json").write_text("{}")
    (run_dir / "benchmark_20260101_000000.samples.json.gz").write_bytes(b"")
    progress = ProgressStore(run_dir / "progress.json")
    record = {
        "status": "completed" if completed else "failed",
        "result_path": str(configs / f"{safe}.json"),
    }
    if with_run_id:
        record["mlflow_run_id"] = "abc123"
    progress.data["configs"]["test_config"] = record
    progress.path.write_text(json.dumps(progress.data))
    return run_dir


# --- Aggregate artifact deduplication ---------------------------------------


def test_aggregate_logging_skips_full_sample_payloads(tmp_path, monkeypatch):
    """The aggregate MLflow run must not re-upload per-sample sidecars."""
    from benchmark.tracking import log_aggregate_artifacts_to_mlflow

    run_dir = _make_run_dir(tmp_path)
    logged: list[tuple[str, str]] = []

    class FakeMlflow:
        def active_run(self):
            return None

        def start_run(self, *a, **k):
            return _FakeRun("FINISHED", {}, [])

        def log_artifact(self, path, artifact_path=None):
            logged.append((Path(path).name, artifact_path or ""))

        def log_dict(self, d, artifact_file):
            logged.append((artifact_file.split("/")[-1], "reports"))

        def log_metric(self, *a, **k):
            pass

        def log_artifacts(self, *a, **k):
            pass

    import benchmark.tracking as tracking

    monkeypatch.setattr(tracking, "mlflow", FakeMlflow())
    log_aggregate_artifacts_to_mlflow(run_dir, run_name="agg")

    names = [name for name, _ in logged]
    assert "benchmark_20260101_000000.json" in names
    assert all(not n.endswith(".samples.json.gz") for n in names)


def test_aggregate_logging_links_config_run_ids(tmp_path, monkeypatch):
    """Config MLflow run IDs get linked from the aggregate run."""
    from benchmark.tracking import log_aggregate_artifacts_to_mlflow

    run_dir = _make_run_dir(tmp_path)
    logged_dicts = []

    class FakeMlflow:
        def active_run(self):
            return None

        def start_run(self, *a, **k):
            return _FakeRun("FINISHED", {}, [])

        def log_artifact(self, *a, **k):
            pass

        def log_metric(self, *a, **k):
            pass

        def log_dict(self, d, artifact_file):
            logged_dicts.append((d, artifact_file))

        def log_artifacts(self, *a, **k):
            pass

    import benchmark.tracking as tracking

    monkeypatch.setattr(tracking, "mlflow", FakeMlflow())
    log_aggregate_artifacts_to_mlflow(
        run_dir, run_name="agg", config_run_ids={"test_config": "abc123"}
    )

    links = [d for d, f in logged_dicts if f.endswith("config_run_links.json")]
    assert links and links[0] == {"test_config": "abc123"}


# --- MLflow upload verification ---------------------------------------------


def test_verify_config_run_passes_on_finished_logged_run():
    _FakeClient.configure({"abc123": _FakeRun("FINISHED", {"chunk_size": 512}, ["data/per_sample_results.csv"])})
    ok, reason = verify_config_run("abc123")
    assert ok
    assert reason == ""


@pytest.mark.parametrize(
    "status,params,artifacts",
    [("FAILED", {"a": 1}, ["data/per_sample_results.csv"]), ("FINISHED", {}, ["data/per_sample_results.csv"]), ("FINISHED", {"a": 1}, [])],
)
def test_verify_config_run_rejects_incomplete_uploads(status, params, artifacts):
    _FakeClient.configure({"abc123": _FakeRun(status, params, artifacts)})
    ok, reason = verify_config_run("abc123")
    assert not ok
    assert reason


def test_verify_config_run_handles_missing_run():
    _FakeClient.configure({})
    ok, reason = verify_config_run("does-not-exist")
    assert not ok


# --- Cleanup planning and execution ------------------------------------------


def test_plan_cleanup_selects_verified_completed_configs(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _FakeClient.configure({"abc123": _FakeRun("FINISHED", {"a": 1}, ["data/per_sample_results.csv"])})

    plan = plan_cleanup(run_dir)

    assert [a.config_name for a in plan] == ["test_config"]
    names = {p.name for p in plan[0].paths}
    assert names == {
        "test_config.json",
        "test_config_qa.json",
        "test_config_checkpoint.json",
        "test_config.samples.json.gz",
    }


def test_plan_cleanup_protects_failed_configs(tmp_path):
    run_dir = _make_run_dir(tmp_path, completed=False)
    plan = plan_cleanup(run_dir)
    assert plan == []


def test_plan_cleanup_protects_unverified_uploads(tmp_path):
    """A config whose MLflow run is not FINISHED is never selected."""
    run_dir = _make_run_dir(tmp_path)
    _FakeClient.configure({"abc123": _FakeRun("RUNNING", {"a": 1}, ["data/per_sample_results.csv"])})
    assert plan_cleanup(run_dir) == []


def test_cleanup_run_dir_dry_run_is_default(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _FakeClient.configure({"abc123": _FakeRun("FINISHED", {"a": 1}, ["data/per_sample_results.csv"])})

    removed, actions = cleanup_run_dir(run_dir)  # dry-run by default

    assert removed == []
    assert (run_dir / "configs" / "test_config.json").exists()
    assert len(actions) == 1


def test_cleanup_run_dir_execute_removes_verified_files(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _FakeClient.configure({"abc123": _FakeRun("FINISHED", {"a": 1}, ["data/per_sample_results.csv"])})

    removed, actions = cleanup_run_dir(run_dir, execute=True)

    assert {p.name for p in removed} == {
        "test_config.json",
        "test_config_qa.json",
        "test_config_checkpoint.json",
        "test_config.samples.json.gz",
    }
    assert not (run_dir / "configs" / "test_config.json").exists()
    # Sweep-level artifacts stay untouched.
    assert (run_dir / "benchmark_20260101_000000.json").exists()
    # progress.json stays so the run ledger remains readable.
    assert (run_dir / "progress.json").exists()


def test_cleanup_run_dir_keeps_samples_sidecar_by_default(tmp_path):
    """Full per-sample sidecars are only removed with prune_samples=True."""
    run_dir = _make_run_dir(tmp_path)
    _FakeClient.configure({"abc123": _FakeRun("FINISHED", {"a": 1}, ["data/per_sample_results.csv"])})

    cleanup_run_dir(run_dir, execute=True)
    assert (run_dir / "benchmark_20260101_000000.samples.json.gz").exists()

    cleanup_run_dir(run_dir, execute=True, prune_samples=True)
    assert not (run_dir / "benchmark_20260101_000000.samples.json.gz").exists()


def test_cleanup_tool_class_wraps_module_functions(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _FakeClient.configure({"abc123": _FakeRun("FINISHED", {"a": 1}, ["data/per_sample_results.csv"])})

    tool = CleanupTool(run_dir)
    plan = tool.plan()
    assert len(plan) == 1
    tool.execute()
    assert not (run_dir / "configs" / "test_config.json").exists()


# --- Git policy: generated results are never committed -----------------------


def test_results_dir_is_git_ignored():
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "-q", "results/run999/foo.json"],
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, "results/ is not ignored by .gitignore"


# --- Configurable run output root --------------------------------------------


def test_next_run_dir_respects_env_base(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHMARK_RESULTS_DIR", str(tmp_path / "scratch"))
    run_dir = _next_run_dir()
    assert run_dir.parent.name == "scratch"
    assert run_dir.name == "run1"


# --- Review fixes: cleanup-safe resume and guarded pruning --------------------


def test_resume_after_cleanup_treats_mlflow_backed_record_as_completed(tmp_path):
    """A completed record with an mlflow_run_id stays completed even
    after its local working file was removed by cleanup (#66 review)."""
    run_dir = _make_run_dir(tmp_path)
    progress = ProgressStore(run_dir / "progress.json")
    record = progress.data["configs"]["test_config"]
    Path(record["result_path"]).unlink()
    assert progress.is_completed(_FakeConfig("test_config"))


class _FakeConfig:
    def __init__(self, name):
        self.name = name


def test_resume_without_mlflow_run_id_needs_local_file(tmp_path):
    """Without a recorded MLflow run the local file is still required."""
    run_dir = _make_run_dir(tmp_path, with_run_id=False)
    progress = ProgressStore(run_dir / "progress.json")
    record = progress.data["configs"]["test_config"]
    assert progress.is_completed(_FakeConfig("test_config"))
    Path(record["result_path"]).unlink()
    assert not progress.is_completed(_FakeConfig("test_config"))


def test_prune_samples_skipped_without_progress_ledger(tmp_path):
    """Sidecars must survive when there is no progress.json (#66 review)."""
    run_dir = _make_run_dir(tmp_path)
    _FakeClient.configure(
        {"abc123": _FakeRun("FINISHED", {"a": 1}, ["data/per_sample_results.csv"])}
    )
    (run_dir / "progress.json").unlink()

    removed, _ = cleanup_run_dir(run_dir, execute=True, prune_samples=True)

    # Without the ledger nothing can be verified, so nothing is removed.
    assert removed == []
    assert (run_dir / "benchmark_20260101_000000.samples.json.gz").exists()
    assert (run_dir / "configs" / "test_config.json").exists()


def test_prune_samples_skipped_when_config_unverified(tmp_path):
    """Sidecars must survive when a completed config has no verified
    MLflow per-sample record (#66 review)."""
    run_dir = _make_run_dir(tmp_path)
    # FINISHED run without the canonical per-sample CSV -> unverified.
    _FakeClient.configure({"abc123": _FakeRun("FINISHED", {"a": 1}, [])})

    removed, _ = cleanup_run_dir(run_dir, execute=True, prune_samples=True)

    assert (run_dir / "benchmark_20260101_000000.samples.json.gz").exists()


def test_prune_samples_runs_when_all_completed_configs_verified(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _FakeClient.configure(
        {"abc123": _FakeRun("FINISHED", {"a": 1}, ["data/per_sample_results.csv"])}
    )

    cleanup_run_dir(run_dir, execute=True, prune_samples=True)

    assert not (run_dir / "benchmark_20260101_000000.samples.json.gz").exists()


# --- Aggregate reports must stay loadable without sidecars --------------------


def test_aggregate_report_upload_strips_samples_file(tmp_path, monkeypatch):
    """The uploaded aggregate report must not reference missing sidecars."""
    from benchmark.tracking import log_aggregate_artifacts_to_mlflow

    run_dir = _make_run_dir(tmp_path)
    report = {
        "results": [
            {"config_name": "test_config", "samples_file": "configs/x.samples.json.gz"}
        ]
    }
    (run_dir / "benchmark_20260101_000000.json").write_text(json.dumps(report))
    uploaded: dict = {}

    class FakeMlflow:
        def active_run(self):
            return None

        def start_run(self, *a, **k):
            return _FakeRun("FINISHED", {}, [])

        def log_artifact(self, *a, **k):
            pass

        def log_dict(self, d, artifact_file):
            uploaded[artifact_file] = d

        def log_metric(self, *a, **k):
            pass

        def log_artifacts(self, *a, **k):
            pass

    import benchmark.tracking as tracking

    monkeypatch.setattr(tracking, "mlflow", FakeMlflow())
    log_aggregate_artifacts_to_mlflow(run_dir, run_name="agg")

    payload = uploaded["reports/benchmark_20260101_000000.json"]
    assert "samples_file" not in payload["results"][0]


# --- Canonical per-sample CSV holds contexts ----------------------------------


def test_per_sample_csv_includes_contexts(monkeypatch):
    """The canonical per-sample artifact must contain the retrieved
    contexts so cleanup --prune-samples loses no data (#66 review)."""
    import csv as csv_mod
    from types import SimpleNamespace

    from benchmark.tracking import _log_per_sample_csv

    logged: dict = {}

    class FakeMlflow:
        def log_artifact(self, path, artifact_path=None):
            logged["path"] = Path(path)
            logged["artifact_path"] = artifact_path

    import benchmark.tracking as tracking

    monkeypatch.setattr(tracking, "mlflow", FakeMlflow())

    sample = SimpleNamespace(
        question="q",
        ground_truth="gt",
        answer="a",
        contexts=["doc one", "doc two"],
        ttft_seconds=0.1,
        total_seconds=1.0,
        token_count=10,
        tokens_per_second=10.0,
        input_tokens=5,
        output_tokens=5,
        total_tokens=10,
        estimated_cost_usd=0.0,
        retrieval_metadata={},
        adapter_diagnostics={},
        ragas_scores={},
        custom_scores={},
    )
    result = SimpleNamespace(per_sample=[sample])
    _log_per_sample_csv(result, "run123")

    with open(logged["path"], newline="", encoding="utf-8") as f:
        rows = list(csv_mod.DictReader(f))
    assert logged["artifact_path"] == "data"
    assert json.loads(rows[0]["contexts_json"]) == ["doc one", "doc two"]
