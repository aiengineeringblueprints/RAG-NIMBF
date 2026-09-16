"""Parser evaluation runner through the worker (OCR-06).

Worker-level tests with a stub parser and a tiny fixture GT dataset; the
MLflow tracker is faked and run directories are temp dirs.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

import pytest

from benchmark.orchestration.matrix import (
    ExperimentSpec,
    build_configs_from_spec,
)
from benchmark.orchestration.worker import ExperimentWorker, WorkerOptions
from benchmark.parsing import (
    ParsedPage,
    ParseResult,
    register_parser_adapter,
)

# ── Tiny OmniDocBench-style GT fixture ───────────────────────────────


def _write_gt_fixture(tmp_path) -> str:
    pages = [
        {
            "page_info": {"image_path": "doc_a_p1.png", "page_number": 1},
            "layout_dets": [
                {"category_type": "text", "text": "The Eiffel Tower is in Paris."},
                {
                    "category_type": "table",
                    "html": "<html><body><table><tr><td>Name</td><td>City</td></tr>"
                    "<tr><td>Eiffel</td><td>Paris</td></tr></table></body></html>",
                },
            ],
        },
        {
            "page_info": {"image_path": "doc_a_p2.png", "page_number": 2},
            "layout_dets": [
                {"category_type": "text", "text": "Slide deck about retrieval."},
            ],
        },
        {
            "page_info": {"image_path": "doc_b_p1.png", "page_number": 1},
            "layout_dets": [
                {"category_type": "title", "text": "Quarterly Report"},
            ],
        },
    ]
    path = tmp_path / "gt_fixture.json"
    path.write_text(json.dumps(pages), encoding="utf-8")
    return str(path)


# ── Stub parsers producing two quality levels ────────────────────────


class _StubParser:
    def __init__(self, quality: str) -> None:
        self._quality = quality
        self.parse_calls: list[str] = []

    @property
    def name(self) -> str:
        return f"stub-{self._quality}"

    @property
    def parser_version(self) -> str | None:
        return "0.1.0"

    def parse(self, document: dict, config: Any = None) -> ParseResult:
        self.parse_calls.append(document["document_id"])
        page_number = int(
            (document.get("metadata") or {}).get("page_number", 1)
        )
        if self._quality == "good":
            markdown = "The Eiffel Tower is in Paris."
        else:
            markdown = "Completely wrong text."
        return ParseResult(
            document_id=document["document_id"],
            pages=(ParsedPage(page_number=page_number, markdown=markdown),),
            parser_name=self.name,
            parser_version=self.parser_version,
            total_seconds=0.01,
        )


def _stub_factory(config: Any) -> _StubParser:
    quality = "bad" if "bad" in str(config.parser_adapter) else "good"
    return _StubParser(quality)


register_parser_adapter("stub-parser-good", _stub_factory)
register_parser_adapter("stub-parser-bad", _stub_factory)


def _parser_configs(gt_path: str) -> list:
    spec = ExperimentSpec(
        name="parser-leaderboard",
        dataset={"name": "omnidocbench", "path": gt_path},
        settings={
            "benchmark_stage": "parsing",
            "ragas_enabled": False,
            "custom_metrics_enabled": False,
        },
        matrix={"parser_adapters": ["stub-parser-good", "stub-parser-bad"]},
    )
    return build_configs_from_spec(spec)


# ── Fakes ────────────────────────────────────────────────────────────


@pytest.fixture
def fake_tracking(monkeypatch):
    import mlflow

    import benchmark.orchestration.worker as worker_module

    calls: dict[str, list] = {"parsing": [], "runs": [], "aggregate": []}

    def fake_log_parsing_run(summary, reproducibility_dir=None, nested=None):
        calls["parsing"].append(summary["config_name"])

    def fake_log_run(result, reproducibility_dir=None, nested=None):
        calls["runs"].append(result.config_name)

    def fake_log_aggregate(run_dir, run_name=None, reproducibility_dir=None):
        calls["aggregate"].append(str(run_dir))

    @contextmanager
    def fake_start_run(*args, **kwargs):
        yield None

    monkeypatch.setattr(worker_module, "log_benchmark_run", fake_log_run)
    monkeypatch.setattr(
        worker_module, "log_parsing_run", fake_log_parsing_run
    )
    monkeypatch.setattr(
        worker_module, "log_aggregate_artifacts_to_mlflow", fake_log_aggregate
    )
    monkeypatch.setattr(mlflow, "start_run", fake_start_run)
    return calls


# ── Tests: manifest → config matrix ──────────────────────────────────


def test_matrix_expands_parser_adapters(tmp_path):
    gt_path = _write_gt_fixture(tmp_path)
    configs = _parser_configs(gt_path)

    assert len(configs) == 2
    assert [c.parser_adapter for c in configs] == [
        "stub-parser-good",
        "stub-parser-bad",
    ]
    assert all(c.benchmark_stage == "parsing" for c in configs)
    # Cells must not collide in config names (resume keying).
    assert len({c.name for c in configs}) == 2
    assert all(c.dataset_license for c in configs)


def test_matrix_rejects_parsing_stage_without_parser(tmp_path):
    spec = ExperimentSpec(
        name="parser-invalid",
        dataset={"name": "omnidocbench", "path": _write_gt_fixture(tmp_path)},
        settings={"benchmark_stage": "parsing"},
        matrix={},
    )
    with pytest.raises(ValueError, match="parser"):
        build_configs_from_spec(spec)


# ── Tests: worker-driven parsing run ─────────────────────────────────


def _run_worker(configs, run_dir):
    worker = ExperimentWorker(
        configs, WorkerOptions(run_dir=run_dir, experiment_name="parser-run")
    )
    return worker.run()


def test_worker_parsing_run_produces_leaderboard_and_checkpoints(
    tmp_path, fake_tracking
):
    gt_path = _write_gt_fixture(tmp_path)
    configs = _parser_configs(gt_path)
    run_dir = tmp_path / "run1"

    results = _run_worker(configs, run_dir)

    assert len(results) == 2

    # Leaderboard report in the run dir.
    leaderboards = list(run_dir.glob("parsing_leaderboard_*.json"))
    assert len(leaderboards) == 1
    board = json.loads(leaderboards[0].read_text(encoding="utf-8"))

    # Per-parser rows with text metrics, TEDS, and overall score.
    by_parser = {row["parser"]: row for row in board["parsers"]}
    assert set(by_parser) == {"stub-good", "stub-bad"}
    good = by_parser["stub-good"]
    assert good["parser_version"] == "0.1.0"
    for metric in ("cer", "wer", "normalized_edit_distance", "teds", "overall"):
        assert metric in good["metrics"], metric
    assert good["metrics"]["normalized_edit_distance"] < by_parser["stub-bad"][
        "metrics"
    ]["normalized_edit_distance"]
    assert good["metrics"]["overall"] > by_parser["stub-bad"]["metrics"]["overall"]

    # Per-category breakdown present.
    assert board["per_category"]
    category = board["per_category"]["default"]
    assert "stub-good" in category

    # Run metadata: matching algorithm version + dataset license flags.
    assert board["match_algorithm"]["version"]
    assert board["dataset"]["license"]
    assert "research_only" in board["dataset"]

    # Per-page details in each cell's directory.
    for config in configs:
        safe_name = config.name.replace(":", "_").replace("/", "_")
        cell_dir = run_dir / "configs" / f"{safe_name}_parsing"
        details = json.loads(
            (cell_dir / "page_details.json").read_text(encoding="utf-8")
        )
        assert len(details["documents"]) == 3
        scored_doc = next(
            d for d in details["documents"]
            if d["document_id"].startswith("doc_a_p1")
        )
        assert scored_doc["per_page"], "per-page scores must be present"
        page = scored_doc["per_page"][0]
        assert "cer" in page and "teds" in page

    # Per-document checkpoints written next to the page details.
    safe_name = configs[0].name.replace(":", "_").replace("/", "_")
    cell_dir = run_dir / "configs" / f"{safe_name}_parsing"
    checkpoints = sorted(cell_dir.glob("doc_*.json"))
    assert len(checkpoints) == 3

    # MLflow child runs for each cell, none of the RAG runner's.
    assert sorted(fake_tracking["parsing"]) == sorted(c.name for c in configs)
    assert fake_tracking["runs"] == []


def test_worker_parsing_resume_skips_parsed_documents(tmp_path, fake_tracking):
    gt_path = _write_gt_fixture(tmp_path)
    configs = _parser_configs(gt_path)
    run_dir = tmp_path / "run1"

    _run_worker(configs, run_dir)

    safe_name = configs[0].name.replace(":", "_").replace("/", "_")
    cell_dir = run_dir / "configs" / f"{safe_name}_parsing"
    mtimes = {p.name: p.stat().st_mtime_ns for p in cell_dir.glob("doc_*.json")}

    _run_worker(configs, run_dir)

    # The second run served everything from the per-document checkpoints:
    # no checkpoint file was rewritten.
    assert {
        p.name: p.stat().st_mtime_ns for p in cell_dir.glob("doc_*.json")
    } == mtimes


def test_worker_parsing_resume_continues_after_partial_run(
    tmp_path, fake_tracking
):
    gt_path = _write_gt_fixture(tmp_path)
    configs = _parser_configs(gt_path)[:1]
    run_dir = tmp_path / "run1"

    # Simulate a crashed first attempt: one document checkpoint exists,
    # the cell is not marked completed in the worker progress ledger.
    config = configs[0]
    safe_name = config.name.replace(":", "_").replace("/", "_")
    cell_dir = run_dir / "configs" / f"{safe_name}_parsing"
    cell_dir.mkdir(parents=True)
    (cell_dir / "doc_a_p1.png.json").write_text(
        json.dumps({"document_id": "doc_a_p1.png", "status": "completed"}),
        encoding="utf-8",
    )

    results = _run_worker(configs, run_dir)

    assert len(results) == 1
    records = {
        json.loads(p.read_text(encoding="utf-8"))["document_id"]
        for p in cell_dir.glob("doc_*")
    }
    assert records == {"doc_a_p1.png", "doc_a_p2.png", "doc_b_p1.png"}
    progress = json.loads(
        (run_dir / "progress.json").read_text(encoding="utf-8")
    )
    assert progress["configs"][config.name]["status"] == "completed"
