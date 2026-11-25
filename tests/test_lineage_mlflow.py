"""Lineage reaches MLflow as searchable tags and self-contained artifacts (#65)."""

from __future__ import annotations

import json

import mlflow
import pytest
from test_mlflow_tracing import tracking_store  # noqa: F401 - pytest fixture
from test_run_path_uniformity import (  # noqa: F401 - pytest fixture
    CORPUS,
    SAMPLES,
    UniformityConfig,
    stubbed_seam,
)

from benchmark.tracking import log_benchmark_run

main = pytest.importorskip("main")

EXPERIMENT = {
    "study_id": "retrieval-k",
    "candidate_id": "k8",
    "baseline_run_id": "base-run",
    "issue_reference": "#65",
}
DECLARED = {"system_id": "acme-rag", "version_source": "container", "version": "sha256:feed"}


@pytest.fixture
def logged_run(tracking_store, stubbed_seam):  # noqa: F811 - fixture names
    config = UniformityConfig(
        rag_system_adapter="stub-external",
        extra={
            "experiment_metadata_json": json.dumps(EXPERIMENT),
            "system_provenance_json": json.dumps(DECLARED),
        },
    )
    result = main.run_single_benchmark(config, SAMPLES, corpus=CORPUS)
    return result, log_benchmark_run(result, nested=False)


def test_lineage_becomes_searchable_tags(logged_run):
    result, run_id = logged_run

    tags = mlflow.get_run(run_id).data.tags

    assert tags["experiment.study_id"] == "retrieval-k"
    assert tags["experiment.candidate_id"] == "k8"
    assert tags["system.id"] == "acme-rag"
    assert tags["system.status"] == "user_supplied"
    assert tags["system.version"] == "sha256:feed"
    assert tags["dataset.question_set_hash"] == result.lineage["dataset"]["question_set_hash"]
    assert tags["dataset.corpus_fingerprint"] == result.lineage["dataset"]["corpus_fingerprint"]
    assert len(tags["scorer.fingerprint"]) == 64


def test_no_question_text_in_tags(logged_run):
    _, run_id = logged_run

    tags = mlflow.get_run(run_id).data.tags

    for sample in SAMPLES:
        assert not any(sample["question"] in value for value in tags.values())


def test_lineage_and_sample_metrics_are_logged_as_artifacts(logged_run):
    result, run_id = logged_run

    lineage = mlflow.artifacts.load_dict(f"runs:/{run_id}/lineage/lineage.json")
    samples = mlflow.artifacts.load_dict(f"runs:/{run_id}/lineage/samples.json")

    assert lineage["experiment"] == EXPERIMENT
    assert lineage["system"]["system_id"] == "acme-rag"
    assert [s["sample_id"] for s in samples["samples"]] == [
        s.sample_id for s in result.per_sample
    ]
    assert "total_seconds" in samples["samples"][0]["metrics"]
    assert "question" not in samples["samples"][0]


def test_runs_compare_by_mlflow_run_id(logged_run):
    from benchmark.lineage.compare import compare_runs
    from benchmark.lineage.runs import load_run_record

    _, run_id = logged_run

    record = load_run_record(run_id)
    result = compare_runs(record, record)

    assert record.lineage["system"]["system_id"] == "acme-rag"
    assert result["status"] == "paired"
    assert result["pairing"]["paired"] == len(SAMPLES)
