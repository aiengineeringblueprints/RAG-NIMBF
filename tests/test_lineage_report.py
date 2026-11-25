"""Lineage flows from a run into its report and back (issue #65)."""

from __future__ import annotations

import json

import pytest
from test_run_path_uniformity import (  # noqa: F401 - fixture import
    CORPUS,
    SAMPLES,
    UniformityConfig,
    stubbed_seam,
)

from benchmark.lineage.dataset import assign_sample_ids, corpus_fingerprint
from benchmark.lineage.runs import load_run_record
from benchmark.reporting.exports import load_json_report, write_json_report
from benchmark.reporting.models import BenchmarkRun

main = pytest.importorskip("main")

EXPERIMENT = {
    "study_id": "retrieval-k",
    "hypothesis": "top-k 8 raises recall",
    "candidate_id": "k8",
    "baseline_run_id": "abc123",
    "change_under_test": "retrieval_top_k",
    "change_summary": "4 -> 8",
    "issue_reference": "#65",
}


def _run(rag_system_adapter="internal", **extra):
    config = UniformityConfig(rag_system_adapter=rag_system_adapter, extra=extra)
    return main.run_single_benchmark(config, SAMPLES, corpus=CORPUS)


def test_run_result_carries_experiment_dataset_and_scorer_lineage(stubbed_seam):
    result = _run(experiment_metadata_json=json.dumps(EXPERIMENT), eval_critic_llm="judge-1")

    assert result.lineage["experiment"] == EXPERIMENT
    assert result.lineage["dataset"]["question_set_hash"]
    assert result.lineage["dataset"]["num_samples"] == len(SAMPLES)
    assert result.lineage["dataset"]["corpus_fingerprint"] == corpus_fingerprint(CORPUS)
    assert result.lineage["scorer"]["eval_critic_llm"] == "judge-1"
    assert [s.sample_id for s in result.per_sample] == assign_sample_ids(SAMPLES)


def test_internal_run_records_framework_as_system(stubbed_seam):
    result = _run()

    assert result.lineage["system"]["system_id"] == "benchmark_internal_pipeline"


def test_external_adapter_without_capture_uses_manifest_provenance(stubbed_seam):
    declared = {"system_id": "acme-rag", "version_source": "container", "version": "sha256:feed"}

    result = _run(rag_system_adapter="stub-external", system_provenance_json=json.dumps(declared))

    system = result.lineage["system"]
    assert system["system_id"] == "acme-rag"
    assert system["status"] == "user_supplied"
    assert system["version"] == "sha256:feed"


def test_external_adapter_without_any_provenance_is_not_available(stubbed_seam):
    result = _run(rag_system_adapter="stub-external")

    assert result.lineage["system"]["status"] == "not_available"
    assert result.lineage["experiment"] == {}


def test_lineage_and_sample_ids_survive_the_json_report(stubbed_seam, tmp_path):
    result = _run(experiment_metadata_json=json.dumps(EXPERIMENT))
    run = BenchmarkRun(
        timestamp="t", dataset_name="unit-test", dataset_subset="",
        dataset_sample_size=0, system_info={}, results=(result,),
    )

    path = write_json_report(run, tmp_path / "report.json")
    loaded = load_json_report(path)["results"][0]

    assert loaded["lineage"]["experiment"] == EXPERIMENT
    assert loaded["lineage"]["dataset"] == result.lineage["dataset"]
    assert loaded["per_sample"][0]["sample_id"] == result.per_sample[0].sample_id
    assert "category" in loaded["per_sample"][0]

    record = load_run_record(str(path))
    assert record.samples[0]["question_id"]
    assert record.samples[0]["answer_valid"] is True
    assert record.run_errors == {"evaluation_error": result.evaluation_error}


def test_reproducibility_manifest_records_experiment_metadata():
    from types import SimpleNamespace

    from benchmark.reproducibility import build_reproducibility_manifest

    configs = [
        SimpleNamespace(experiment_metadata_json='{"study_id": "s1", "candidate_id": "a", "baseline_run_id": "b"}'),
        SimpleNamespace(experiment_metadata_json='{"study_id": "s1", "candidate_id": "a", "baseline_run_id": "b"}'),
        SimpleNamespace(experiment_metadata_json=None),
    ]

    manifest = build_reproducibility_manifest(configs)

    assert manifest["experiments"] == [
        {"study_id": "s1", "candidate_id": "a", "baseline_run_id": "b"}
    ]


def test_run_dir_reproducibility_bundle_holds_lineage(tmp_path, stubbed_seam):
    config = UniformityConfig(rag_system_adapter="stub-external")

    main.run_single_benchmark(config, SAMPLES, run_dir=tmp_path, corpus=CORPUS)

    (record,) = (tmp_path / "reproducibility" / "lineage").glob("*.json")
    payload = json.loads(record.read_text())
    assert payload["dataset"]["num_samples"] == len(SAMPLES)
    assert len(payload["samples"]) == len(SAMPLES)
    assert "framework" in payload
