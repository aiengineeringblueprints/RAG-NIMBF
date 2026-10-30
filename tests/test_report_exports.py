"""Benchmark JSON reports: compact summary + gzipped per-sample sidecar."""

import json

from benchmark.reporting.exports import (
    compact_json_report,
    load_json_report,
    write_json_report,
)
from benchmark.reporting.models import (
    BenchmarkResultExtended,
    BenchmarkRun,
    PerSampleResult,
)

SHARED_CONTEXT = "Paris is the capital of France. " * 300  # ~10 KB


def _sample(i: int, contexts: tuple[str, ...]) -> PerSampleResult:
    return PerSampleResult(
        question=f"q{i}",
        ground_truth="Paris",
        answer=f"answer {i}",
        contexts=contexts,
        ttft_seconds=0.1,
        total_seconds=1.0,
        token_count=5,
        tokens_per_second=5.0,
        gpu_usage=None,
        ragas_scores={"faithfulness": 0.5},
        retrieval_metadata=({"source": "wiki"},),
    )


def _result(name: str, samples: tuple[PerSampleResult, ...]) -> BenchmarkResultExtended:
    return BenchmarkResultExtended(
        config_name=name,
        llm_model="model",
        embedding_model="emb",
        prompt_template="concise",
        chunking_strategy="recursive",
        chunk_size=512,
        chunk_overlap=64,
        num_chunks=10,
        num_questions=len(samples),
        avg_ttft_seconds=0.1,
        avg_tokens_per_second=5.0,
        avg_gpu_utilization_pct=None,
        avg_gpu_memory_used_mb=None,
        ragas_faithfulness=0.5,
        ragas_answer_relevancy=None,
        ragas_answer_correctness=None,
        ragas_context_precision=None,
        ragas_context_recall=None,
        ragas_semantic_similarity=None,
        total_time_seconds=1.0,
        per_sample=samples,
        ttft_stats=None,
        tps_stats=None,
        gpu_util_stats=None,
        gpu_mem_stats=None,
        ragas_faithfulness_stats=None,
        ragas_answer_relevancy_stats=None,
        ragas_answer_correctness_stats=None,
        ragas_context_precision_stats=None,
        ragas_context_recall_stats=None,
        ragas_semantic_similarity_stats=None,
    )


def _run(*results: BenchmarkResultExtended) -> BenchmarkRun:
    return BenchmarkRun(
        timestamp="20261030_120000",
        dataset_name="ragbench",
        dataset_subset="techqa",
        dataset_sample_size=50,
        system_info={},
        results=results,
    )


def test_report_roundtrips_per_sample_through_sidecar(tmp_path):
    run = _run(
        _result("a", (_sample(0, ("ctx one", "ctx two")),)),
        _result("b", (_sample(1, ("ctx two",)),)),
    )
    path = write_json_report(run, tmp_path / "benchmark_x.json")

    summary = json.loads(path.read_text())
    assert all("per_sample" not in r for r in summary["results"])
    assert summary["results"][0]["config_name"] == "a"

    report = load_json_report(path)
    first, second = (r["per_sample"] for r in report["results"])
    assert first[0]["contexts"] == ["ctx one", "ctx two"]
    assert first[0]["answer"] == "answer 0"
    assert first[0]["retrieval_metadata"] == [{"source": "wiki"}]
    assert second[0]["contexts"] == ["ctx two"]
    assert second[0]["ragas_scores"] == {"faithfulness": 0.5}


def test_repeated_contexts_keep_report_small(tmp_path):
    samples = tuple(_sample(i, (SHARED_CONTEXT,)) for i in range(50))
    run = _run(_result("a", samples), _result("b", samples))

    path = write_json_report(run, tmp_path / "benchmark_x.json")

    # 100 samples x 10 KB context inline would be ~1 MB.
    total = sum(p.stat().st_size for p in tmp_path.iterdir())
    assert total < 30_000


def test_legacy_inline_report_loads_unchanged(tmp_path):
    legacy = {"timestamp": "t", "results": [{"config_name": "a", "per_sample": [
        {"question": "q", "contexts": ["c"]}]}]}
    path = tmp_path / "benchmark_old.json"
    path.write_text(json.dumps(legacy))

    assert load_json_report(path) == legacy


def test_compact_converts_legacy_report_losslessly(tmp_path):
    legacy = {"timestamp": "t", "num_configs": 2, "results": [
        {"config_name": "a", "stats": {}, "per_sample": [
            {"question": "q0", "contexts": [SHARED_CONTEXT, "x"], "answer": "a0"}]},
        {"config_name": "b", "stats": {}, "per_sample": [
            {"question": "q1", "contexts": [SHARED_CONTEXT], "answer": "a1"}]},
    ]}
    path = tmp_path / "benchmark_old.json"
    path.write_text(json.dumps(legacy, indent=2))

    assert compact_json_report(path) is True

    assert load_json_report(path) == legacy
    assert "per_sample" not in path.read_text()
    assert compact_json_report(path) is False  # already compact


def test_compact_keeps_results_without_samples_unchanged(tmp_path):
    legacy = {"results": [{"config_name": "failed"}, {"config_name": "ok", "per_sample": []}]}
    path = tmp_path / "benchmark_old.json"
    path.write_text(json.dumps(legacy))

    compact_json_report(path)

    assert load_json_report(path) == legacy
