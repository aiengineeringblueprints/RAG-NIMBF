import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmark.reporting.models import BenchmarkResultExtended, PerSampleResult
from benchmark.stage_timing import (
    CANONICAL_STAGES,
    StageTimings,
    stage_timer,
    timed,
)
from benchmark.tracking import _derive_throughput_metrics, log_benchmark_run


def test_canonical_stages_complete():
    assert CANONICAL_STAGES == (
        "chunking",
        "embedding",
        "indexing",
        "retrieval",
        "reranking",
        "generation",
        "evaluation",
        "total",
    )


def test_stage_timer_records_elapsed():
    rec = StageTimings()
    with stage_timer(rec, "chunking"):
        time.sleep(0.05)
    samples = rec.samples["chunking"]
    assert len(samples) == 1
    assert samples[0] >= 0.05
    assert samples[0] < 1.0


def test_stage_timer_recorder_none_is_noop():
    with stage_timer(None, "chunking"):
        time.sleep(0.01)


def test_decorator_records_each_call():
    rec = StageTimings()

    @timed(rec, "generation")
    def gen():
        time.sleep(0.02)
        return 42

    assert gen() == 42
    assert gen() == 42
    samples = rec.samples["generation"]
    assert len(samples) == 2
    assert all(s >= 0.02 for s in samples)


def test_summary_aggregates_percentiles():
    rec = StageTimings()
    for value in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        rec.record("retrieval", value)

    stats = rec.stats("retrieval")
    assert stats is not None
    assert stats.count == 10
    assert stats.total_s == pytest.approx(5.5)
    assert stats.mean_s == pytest.approx(0.55)
    assert stats.p50_s == pytest.approx(0.55)
    assert 0.9 <= stats.p95_s <= 1.0
    assert stats.max_s == pytest.approx(1.0)


def test_summary_missing_stage_returns_none():
    rec = StageTimings()
    assert rec.stats("nonexistent") is None
    assert rec.summary() == {}


def test_to_dict_serializes_for_json():
    rec = StageTimings()
    rec.record("chunking", 1.0)
    rec.record("retrieval", 0.1)
    rec.record("retrieval", 0.2)
    rec.set_count("chunks", 42)
    rec.set_count("questions", 2)

    payload = rec.to_dict()
    serialized = json.dumps(payload)
    restored = json.loads(serialized)

    assert "stages" in restored
    assert restored["stages"]["chunking"]["count"] == 1
    assert restored["stages"]["chunking"]["total_s"] == 1.0
    assert restored["stages"]["retrieval"]["count"] == 2
    assert restored["counts"]["chunks"] == 42
    assert "throughput" in restored


def test_throughput_derivation():
    rec = StageTimings()
    rec.record("indexing", 4.0)
    rec.set_count("chunks", 100)
    rec.record("retrieval", 0.5)
    rec.record("retrieval", 0.5)

    throughput = rec.throughput()
    assert throughput["embed_index_chunks_s"] == pytest.approx(25.0)
    assert throughput["retrieve_qps"] == pytest.approx(2.0)


def test_mlflow_metrics_keys_have_lat_prefix():
    rec = StageTimings()
    rec.record("chunking", 1.0)
    rec.record("indexing", 4.0)
    rec.record("generation", 0.5)
    rec.set_count("chunks", 10)

    metrics = rec.mlflow_metrics()
    assert "lat_chunking_total_s" in metrics
    assert "lat_chunking_mean_s" in metrics
    assert "lat_generation_p95_s" in metrics
    assert "lat_generation_count" in metrics
    assert "throughput_embed_index_chunks_s" in metrics


def test_save_json_writes_file(tmp_path: Path):
    rec = StageTimings()
    rec.record("chunking", 1.0)
    out = rec.save_json(tmp_path / "stage_timings" / "cfg1.json")
    assert out.exists()
    payload = json.loads(out.read_text())
    assert "stages" in payload
    assert payload["stages"]["chunking"]["total_s"] == 1.0


def _make_result(stage_latency=None, num_chunks=0) -> BenchmarkResultExtended:
    return BenchmarkResultExtended(
        config_name="test",
        llm_model="m",
        embedding_model="e",
        prompt_template="t",
        chunking_strategy="recursive",
        chunk_size=100,
        chunk_overlap=20,
        num_chunks=num_chunks,
        num_questions=2,
        avg_ttft_seconds=0.0,
        avg_tokens_per_second=0.0,
        avg_gpu_utilization_pct=None,
        avg_gpu_memory_used_mb=None,
        ragas_faithfulness=None,
        ragas_answer_relevancy=None,
        ragas_answer_correctness=None,
        ragas_context_precision=None,
        ragas_context_recall=None,
        ragas_semantic_similarity=None,
        total_time_seconds=1.0,
        per_sample=(
            PerSampleResult(
                question="q",
                ground_truth="gt",
                answer="a",
                contexts=(),
                ttft_seconds=0.0,
                total_seconds=0.0,
                token_count=0,
                tokens_per_second=0.0,
                gpu_usage=None,
                ragas_scores={},
            ),
        ),
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
        stage_latency=stage_latency,
    )


def test_derive_throughput_metrics_uses_num_chunks():
    result = _make_result(
        stage_latency={"indexing": {"total_s": 5.0, "count": 1}},
        num_chunks=50,
    )
    tp = _derive_throughput_metrics(result)
    assert tp["embed_index_chunks_s"] == pytest.approx(10.0)


def test_log_benchmark_run_emits_lat_and_throughput_metrics(monkeypatch):
    captured: dict[str, float] = {}

    class _DummyInfo:
        run_id = "test-run"

    class _DummyRun:
        info = _DummyInfo()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _no_op(*args, **kwargs):
        pass

    monkeypatch.setattr("benchmark.tracking.mlflow.set_experiment", _no_op)
    monkeypatch.setattr("benchmark.tracking.mlflow.active_run", lambda: None)
    monkeypatch.setattr("benchmark.tracking.mlflow.start_run", lambda **kw: _DummyRun())
    monkeypatch.setattr("benchmark.tracking.mlflow.log_params", _no_op)
    monkeypatch.setattr("benchmark.tracking.mlflow.log_metrics", lambda m: captured.update(m))
    monkeypatch.setattr("benchmark.tracking.mlflow.log_artifact", _no_op)
    monkeypatch.setattr("benchmark.tracking.mlflow.log_artifacts", _no_op)
    monkeypatch.setattr("benchmark.tracking.mlflow.set_tag", _no_op)
    monkeypatch.setattr(
        "benchmark.tracking._log_per_sample_csv", _no_op
    )
    monkeypatch.setattr(
        "benchmark.tracking._log_classic_retriever_metrics", _no_op
    )
    monkeypatch.setattr(
        "benchmark.tracking._log_genai_rag_judges", _no_op
    )

    stage_latency = {
        "chunking": {"count": 1, "total_s": 2.0, "mean_s": 2.0, "p50_s": 2.0, "p95_s": 2.0, "max_s": 2.0},
        "retrieval": {"count": 2, "total_s": 0.4, "mean_s": 0.2, "p50_s": 0.2, "p95_s": 0.2, "max_s": 0.2},
        "indexing": {"count": 1, "total_s": 4.0, "mean_s": 4.0, "p50_s": 4.0, "p95_s": 4.0, "max_s": 4.0},
    }
    result = _make_result(stage_latency=stage_latency, num_chunks=20)

    log_benchmark_run(result, nested=False)

    assert "lat_chunking_total_s" in captured
    assert captured["lat_chunking_total_s"] == pytest.approx(2.0)
    assert "lat_retrieval_p95_s" in captured
    assert "lat_retrieval_count" in captured
    assert captured["lat_retrieval_count"] == pytest.approx(2.0)
    assert "throughput_embed_index_chunks_s" in captured
    assert captured["throughput_embed_index_chunks_s"] == pytest.approx(5.0)
    assert "throughput_retrieve_qps" in captured
    assert captured["throughput_retrieve_qps"] == pytest.approx(5.0)
