"""Tests for benchmark.reporting.models — data models and stats."""

import math
import pytest

from benchmark.reporting.models import (
    StatSummary,
    compute_stats,
    PerSampleResult,
    BenchmarkResultExtended,
)


class TestComputeStats:
    def test_basic_values(self):
        result = compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert isinstance(result, StatSummary)
        assert result.mean == pytest.approx(3.0)
        assert result.min == 1.0
        assert result.max == 5.0
        assert result.median == pytest.approx(3.0)
        assert result.count == 5

    def test_std_dev(self):
        result = compute_stats([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert result.std == pytest.approx(2.0, rel=0.1)

    def test_single_value_std_is_zero(self):
        result = compute_stats([42.0])
        assert result.std == 0.0

    def test_empty_list_returns_none(self):
        assert compute_stats([]) is None

    def test_nan_values_filtered(self):
        result = compute_stats([1.0, float("nan"), 3.0])
        assert result is not None
        assert result.count == 2
        assert result.mean == pytest.approx(2.0)

    def test_all_nan_returns_none(self):
        assert compute_stats([float("nan"), float("nan")]) is None


class TestPerSampleResult:
    def test_creation(self):
        r = PerSampleResult(
            question="What?",
            ground_truth="Answer",
            answer="My answer",
            contexts=("ctx1", "ctx2"),
            ttft_seconds=0.5,
            total_seconds=1.0,
            token_count=10,
            tokens_per_second=10.0,
            gpu_usage={"gpu_utilization_pct": 50.0},
            ragas_scores={"faithfulness": 0.9},
        )
        assert r.question == "What?"
        assert r.contexts == ("ctx1", "ctx2")
        assert r.ragas_scores["faithfulness"] == 0.9

    def test_frozen(self):
        r = PerSampleResult(
            question="q", ground_truth="g", answer="a",
            contexts=(), ttft_seconds=0, total_seconds=0,
            token_count=0, tokens_per_second=0,
            gpu_usage=None, ragas_scores={},
        )
        with pytest.raises(AttributeError):
            r.question = "new"


class TestBenchmarkResultExtended:
    def test_defaults(self):
        r = BenchmarkResultExtended(
            config_name="test",
            llm_model="model",
            embedding_model="emb",
            prompt_template="concise",
            chunking_strategy="recursive",
            chunk_size=1000,
            chunk_overlap=200,
            num_chunks=10,
            num_questions=5,
            avg_ttft_seconds=0.5,
            avg_tokens_per_second=100.0,
            avg_gpu_utilization_pct=None,
            avg_gpu_memory_used_mb=None,
            ragas_faithfulness=None,
            ragas_answer_relevancy=None,
            ragas_answer_correctness=None,
            ragas_context_precision=None,
            ragas_context_recall=None,
            total_time_seconds=30.0,
            per_sample=(),
            ttft_stats=None,
            tps_stats=None,
            gpu_util_stats=None,
            gpu_mem_stats=None,
            ragas_faithfulness_stats=None,
            ragas_answer_relevancy_stats=None,
            ragas_answer_correctness_stats=None,
            ragas_context_precision_stats=None,
            ragas_context_recall_stats=None,
        )
        assert r.evaluation_error is None
        assert r.ragas_valid_sample_counts is None
        assert r.per_sample == ()
