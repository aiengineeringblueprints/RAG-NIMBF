import pytest

from benchmark.reporting.models import BenchmarkResultExtended, PerSampleResult
from benchmark.tracking import (
    _classic_retriever_metric_means,
    _log_classic_retriever_metrics,
)


def test_classic_retriever_metric_means():
    rows = [
        {
            "query": "q1",
            "retrieved_doc_ids": ["a", "b", "c"],
            "ground_truth_doc_ids": ["a", "x"],
        },
        {
            "query": "q2",
            "retrieved_doc_ids": ["z", "y", "x"],
            "ground_truth_doc_ids": ["x"],
        },
    ]

    metrics = _classic_retriever_metric_means(rows, 3)

    assert metrics["classic_precision_at_3"] == pytest.approx(1 / 3)
    assert metrics["classic_recall_at_3"] == pytest.approx(0.75)
    assert metrics["classic_ndcg_at_3"] == pytest.approx(
        ((1 / (1 + 1 / 1.5849625007211563)) + 0.5) / 2
    )


def test_log_classic_retriever_metrics_does_not_use_deprecated_evaluate(monkeypatch):
    logged_metrics = {}
    tags = {}

    monkeypatch.setattr(
        "benchmark.tracking.mlflow.evaluate",
        lambda *args, **kwargs: pytest.fail("deprecated mlflow.evaluate was called"),
    )
    monkeypatch.setattr(
        "benchmark.tracking.mlflow.log_metrics",
        lambda metrics: logged_metrics.update(metrics),
    )
    monkeypatch.setattr(
        "benchmark.tracking.mlflow.set_tag",
        lambda key, value: tags.update({key: value}),
    )

    result = BenchmarkResultExtended(
        config_name="test",
        llm_model="model",
        embedding_model="emb",
        prompt_template="concise",
        chunking_strategy="recursive",
        chunk_size=1000,
        chunk_overlap=200,
        num_chunks=10,
        num_questions=1,
        avg_ttft_seconds=0.5,
        avg_tokens_per_second=100.0,
        avg_gpu_utilization_pct=None,
        avg_gpu_memory_used_mb=None,
        ragas_faithfulness=None,
        ragas_answer_relevancy=None,
        ragas_answer_correctness=None,
        ragas_context_precision=None,
        ragas_context_recall=None,
        ragas_semantic_similarity=None,
        total_time_seconds=30.0,
        per_sample=(
            PerSampleResult(
                question="q1",
                ground_truth="gt",
                answer="answer",
                contexts=(),
                ttft_seconds=0.1,
                total_seconds=1.0,
                token_count=10,
                tokens_per_second=10.0,
                gpu_usage=None,
                ragas_scores={},
                retrieved_doc_ids=("doc-1", "doc-2"),
                ground_truth_doc_ids=("doc-1",),
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
        retrieval_top_k=2,
    )

    _log_classic_retriever_metrics(result)

    assert tags["mlflow_classic_retriever_metrics"] == "logged"
    assert logged_metrics == {
        "classic_precision_at_2": 0.5,
        "classic_recall_at_2": 1.0,
        "classic_ndcg_at_2": 1.0,
    }
