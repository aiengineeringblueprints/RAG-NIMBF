"""Tests for gold-document retrieval metrics."""

from benchmark.gold_retrieval_metrics import compute_gold_doc_retrieval_metrics


def test_gold_doc_metrics_hit_positions():
    result = compute_gold_doc_retrieval_metrics(
        gold_doc_ids=["doc-b", "doc-z"],
        retrieved_metadata=[
            [{"doc_id": "doc-a"}, {"doc_id": "doc-b"}],
            [{"doc_id": "doc-x"}, {"doc_id": "doc-y"}],
        ],
        k_values=[1, 3],
    )

    assert result.per_sample[0]["hit@1"] == 0.0
    assert result.per_sample[0]["hit@3"] == 1.0
    assert result.per_sample[0]["recall@3"] == 1.0
    assert result.per_sample[0]["ndcg@3"] > 0.0
    assert result.per_sample[1]["hit@3"] == 0.0
    assert result.metric_means["hit@3"] == 0.5


def test_gold_doc_metrics_skip_missing_gold_id():
    result = compute_gold_doc_retrieval_metrics(
        gold_doc_ids=[None],
        retrieved_metadata=[[{"doc_id": "doc-a"}]],
        k_values=[1],
    )

    assert result.skipped_samples == 1
    assert result.per_sample[0]["hit@1"] is None
    assert result.metric_means == {}
