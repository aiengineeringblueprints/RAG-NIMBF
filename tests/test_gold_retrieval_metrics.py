"""Tests for gold-document retrieval metrics."""

import math

import pytest

from benchmark.gold_retrieval_metrics import (
    _ndcg_at_k,
    compute_gold_doc_retrieval_metrics,
)


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


def test_span_metrics_distinguish_wrong_section_in_right_document():
    result = compute_gold_doc_retrieval_metrics(
        gold_doc_ids=[["doc-a", "doc-b"]],
        retrieved_metadata=[
            [
                {"doc_id": "doc-a"},
                {"doc_id": "doc-a"},
                {"doc_id": "doc-b"},
            ]
        ],
        retrieved_contexts=[
            [
                "Unrelated governance notes from the correct document.",
                "It was commissioned in 1998 and has a rated capacity of 40 MW.",
                "The maintenance window is Monday 02:00-04:00 UTC.",
            ]
        ],
        sample_metadata=[
            {
                "evidence": [
                    {
                        "source_id": "doc-a",
                        "quote": "It was commissioned in 1998 and has a rated capacity of 40 MW.",
                    },
                    {
                        "source_id": "doc-b",
                        "quote": "The scheduled maintenance window is Monday 02:00-04:00 UTC.",
                    },
                ]
            }
        ],
        k_values=[1, 3],
    )

    assert result.per_sample[0]["hit@1"] == 1.0
    assert result.per_sample[0]["span_hit@1"] == 0.0
    assert result.per_sample[0]["span_mrr@3"] == 0.5
    assert result.per_sample[0]["span_recall@3"] == 1.0
    assert result.per_sample[0]["span_ndcg@3"] > 0.0


class TestNdcgDuplicateDocs:
    """Regression: nDCG counted duplicate chunks from the same gold document.

    run7 reported ndcg@10 means of 1.41–1.66 although per-question nDCG is
    bounded by 1: with a single gold document and hybrid retrieval returning
    several chunks of it (e.g. [doc1, x, doc1, x, doc1]), every duplicate
    position added gain to the DCG while IDCG assumed a single relevant item.
    nDCG must rank *unique* documents.
    """

    def test_ndcg_never_exceeds_one_with_duplicate_gold_chunks(self):
        # 3 of the top-5 retrieved chunks come from the single gold document.
        assert _ndcg_at_k({"doc1"}, ["doc1", "x", "doc1", "x", "doc1"], 5) == 1.0

    def test_ndcg_with_duplicates_still_ranks_earlier_hits_higher(self):
        perfect = _ndcg_at_k({"doc1"}, ["doc1", "x", "doc1", "x", "doc1"], 5)
        later = _ndcg_at_k({"doc1"}, ["x", "x", "x", "x", "doc1"], 5)
        assert perfect > later > 0.0

    def test_ndcg_keeps_original_positions_and_credits_gold_once(self):
        # Deduplicating before ranking would renumber doc1 to rank 1 and
        # inflate nDCG to ~0.63; doc1 actually sits at position 5.
        assert _ndcg_at_k({"doc1"}, ["x", "x", "x", "x", "doc1"], 5) == pytest.approx(
            1.0 / math.log2(6)
        )
        # A gold document appearing several times is credited once, at its
        # first (real) rank.
        assert _ndcg_at_k({"doc1"}, ["doc1", "x", "doc1", "x", "doc1"], 5) == 1.0

    def test_ndcg_multi_gold_capped_at_one(self):
        retrieved = ["a", "a", "b", "b", "c", "d", "e", "f", "g", "h"]
        assert _ndcg_at_k({"a", "b", "c"}, retrieved, 10) <= 1.0


class TestMissingDocIdsKeepRanks:
    """Regression: chunks without a doc id shifted gold documents up.

    Dropping chunks without a provider doc id from the ranked list moved gold
    documents to better ranks and widened the top-k window beyond k chunks.
    The metrics must be computed on the ranked list with ``None`` placeholders
    kept in place.
    """

    def test_gold_behind_chunks_without_doc_id_is_not_a_hit(self):
        result = compute_gold_doc_retrieval_metrics(
            gold_doc_ids=["doc1"],
            retrieved_metadata=[[{}, {}, {"doc_id": "doc1"}]],
            k_values=[1],
        )
        assert result.per_sample[0]["hit@1"] == 0.0
        assert result.per_sample[0]["ndcg@1"] == 0.0
        assert result.per_sample[0]["recall@1"] == 0.0

    def test_gold_at_true_rank_gets_true_ndcg(self):
        result = compute_gold_doc_retrieval_metrics(
            gold_doc_ids=["doc1"],
            retrieved_metadata=[[{}, {"doc_id": "doc1"}]],
            k_values=[2],
        )
        # doc1 is at rank 2 → 1/log2(3), not rank 1 → 1.0
        assert result.per_sample[0]["ndcg@2"] == pytest.approx(1.0 / math.log2(3))
