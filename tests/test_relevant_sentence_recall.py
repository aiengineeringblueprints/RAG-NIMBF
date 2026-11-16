"""Tests for the RAGBench relevant_sentence_recall metric and label loading."""

import pytest

from benchmark.dataset_adapters import get_adapter
from benchmark.gold_retrieval_metrics import (
    compute_relevant_sentence_recall,
    resolve_relevant_sentences,
)


def _meta(
    docs_sentences=None,
    relevant_keys=None,
    adherence_score=1.0,
):
    metadata = {
        "documents_sentences": docs_sentences or [],
        "all_relevant_sentence_keys": relevant_keys or [],
        "adherence_score": adherence_score,
    }
    return metadata


class TestResolveRelevantSentences:
    def test_resolves_doc_sent_keys(self):
        meta = _meta(
            docs_sentences=[["First sentence.", "Second sentence."]],
            relevant_keys=["0.1"],
        )
        assert resolve_relevant_sentences(meta) == ["Second sentence."]

    def test_handles_list_of_sentence_lists_per_document(self):
        meta = _meta(
            docs_sentences=[["A.", "B."], ["C.", "D."]],
            relevant_keys=["1.0", "0.1"],
        )
        assert set(resolve_relevant_sentences(meta)) == {"C.", "B."}

    def test_no_labels_returns_empty(self):
        assert resolve_relevant_sentences(None) == []
        assert resolve_relevant_sentences({}) == []
        assert resolve_relevant_sentences(_meta()) == []

    def test_resolves_ragbench_techqa_key_pairs(self):
        # Real RAGBench shape: documents are lists of [key, sentence] pairs
        # and relevant keys reference those keys directly ("3a" = doc 3, "a").
        meta = _meta(
            docs_sentences=[
                [
                    ["0a", "First sentence."],
                    ["0b", "Second sentence."],
                ],
                [["3a", "Third sentence."]],
            ],
            relevant_keys=["3a", "0b", "3a."],
        )
        assert set(resolve_relevant_sentences(meta)) == {
            "Third sentence.",
            "Second sentence.",
        }


class TestRelevantSentenceRecall:
    def test_perfect_retrieval(self):
        contexts = [["First sentence.", "Second sentence."]]
        meta = _meta(
            docs_sentences=[["First sentence.", "Second sentence."]],
            relevant_keys=["0.0", "0.1"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 1.0
        assert result.metric_means["relevant_sentence_recall"] == 1.0
        assert result.samples_with_valid_scores == 1

    def test_partial_retrieval(self):
        contexts = [["First sentence."]]
        meta = _meta(
            docs_sentences=[["First sentence.", "Second sentence."]],
            relevant_keys=["0.0", "0.1"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 0.5
        assert result.metric_means["relevant_sentence_recall"] == pytest.approx(0.5)

    def test_robust_to_case_and_whitespace(self):
        contexts = [["First   SENTENCE.\n"]]
        meta = _meta(
            docs_sentences=[["First sentence."]],
            relevant_keys=["0.0"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 1.0

    def test_sentence_split_across_chunk_boundary(self):
        # The relevant sentence is split across two retrieved chunks.
        contexts = [["First sen", "tence."]]
        meta = _meta(
            docs_sentences=[["First sentence."]],
            relevant_keys=["0.0"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 1.0

    def test_sentence_split_with_overlapping_chunks(self):
        # Chunks overlap: "brown" appears in both fragments.
        contexts = [["The quick brown", "brown fox jumps."]]
        meta = _meta(
            docs_sentences=[["The quick brown fox jumps."]],
            relevant_keys=["0.0"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 1.0

    def test_intact_sentence_matches_in_any_chunk(self):
        # A complete sentence inside any single chunk counts regardless of
        # which chunk holds it (order independence for intact sentences).
        contexts = [["Unrelated text.", "First sentence."]]
        meta = _meta(
            docs_sentences=[["First sentence."]],
            relevant_keys=["0.0"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 1.0

    def test_word_boundary_preserved_no_false_positive(self):
        # "The rapist escaped." must not match a chunk saying
        # "Therapist escaped." — separators are meaningful.
        contexts = [["Therapist escaped."]]
        meta = _meta(
            docs_sentences=[["The rapist escaped."]],
            relevant_keys=["0.0"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 0.0

    def test_short_sentence_not_matched_inside_longer_word(self):
        # "Yes." must not match "yesterday" — boundary-padded comparison.
        contexts = [["yesterday"]]
        meta = _meta(docs_sentences=[["Yes."]], relevant_keys=["0.0"])
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 0.0

    def test_compact_repair_never_matches_inside_single_chunk(self):
        # With >= 2 chunks the compact joins contain each chunk whole; a
        # sentence that sits entirely inside one chunk must still be scored
        # only by the word-boundary path ("The rapist ..." vs
        # "Therapist ...").
        contexts = [["Therapist escaped.", "Other text."]]
        meta = _meta(
            docs_sentences=[["The rapist escaped."]],
            relevant_keys=["0.0"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 0.0

    def test_split_sentence_matches_across_non_adjacent_ranks(self):
        # The two fragments retrieved at ranks 1 and 3 (not adjacent) must
        # still match; recall must not depend on reranker order.
        contexts = [["tence.", "Unrelated chunk.", "First sen"]]
        meta = _meta(
            docs_sentences=[["First sentence."]],
            relevant_keys=["0.0"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 1.0

    def test_duplicate_keys_counted_once(self):
        # Duplicate keys ("0a" twice, "0a."/"0a" variants) must not inflate
        # the denominator: 1 of 2 unique sentences retrieved → 0.5, not 0.25.
        contexts = [["Second sentence."]]
        meta = _meta(
            docs_sentences=[
                [["0a", "First sentence."], ["0b", "Second sentence."]]
            ],
            relevant_keys=["0a", "0a", "0a.", "0b"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == pytest.approx(0.5)

    def test_empty_and_non_latin_sentences_never_score(self):
        # ASCII-only normalisation would make these empty strings, which
        # trivially "match" anything; they must be excluded instead.
        contexts = [[]]
        meta = _meta(
            docs_sentences=[["", "你好世界。", "Real sentence."]],
            relevant_keys=["0.0", "0.1", "0.2"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 0.0  # only the real sentence is scored

    def test_malformed_pair_values_are_ignored(self):
        meta = _meta(
            docs_sentences=[[["0a", None], ["0b", ["nested"]], ["0c", "Real."]]],
            relevant_keys=["0a", "0b", "0c"],
        )
        assert resolve_relevant_sentences(meta) == ["Real."]

    def test_no_relevant_sentences_gets_no_score_not_zero(self):
        contexts = [["Irrelevant chunk."]]
        meta = _meta(docs_sentences=[["Some sentence."]], relevant_keys=[])
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] is None
        assert result.metric_means == {}
        assert result.samples_with_valid_scores == 0

    def test_mean_excludes_unanswerable_samples(self):
        contexts = [["First sentence."]]
        meta_answerable = _meta(
            docs_sentences=[["First sentence."]],
            relevant_keys=["0.0"],
        )
        meta_unanswerable = _meta(
            docs_sentences=[["First sentence."]],
            relevant_keys=[],
        )
        result = compute_relevant_sentence_recall(
            contexts, [meta_answerable, meta_unanswerable]
        )
        assert result.metric_means["relevant_sentence_recall"] == 1.0
        assert result.samples_with_valid_scores == 1
        assert result.per_sample == [1.0, None]

    def test_dataset_without_labels_skips_silently(self):
        contexts = [["Some chunk."]]
        metadata = [{"gold_doc_id": "doc1"}, {}]
        result = compute_relevant_sentence_recall(contexts, metadata)
        assert result.metric_means == {}
        assert result.per_sample == [None, None]
        assert result.samples_with_valid_scores == 0


class TestRagbenchMetadataKeys:
    def test_ragbench_adapter_carries_sentence_labels(self):
        adapter = get_adapter("ragbench")
        for key in (
            "all_relevant_sentence_keys",
            "documents_sentences",
            "adherence_score",
        ):
            assert key in adapter.metadata_keys

    def test_answerable_flag_true_with_relevant_sentences(self):
        from benchmark.dataset_adapters import derive_answerable_flag

        metadata = {"all_relevant_sentence_keys": ["0.0"]}
        derive_answerable_flag(metadata)
        assert metadata["answerable"] is True

    def test_answerable_flag_false_without_relevant_sentences(self):
        from benchmark.dataset_adapters import derive_answerable_flag

        metadata = {"all_relevant_sentence_keys": []}
        derive_answerable_flag(metadata)
        assert metadata["answerable"] is False

    def test_answerable_flag_untouched_without_labels(self):
        from benchmark.dataset_adapters import derive_answerable_flag

        metadata = {"gold_doc_id": "doc1"}
        derive_answerable_flag(metadata)
        assert "answerable" not in metadata


class TestRunnerIntegration:
    def test_with_relevant_sentence_recall_merges_metric(self):
        from benchmark.custom_metrics import CustomMetricsResult
        from benchmark.orchestration.runner import _with_relevant_sentence_recall

        custom_result = CustomMetricsResult(
            metric_means={"context_relevance": 0.7},
            per_sample=[{"context_relevance": 0.7}],
            samples_with_valid_scores={"context_relevance": 1},
        )
        metadata = [
            {
                "documents_sentences": [["Found it.", "Other."]],
                "all_relevant_sentence_keys": ["0.0"],
                "adherence_score": 1.0,
            }
        ]
        merged = _with_relevant_sentence_recall(
            custom_result, metadata, [["Found it."]]
        )
        assert merged.metric_means["relevant_sentence_recall"] == 1.0
        assert merged.per_sample[0]["relevant_sentence_recall"] == 1.0

    def test_without_labels_custom_result_unchanged(self):
        from benchmark.custom_metrics import CustomMetricsResult
        from benchmark.orchestration.runner import _with_relevant_sentence_recall

        custom_result = CustomMetricsResult(
            metric_means={"context_relevance": 0.7},
            per_sample=[{"context_relevance": 0.7}],
            samples_with_valid_scores={"context_relevance": 1},
        )
        merged = _with_relevant_sentence_recall(
            custom_result, [{"gold_doc_id": "d"}], [["chunk"]]
        )
        assert merged.metric_means == {"context_relevance": 0.7}
        assert merged.per_sample == [{"context_relevance": 0.7}]

    def test_corpus_metadata_excludes_sentence_labels(self):
        from unittest.mock import patch

        from benchmark.dataset import load_corpus_and_questions
        from benchmark.dataset_adapters import strip_sentence_label_metadata

        metadata = {
            "documents_sentences": [["Document text"]],
            "all_relevant_sentence_keys": ["0a"],
            "adherence_score": 1.0,
            "answerable": True,
            "gold_doc_id": "doc1",
        }
        assert strip_sentence_label_metadata(metadata) == {
            "gold_doc_id": "doc1"
        }

        samples = [
            {
                "question": f"q{i}",
                "ground_truth": "gt",
                "context": f"Document text {i} with content.",
                "metadata": dict(metadata),
            }
            for i in range(2)
        ]
        with patch(
            "benchmark.dataset.load_benchmark_data", return_value=samples
        ):
            corpus, questions = load_corpus_and_questions(dataset_name="squad")
        for doc in corpus:
            assert "documents_sentences" not in doc["metadata"]
            assert "all_relevant_sentence_keys" not in doc["metadata"]
            assert "adherence_score" not in doc["metadata"]
        # Sample metadata keeps the labels for the metric.
        assert questions[0]["metadata"]["all_relevant_sentence_keys"] == ["0a"]

    def test_normalize_sample_derives_answerable_flag(self):
        from benchmark.dataset import normalize_sample

        sample = normalize_sample(
            {
                "question": "q",
                "ground_truth": "gt",
                "context": "ctx",
                "metadata": {"all_relevant_sentence_keys": []},
            }
        )
        assert sample["metadata"]["answerable"] is False
