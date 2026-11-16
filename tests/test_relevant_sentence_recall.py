"""Tests for the RAGBench relevant_sentence_recall metric and label loading."""

import pytest

from benchmark.dataset_adapters import get_adapter
from benchmark.gold_retrieval_metrics import (
    compute_relevant_sentence_recall,
    resolve_relevant_sentences,
)

# Sentences long enough to be scored (>= 3 words / >= 20 compact chars).
SENT_A = "An out of memory dump may be observed on the system."
SENT_B = "Further investigation reveals a failed thread creation message."


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
            docs_sentences=[["First sentence.", SENT_A]],
            relevant_keys=["0.1"],
        )
        assert resolve_relevant_sentences(meta) == [SENT_A]

    def test_handles_list_of_sentence_lists_per_document(self):
        meta = _meta(
            docs_sentences=[["A.", SENT_A], ["C.", SENT_B]],
            relevant_keys=["1.0", "0.1"],
        )
        assert set(resolve_relevant_sentences(meta)) == {"C.", SENT_A}

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
                    ["0b", SENT_A],
                ],
                [["3a", SENT_B]],
            ],
            relevant_keys=["3a", "0b", "3a."],
        )
        assert set(resolve_relevant_sentences(meta)) == {
            SENT_B,
            SENT_A,
        }

    def test_malformed_pair_values_are_ignored(self):
        meta = _meta(
            docs_sentences=[[["0a", None], ["0b", ["nested"]], ["0c", SENT_A]]],
            relevant_keys=["0a", "0b", "0c"],
        )
        assert resolve_relevant_sentences(meta) == [SENT_A]


class TestRelevantSentenceRecall:
    def test_perfect_retrieval(self):
        contexts = [[SENT_A, SENT_B]]
        meta = _meta(
            docs_sentences=[[SENT_A, SENT_B]],
            relevant_keys=["0.0", "0.1"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 1.0
        assert result.metric_means["relevant_sentence_recall"] == 1.0
        assert result.samples_with_valid_scores == 1

    def test_partial_retrieval(self):
        contexts = [[SENT_A]]
        meta = _meta(
            docs_sentences=[[SENT_A, SENT_B]],
            relevant_keys=["0.0", "0.1"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 0.5
        assert result.metric_means["relevant_sentence_recall"] == pytest.approx(0.5)

    def test_robust_to_case_and_whitespace(self):
        contexts = [["An   OUT   of\nmemory DUMP may be observed on the system."]]
        meta = _meta(
            docs_sentences=[[SENT_A]],
            relevant_keys=["0.0"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 1.0

    def test_sentence_split_across_chunk_boundary(self):
        # The relevant sentence is split across two retrieved chunks.
        contexts = [["An out of memory dump may be", " observed on the system."]]
        meta = _meta(
            docs_sentences=[[SENT_A]],
            relevant_keys=["0.0"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 1.0

    def test_sentence_split_with_overlapping_chunks(self):
        # Chunks overlap: "brown" appears in both fragments.
        contexts = [["The quick brown", "brown fox jumps over the lazy dog."]]
        meta = _meta(
            docs_sentences=[["The quick brown fox jumps over the lazy dog."]],
            relevant_keys=["0.0"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 1.0

    def test_intact_sentence_matches_in_any_chunk(self):
        # A complete sentence inside any single chunk counts regardless of
        # which chunk holds it (order independence for intact sentences).
        contexts = [["Unrelated text.", SENT_A]]
        meta = _meta(
            docs_sentences=[[SENT_A]],
            relevant_keys=["0.0"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 1.0

    def test_split_sentence_matches_across_non_adjacent_ranks(self):
        # The two fragments retrieved at ranks 1 and 3 (not adjacent) must
        # still match; recall must not depend on reranker order.
        contexts = [[
            " observed on the system.",
            "Unrelated chunk with filler text.",
            "An out of memory dump may be",
        ]]
        meta = _meta(
            docs_sentences=[[SENT_A]],
            relevant_keys=["0.0"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 1.0

    def test_duplicate_keys_counted_once(self):
        # Duplicate keys ("0a" twice, "0a."/"0a" variants) must not inflate
        # the denominator: 1 of 2 unique sentences retrieved → 0.5, not 0.25.
        contexts = [[SENT_B]]
        meta = _meta(
            docs_sentences=[
                [["0a", SENT_A], ["0b", SENT_B]]
            ],
            relevant_keys=["0a", "0a", "0a.", "0b"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == pytest.approx(0.5)

    def test_no_relevant_sentences_gets_no_score_not_zero(self):
        contexts = [["Irrelevant chunk."]]
        meta = _meta(docs_sentences=[[SENT_A]], relevant_keys=[])
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] is None
        assert result.metric_means == {}
        assert result.samples_with_valid_scores == 0

    def test_mean_excludes_unanswerable_samples(self):
        contexts = [[SENT_A]]
        meta_answerable = _meta(
            docs_sentences=[[SENT_A]],
            relevant_keys=["0.0"],
        )
        meta_unanswerable = _meta(
            docs_sentences=[[SENT_A]],
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

    def test_word_boundary_preserved_no_false_positive(self):
        # "The rapist escaped quickly from custody." must not match a chunk
        # saying "Therapist escaped quickly from custody." — separators are
        # meaningful.
        contexts = [["Therapist escaped quickly from custody."]]
        meta = _meta(
            docs_sentences=[["The rapist escaped quickly from custody."]],
            relevant_keys=["0.0"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 0.0

    def test_short_sentence_not_matched_inside_longer_word(self):
        # "Yes." is label-like (fewer than 3 words and < 20 compact chars):
        # it is excluded from scoring entirely instead of matching anything.
        contexts = [["yesterday"]]
        meta = _meta(docs_sentences=[["Yes."]], relevant_keys=["0.0"])
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] is None

    def test_label_like_sentence_excluded_from_denominator(self):
        # Only the real sentence is scored; label-like ones don't drag it.
        contexts = [[SENT_A]]
        meta = _meta(
            docs_sentences=[["Yes.", SENT_A]],
            relevant_keys=["0.0", "0.1"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 1.0

    def test_reviewer_repro_results_heading(self):
        # Reviewer repro: 'Results.' vs 'The results were bad.' — the
        # label-like sentence is excluded, the sample gets no score.
        contexts = [["The results were bad."]]
        meta = _meta(docs_sentences=[["Results."]], relevant_keys=["0.0"])
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] is None

    def test_reviewer_repro_table_reference(self):
        # Reviewer repro: 'Table 1.' vs 'see table 1 for x' — excluded.
        contexts = [["see table 1 for x"]]
        meta = _meta(docs_sentences=[["Table 1."]], relevant_keys=["0.0"])
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] is None

    def test_reviewer_repro_split_tiny_fragments(self):
        # Reviewer repro: 'Yes it is.' matching via tiny fragments
        # ("yes" + "it is") across two chunks — split repair requires
        # fragments of >= 4 compact chars, so no match; the sentence is
        # label-like anyway and excluded from scoring.
        contexts = [["he said yes", "it is raining"]]
        meta = _meta(docs_sentences=[["Yes it is."]], relevant_keys=["0.0"])
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] is None

    def test_reviewer_repro_overlap_reverse_order(self):
        # Reviewer repro: overlap between the two fragments in reverse rank
        # order must be repaired as well.
        contexts = [["gamma delta epsilon", "alpha beta gamma"]]
        meta = _meta(
            docs_sentences=[["alpha beta gamma delta epsilon."]],
            relevant_keys=["0.0"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 1.0

    def test_reviewer_repro_short_sentence_not_trivially_matched(self):
        # 'Yes.' inside a chunk must not count — the sentence is label-like
        # and excluded from scoring, so the sample gets no score at all
        # rather than a spurious 1.0.
        contexts = [["Yes, the drug is not approved yet."]]
        meta = _meta(docs_sentences=[["Yes."]], relevant_keys=["0.0"])
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] is None

    def test_compact_repair_never_matches_inside_single_chunk(self):
        # With >= 2 chunks the compact joins contain each chunk whole; a
        # sentence that sits entirely inside one chunk must still be scored
        # only by the word-boundary path.
        contexts = [["Therapist escaped quickly from custody.", "Other text."]]
        meta = _meta(
            docs_sentences=[["The rapist escaped quickly from custody."]],
            relevant_keys=["0.0"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 0.0

    def test_empty_and_non_latin_sentences_never_score(self):
        # ASCII-only normalisation would make these empty strings, which
        # trivially "match" anything; they must be excluded instead.
        contexts = [[]]
        meta = _meta(
            docs_sentences=[["", "你好世界。", SENT_A]],
            relevant_keys=["0.0", "0.1", "0.2"],
        )
        result = compute_relevant_sentence_recall(contexts, [meta])
        assert result.per_sample[0] == 0.0  # only the real sentence is scored


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
                "documents_sentences": [[SENT_A, SENT_B]],
                "all_relevant_sentence_keys": ["0.0"],
                "adherence_score": 1.0,
            }
        ]
        merged = _with_relevant_sentence_recall(
            custom_result, metadata, [[SENT_A]]
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

    def test_normalize_sample_does_not_mutate_caller_metadata(self):
        from benchmark.dataset import normalize_sample

        metadata = {"all_relevant_sentence_keys": ["0a"]}
        sample = normalize_sample(
            {
                "question": "q",
                "ground_truth": "gt",
                "context": "ctx",
                "metadata": metadata,
            }
        )
        assert sample["metadata"]["answerable"] is True
        assert "answerable" not in metadata

    def test_answerable_flag_parses_json_string_values(self):
        from benchmark.dataset_adapters import derive_answerable_flag

        metadata = {"all_relevant_sentence_keys": "[]"}
        derive_answerable_flag(metadata)
        assert metadata["answerable"] is False

        metadata = {"all_relevant_sentence_keys": '["0a"]'}
        derive_answerable_flag(metadata)
        assert metadata["answerable"] is True

    def test_answerable_flag_blank_csv_cell_is_not_answerable(self):
        # Reviewer repro: an empty CSV cell ("" / whitespace) must behave
        # like an empty key list, not become answerable via bool(['']).
        from benchmark.dataset_adapters import derive_answerable_flag

        for blank in ("", "   "):
            metadata = {"all_relevant_sentence_keys": blank}
            derive_answerable_flag(metadata)
            assert metadata["answerable"] is False, repr(blank)

    def test_chroma_safe_metadata_drops_empty_lists(self):
        # Reviewer follow-up: empty lists are dropped entirely so a key
        # keeps one type across all documents (no "[]" vs list mismatch).
        from benchmark.dataset import _chroma_safe_metadata

        safe = _chroma_safe_metadata({"all_relevant_sentence_keys": []})
        assert "all_relevant_sentence_keys" not in safe

    def test_chroma_safe_metadata_json_encodes_none_and_mixed_lists(self):
        # Reviewer repro: [1, None], [None], [1, 'a'] crash Chroma 1.5.9;
        # they must be serialised instead of passed through.
        from benchmark.dataset import _chroma_safe_metadata

        for bad in ([1, None], [None], [1, "a"]):
            safe = _chroma_safe_metadata({"tags": bad})
            assert safe["tags"] == __import__("json").dumps(bad)

    def test_chroma_safe_metadata_keeps_uniform_primitive_lists(self):
        from benchmark.dataset import _chroma_safe_metadata

        safe = _chroma_safe_metadata({"tags": ["a", "b"], "pages": [1, 2]})
        assert safe["tags"] == ["a", "b"]
        assert safe["pages"] == [1, 2]

    def test_chunk_documents_strips_sentence_labels(self):
        from benchmark.chunking import chunk_documents
        from benchmark.chunking import get_chunker

        chunker = get_chunker("recursive", 512, 64)
        docs = [
            {
                "context": "Some long enough document text for chunking. " * 5,
                "metadata": {
                    "doc_id": "doc1",
                    "documents_sentences": [["Full document text."]],
                    "all_relevant_sentence_keys": ["0a"],
                    "adherence_score": 1.0,
                },
            }
        ]
        chunks = chunk_documents(chunker, docs)
        for chunk in chunks:
            assert "documents_sentences" not in chunk.metadata
            assert "all_relevant_sentence_keys" not in chunk.metadata
            assert "adherence_score" not in chunk.metadata
            assert chunk.metadata["doc_id"] == "doc1"
