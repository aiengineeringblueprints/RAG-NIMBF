"""Tests for benchmark.dataset and benchmark.dataset_adapters."""

import pytest
from unittest.mock import patch, MagicMock

from benchmark.dataset_adapters import (
    get_adapter,
    REGISTRY,
    _t2_ragbench_context,
    _ragbench_context,
    _squad_ground_truth,
    _ragas_wikiqa_context,
    _ragperf_wikipedia_nq_context,
)
from benchmark.dataset import load_benchmark_data, load_corpus_and_questions


# ---------------------------------------------------------------------------
# Adapter context builders
# ---------------------------------------------------------------------------

class TestT2RagbenchContext:
    def test_all_fields(self):
        row = {"pre_text": "before", "table": "table data", "post_text": "after"}
        result = _t2_ragbench_context(row)
        assert "before" in result
        assert "table data" in result
        assert "after" in result

    def test_pre_text_only(self):
        result = _t2_ragbench_context({"pre_text": "hello"})
        assert result == "hello"

    def test_fallback_to_context_field(self):
        result = _t2_ragbench_context({"context": "fallback"})
        assert result == "fallback"

    def test_empty_row(self):
        result = _t2_ragbench_context({})
        assert result == ""


class TestRagbenchContext:
    def test_documents_list(self):
        row = {"documents": ["doc1", "doc2"]}
        result = _ragbench_context(row)
        assert "doc1" in result
        assert "doc2" in result

    def test_context_string_fallback(self):
        row = {"context": "some context text"}
        result = _ragbench_context(row)
        assert result == "some context text"

    def test_empty_row(self):
        result = _ragbench_context({})
        assert result == ""


class TestSquadGroundTruth:
    def test_dict_with_text(self):
        raw = {"text": ["Paris"], "answer_start": [42]}
        assert _squad_ground_truth(raw) == "Paris"

    def test_empty_text_list(self):
        raw = {"text": [], "answer_start": []}
        # Falls through to str() because text list is empty (falsy)
        result = _squad_ground_truth(raw)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_non_dict(self):
        assert _squad_ground_truth("hello") == "hello"


class TestRagasWikiqaContext:
    def test_list_of_chunks(self):
        row = {"context": ["chunk one", "chunk two", "chunk three"]}
        result = _ragas_wikiqa_context(row)
        assert result == "chunk one\n\nchunk two\n\nchunk three"

    def test_single_string(self):
        row = {"context": "a single string"}
        result = _ragas_wikiqa_context(row)
        assert result == "a single string"

    def test_empty_row(self):
        result = _ragas_wikiqa_context({})
        assert result == ""


class TestRagperfWikipediaNqContext:
    def test_text_field(self):
        result = _ragperf_wikipedia_nq_context({"text": "Wikipedia article"})
        assert result == "Wikipedia article"

    def test_empty_row(self):
        result = _ragperf_wikipedia_nq_context({})
        assert result == ""


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

class TestDatasetAdapters:
    def test_t2_ragbench_registered(self):
        adapter = get_adapter("t2-ragbench")
        assert adapter.hf_id == "G4KMU/t2-ragbench"
        assert adapter.question_key == "question"
        assert adapter.ground_truth_key == "program_answer"
        assert adapter.requires_subset is True

    def test_ragbench_registered(self):
        adapter = get_adapter("ragbench")
        assert adapter.hf_id == "rungalileo/ragbench"
        assert adapter.question_key == "question"
        assert adapter.ground_truth_key == "response"
        assert adapter.requires_subset is True

    def test_squad_registered(self):
        adapter = get_adapter("squad")
        assert adapter.hf_id == "rajpurkar/squad"
        assert adapter.ground_truth_transform is not None

    def test_ragas_wikiqa_registered(self):
        adapter = get_adapter("ragas-wikiqa")
        assert adapter.hf_id == "vibrantlabsai/ragas-wikiqa"
        assert adapter.question_key == "question"
        assert adapter.ground_truth_key == "correct_answer"
        assert adapter.preferred_split == "train"
        assert adapter.requires_subset is False

    def test_ragperf_wikipedia_nq_registered(self):
        adapter = get_adapter("ragperf-wikipedia-nq")
        assert adapter.hf_id == "sentence-transformers/natural-questions"
        assert adapter.question_key == "query"
        assert adapter.ground_truth_key == "answer"
        assert adapter.preferred_split == "train"
        assert adapter.has_shared_corpus is True

    def test_unknown_adapter_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_adapter("nonexistent_dataset")

    def test_all_adapters_have_required_fields(self):
        for name, adapter in REGISTRY.items():
            assert adapter.name == name
            assert adapter.hf_id
            assert adapter.question_key
            assert adapter.ground_truth_key
            assert callable(adapter.build_context)


# ---------------------------------------------------------------------------
# load_benchmark_data
# ---------------------------------------------------------------------------

class TestLoadBenchmarkData:
    @patch("benchmark.dataset.load_dataset")
    def test_t2_ragbench_loads_and_transforms(self, mock_load):
        mock_ds = MagicMock()
        mock_split = MagicMock()
        mock_split.__len__ = MagicMock(return_value=1)
        mock_split.__iter__ = MagicMock(return_value=iter([
            {
                "question": "What is the revenue?",
                "program_answer": "42",
                "pre_text": "Revenue report",
                "table": None,
                "post_text": None,
                "context": None,
                "file_name": "report.pdf",
                "company_name": "ACME",
            }
        ]))
        mock_ds.__contains__ = MagicMock(return_value=False)
        mock_ds.keys.return_value = ["train"]
        mock_ds.__getitem__ = MagicMock(return_value=mock_split)
        mock_load.return_value = mock_ds

        mock_split.shuffle.return_value.select.return_value = mock_split

        samples = load_benchmark_data(
            dataset_name="t2-ragbench",
            subset="FinQA",
            sample_size=50,
        )

        assert len(samples) == 1
        assert samples[0]["question"] == "What is the revenue?"
        assert samples[0]["ground_truth"] == "42"

    @patch("benchmark.dataset.load_dataset")
    def test_ragbench_loads(self, mock_load):
        mock_ds = MagicMock()
        mock_split = MagicMock()
        mock_split.__len__ = MagicMock(return_value=1)
        mock_split.__iter__ = MagicMock(return_value=iter([
            {
                "question": "What is X?",
                "response": "Y",
                "documents": ["doc text"],
                "id": "123",
                "dataset_name": "covidqa",
            }
        ]))
        mock_ds.__contains__ = MagicMock(return_value=True)
        mock_ds.keys.return_value = ["test", "train"]
        mock_ds.__getitem__ = MagicMock(return_value=mock_split)
        mock_load.return_value = mock_ds

        mock_split.shuffle.return_value.select.return_value = mock_split

        samples = load_benchmark_data(
            dataset_name="ragbench",
            subset="covidqa",
            sample_size=50,
        )

        assert len(samples) == 1
        assert samples[0]["question"] == "What is X?"
        assert samples[0]["ground_truth"] == "Y"
        assert "doc text" in samples[0]["context"]

    @patch.dict("benchmark.dataset.os.environ", {"RAGPERF_WIKIPEDIA_CORPUS_SIZE": "2"})
    @patch("benchmark.dataset.load_dataset")
    def test_ragperf_wikipedia_nq_loads_corpus_and_questions(self, mock_load):
        wiki = MagicMock()
        wiki.__len__ = MagicMock(return_value=2)
        wiki.__iter__ = MagicMock(return_value=iter([
            {"id": "w1", "title": "Alpha", "text": "Alpha article"},
            {"id": "w2", "title": "Beta", "text": "Beta article"},
        ]))

        nq = MagicMock()
        nq.__len__ = MagicMock(return_value=1)
        nq.__iter__ = MagicMock(return_value=iter([
            {"query": "What is Alpha?", "answer": "Alpha answer"},
        ]))

        mock_load.side_effect = [wiki, nq]

        corpus, questions = load_corpus_and_questions(
            dataset_name="ragperf-wikipedia-nq",
            sample_size=1,
        )

        assert len(corpus) == 2
        assert corpus[0]["context"] == "Alpha article"
        assert corpus[0]["metadata"]["title"] == "Alpha"
        assert len(questions) == 1
        assert questions[0]["question"] == "What is Alpha?"
        assert questions[0]["ground_truth"] == "Alpha answer"
        assert questions[0]["metadata"]["retrieval_ground_truth"] == "unavailable"
