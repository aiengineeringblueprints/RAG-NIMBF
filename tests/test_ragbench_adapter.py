"""Tests for the RAGBench component adapters and TRACe sidecar wiring."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmark.dataset import load_benchmark_data
from benchmark.dataset_adapters import REGISTRY, get_adapter
from benchmark.ragbench_adapter import (
    RAGBENCH_COMPONENTS,
    RAGBENCH_TRACE_KEYS,
    _ragbench_component_context,
    _ragbench_trace_metadata,
    trace_sidecar_path,
    write_trace_sidecar,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_twelve_components_registered(self):
        for component in RAGBENCH_COMPONENTS:
            assert f"ragbench_{component}" in REGISTRY

    def test_cuad_adapter_fields(self):
        adapter = get_adapter("ragbench_cuad")
        assert adapter.hf_id == "rungalileo/ragbench"
        assert adapter.question_key == "question"
        assert adapter.ground_truth_key == "response"
        assert adapter.preferred_split == "test"
        assert adapter.requires_subset is True

    def test_pubmedqa_adapter_fields(self):
        adapter = get_adapter("ragbench_pubmedqa")
        assert adapter.hf_id == "rungalileo/ragbench"
        assert adapter.preferred_split == "test"

    def test_twelve_components_match_paper(self):
        expected = {
            "pubmedqa",
            "covidqa",
            "hotpotqa",
            "msmarco",
            "hagrid",
            "expertqa",
            "cuad",
            "emanual",
            "techqa",
            "finqa",
            "tatqa",
            "delucionqa",
        }
        assert set(RAGBENCH_COMPONENTS) == expected


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


class TestBuildContext:
    def test_documents_list_joined_with_blank_lines(self):
        row = {"documents": ["doc one", "doc two", "doc three"]}
        assert _ragbench_component_context(row) == "doc one\n\ndoc two\n\ndoc three"

    def test_documents_single_string(self):
        assert _ragbench_component_context({"documents": "single"}) == "single"

    def test_unannotated_context_fallback(self):
        row = {"unannotated_context": ["fall", "back"]}
        assert _ragbench_component_context(row) == "fall\n\nback"

    def test_gpt3_context_string_fallback(self):
        assert _ragbench_component_context({"gpt3_context": "gpt"}) == "gpt"

    def test_context_field_fallback(self):
        assert _ragbench_component_context({"context": "ctx"}) == "ctx"

    def test_empty_row_returns_empty_string(self):
        assert _ragbench_component_context({}) == ""


# ---------------------------------------------------------------------------
# TRACe metadata extraction
# ---------------------------------------------------------------------------


class TestTraceMetadata:
    def test_extracts_all_present_keys(self):
        row = {key: f"value-{key}" for key in RAGBENCH_TRACE_KEYS}
        trace = _ragbench_trace_metadata(row)
        assert set(trace) == set(RAGBENCH_TRACE_KEYS)

    def test_omits_missing_keys(self):
        row = {"all_relevant_sentence_keys": ["0a", "1b"], "relevance_score": 0.5}
        trace = _ragbench_trace_metadata(row)
        assert trace == {"all_relevant_sentence_keys": ["0a", "1b"], "relevance_score": 0.5}

    def test_empty_row_returns_empty_dict(self):
        assert _ragbench_trace_metadata({}) == {}


# ---------------------------------------------------------------------------
# Sidecar writer
# ---------------------------------------------------------------------------


class TestSidecar:
    def test_sidecar_path_layout(self, tmp_path):
        path = trace_sidecar_path(tmp_path, "cuad", "test")
        assert path == tmp_path / "ragbench" / "cuad" / "test" / "trace_gold.json"

    def test_write_sidecar_keys_by_id(self, tmp_path):
        rows = [
            {
                "id": "abc",
                "question": "q1",
                "all_relevant_sentence_keys": ["0a"],
                "all_utilized_sentence_keys": ["0a"],
                "relevance_score": 0.9,
            },
            {"id": "def", "question": "q2", "all_relevant_sentence_keys": []},
        ]
        path = write_trace_sidecar(tmp_path, "cuad", "test", rows)
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["component"] == "cuad"
        assert payload["split"] == "test"
        assert set(payload["annotations"]) == {"abc", "def"}
        assert payload["annotations"]["abc"]["relevance_score"] == 0.9

    def test_write_sidecar_falls_back_to_index_key(self, tmp_path):
        rows = [{"question": "q"}]
        path = write_trace_sidecar(tmp_path, "cuad", "test", rows)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "cuad_test_0" in payload["annotations"]


# ---------------------------------------------------------------------------
# load_benchmark_data integration (mocked HF download)
# ---------------------------------------------------------------------------


def _mock_split(rows):
    """Build a MagicMock mimicking a HuggingFace ``Dataset`` split.

    ``shuffle(...).select(range(n))`` returns a view containing the first
    ``n`` rows, mirroring the real ``datasets.Dataset`` contract used by
    ``load_benchmark_data``.
    """

    class _FakeSplit:
        def __init__(self, items):
            self._items = items

        def __len__(self):
            return len(self._items)

        def __iter__(self):
            return iter(self._items)

        def shuffle(self, seed=None):
            return self

        def select(self, indices):
            if isinstance(indices, range):
                return _FakeSplit(self._items[indices.start : indices.stop])
            return _FakeSplit([self._items[i] for i in indices])

    return _FakeSplit(rows)


def _mock_dataset(rows, splits=("test",)):
    mock_ds = MagicMock()
    mock_ds.__contains__ = MagicMock(return_value=True)
    mock_ds.keys.return_value = list(splits)
    mock_ds.__getitem__ = MagicMock(return_value=_mock_split(rows))
    return mock_ds


class TestLoadBenchmarkData:
    @patch("benchmark.dataset.load_dataset")
    def test_ragbench_cuad_loads_question_response_and_trace(self, mock_load, tmp_path):
        row = {
            "id": "cuad-1",
            "question": "What is the termination clause?",
            "response": "Either party may terminate with 30 days notice.",
            "documents": ["SECTION 5. TERMINIATION. ..."],
            "dataset_name": "cuad_test",
            "generation_model_name": "gpt-3.5-turbo",
            "annotating_model_name": "gpt-4o",
            "all_relevant_sentence_keys": ["0a", "0b"],
            "all_utilized_sentence_keys": ["0a"],
            "relevance_score": 0.8,
            "utilization_score": 0.5,
            "completeness_score": 0.6,
            "adherence_score": True,
            "documents_sentences": [[["0a", "SECTION 5."], ["0b", "TERMINATION."]]],
            "response_sentences": [["a", "Either party may terminate."]],
            "sentence_support_information": [],
            "unsupported_response_sentence_keys": [],
        }
        mock_load.return_value = _mock_dataset([row])

        with patch.dict("os.environ", {"DATASET_CACHE_ROOT": str(tmp_path)}):
            samples = load_benchmark_data(
                dataset_name="ragbench_cuad",
                subset=None,
                sample_size=50,
                split="test",
            )

        assert len(samples) == 1
        sample = samples[0]
        assert sample["question"] == "What is the termination clause?"
        assert sample["ground_truth"] == "Either party may terminate with 30 days notice."
        assert "SECTION 5." in sample["context"]
        assert sample["metadata"]["id"] == "cuad-1"
        trace = sample["metadata"]["ragbench_trace"]
        assert trace["all_relevant_sentence_keys"] == ["0a", "0b"]
        assert trace["relevance_score"] == 0.8

        sidecar = tmp_path / "ragbench" / "cuad" / "test" / "trace_gold.json"
        assert sidecar.exists()
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        assert "cuad-1" in payload["annotations"]

        # HF was called with name='cuad' derived from the adapter name, not None.
        _, kwargs = mock_load.call_args
        assert kwargs.get("name") == "cuad"

    @patch("benchmark.dataset.load_dataset")
    def test_explicit_subset_overrides_adapter_name(self, mock_load, tmp_path):
        mock_load.return_value = _mock_dataset(
            [{"id": "1", "question": "q", "response": "a", "documents": ["d"]}]
        )
        with patch.dict("os.environ", {"DATASET_CACHE_ROOT": str(tmp_path)}):
            load_benchmark_data(
                dataset_name="ragbench_cuad",
                subset="cuad",
                sample_size=10,
            )
        _, kwargs = mock_load.call_args
        assert kwargs.get("name") == "cuad"

    @patch("benchmark.dataset.load_dataset")
    def test_split_selection_dev_falls_back_to_first(self, mock_load, tmp_path):
        mock_ds = MagicMock()
        mock_ds.__contains__ = MagicMock(return_value=False)
        mock_ds.keys.return_value = ["train", "validation", "test"]
        mock_ds.__getitem__ = MagicMock(
            return_value=_mock_split(
                [{"id": "1", "question": "q", "response": "a", "documents": ["d"]}]
            )
        )
        mock_load.return_value = mock_ds

        samples = load_benchmark_data(
            dataset_name="ragbench_cuad",
            sample_size=10,
            split="dev",  # not present in the mocked dataset
        )
        assert len(samples) == 1
        # Dev wasn't present, so the first available split (train) was used.
        mock_ds.__getitem__.assert_called_with("train")

    @patch("benchmark.dataset.load_dataset")
    def test_max_examples_caps_rows(self, mock_load, tmp_path):
        rows = [
            {
                "id": str(i),
                "question": f"q{i}",
                "response": f"a{i}",
                "documents": [f"d{i}"],
            }
            for i in range(50)
        ]
        mock_load.return_value = _mock_dataset(rows)

        samples = load_benchmark_data(
            dataset_name="ragbench_cuad",
            sample_size=1000,  # larger than max_examples
            max_examples=5,
        )
        assert len(samples) == 5

    @patch("benchmark.dataset.load_dataset")
    def test_max_examples_from_env(self, mock_load, tmp_path):
        rows = [
            {"id": str(i), "question": f"q{i}", "response": "a", "documents": ["d"]}
            for i in range(20)
        ]
        mock_load.return_value = _mock_dataset(rows)

        with patch.dict("os.environ", {"DATASET_MAX_EXAMPLES": "3"}):
            samples = load_benchmark_data(
                dataset_name="ragbench_cuad",
                sample_size=1000,
            )
        assert len(samples) == 3

    @patch("benchmark.dataset.load_dataset")
    def test_network_error_surfaces_clear_message(self, mock_load, tmp_path):
        mock_load.side_effect = ConnectionError("HF unreachable")
        with pytest.raises(RuntimeError, match="Failed to download dataset"):
            with patch.dict("os.environ", {"DATASET_CACHE_ROOT": str(tmp_path)}):
                load_benchmark_data(dataset_name="ragbench_cuad", sample_size=10)

    @patch("benchmark.dataset.load_dataset")
    def test_non_ragbench_adapter_writes_no_sidecar(self, mock_load, tmp_path):
        mock_load.return_value = _mock_dataset(
            [
                {
                    "question": "q",
                    "program_answer": "42",
                    "pre_text": "ctx",
                }
            ]
        )
        with patch.dict("os.environ", {"DATASET_CACHE_ROOT": str(tmp_path)}):
            load_benchmark_data(dataset_name="t2-ragbench", subset="FinQA", sample_size=5)
        # t2-ragbench is not a ragbench_<component> adapter -> no sidecar.
        assert not (tmp_path / "ragbench").exists()
