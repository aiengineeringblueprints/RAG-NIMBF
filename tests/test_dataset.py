"""Tests for benchmark.dataset — data loading and context building."""

from unittest.mock import patch, MagicMock

from benchmark.dataset import _build_context, load_benchmark_data


class TestBuildContext:
    def test_all_fields(self):
        row = {"pre_text": "before", "table": "table data", "post_text": "after"}
        result = _build_context(row)
        assert "before" in result
        assert "table data" in result
        assert "after" in result

    def test_pre_text_only(self):
        result = _build_context({"pre_text": "hello"})
        assert result == "hello"

    def test_fallback_to_context_field(self):
        result = _build_context({"context": "fallback"})
        assert result == "fallback"

    def test_empty_row(self):
        result = _build_context({})
        assert result == ""


class TestLoadBenchmarkData:
    @patch("benchmark.dataset.load_dataset")
    def test_loads_and_transforms(self, mock_load):
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

        # shuffle().select() chain
        mock_split.shuffle.return_value.select.return_value = mock_split

        samples = load_benchmark_data(subset="FinQA", sample_size=50)

        assert len(samples) == 1
        assert samples[0]["question"] == "What is the revenue?"
        assert samples[0]["ground_truth"] == "42"
