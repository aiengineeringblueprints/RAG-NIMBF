"""Tests for OCR ground-truth dataset adapters (OCR-05).

Tiny local fixtures stand in for downloads; no network is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.dataset import load_benchmark_data
from benchmark.dataset_adapters import get_adapter
from benchmark.ocr_datasets import (
    load_ocr_gt_dataset,
    olmocr_bench_capability_statement,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def omnidocbench_path(tmp_path: Path) -> str:
    pages = [
        {
            "page_info": {
                "page_number": 0,
                "image_path": "docs/page_0.jpg",
                "height": 100,
                "width": 100,
            },
            "layout_dets": [
                {"category_type": "title", "text": "Quarterly Report"},
                {"category_type": "text block", "text": "Revenue grew by 4%."},
                {"category_type": "table", "html": "<table><tr><td>Q1</td></tr></table>"},
                {"category_type": "figure", "text": ""},
            ],
        },
        {
            "page_info": {"page_number": 1, "image_path": "docs/page_1.jpg"},
            "layout_dets": [
                {"category_type": "text block", "text": "Second page text."},
            ],
        },
    ]
    path = tmp_path / "OmniDocBench.json"
    path.write_text(json.dumps(pages), encoding="utf-8")
    return str(path)


@pytest.fixture
def dp_bench_path(tmp_path: Path) -> str:
    docs = [
        {
            "paper_id": "doc-1",
            "paper_title": "A Study of Tables",
            "filetype": "PDF",
            "language": "English",
            "gt_markdown": "# A Study of Tables\n\nBody text.",
            "gt_html": "<h1>A Study of Tables</h1><p>Body text.</p>",
        },
        {
            "document_id": "doc-2",
            "filetype": "PPT",
            "language": "German",
            "gt_html": "<p>Nur HTML</p>",
        },
    ]
    path = tmp_path / "dp-bench.json"
    path.write_text(json.dumps(docs), encoding="utf-8")
    return str(path)


@pytest.fixture
def olmocr_bench_path(tmp_path: Path) -> str:
    rows = [
        {
            "unit_test": "old_scan",
            "pdf_path": "units/old_scan/sample.pdf",
            "expected_text": "Expected plain text",
        },
        {
            "unit_test": "header_footer",
            "pdf_path": "units/header_footer/sample.pdf",
        },
    ]
    path = tmp_path / "olmocr-bench.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    return str(path)


# ---------------------------------------------------------------------------
# License metadata (all three datasets)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "license", "research_only"),
    [
        ("omnidocbench", "research-only", True),
        ("dp-bench", "MIT", False),
        ("olmocr-bench", "Apache-2.0", False),
    ],
)
def test_registry_carries_license_metadata(name, license, research_only):
    adapter = get_adapter(name)
    assert adapter.license == license
    assert adapter.license_research_only is research_only


@pytest.mark.parametrize("name", ["omnidocbench", "dp-bench", "olmocr-bench"])
def test_loaded_samples_carry_license_metadata(
    omnidocbench_path, dp_bench_path, olmocr_bench_path, name
):
    paths = {
        "omnidocbench": omnidocbench_path,
        "dp-bench": dp_bench_path,
        "olmocr-bench": olmocr_bench_path,
    }
    samples = load_ocr_gt_dataset(name, dataset_path=paths[name])
    adapter = get_adapter(name)
    for sample in samples:
        assert sample["metadata"]["dataset_license"] == adapter.license
        assert (
            sample["metadata"]["license_research_only"]
            == adapter.license_research_only
        )


# ---------------------------------------------------------------------------
# OmniDocBench
# ---------------------------------------------------------------------------


def test_omnidocbench_loads_text_and_table_gt(omnidocbench_path):
    samples = load_ocr_gt_dataset(
        "omnidocbench", dataset_path=omnidocbench_path
    )

    assert len(samples) == 2
    first = samples[0]
    assert first["question"] == "docs/page_0.jpg"
    assert "Quarterly Report" in first["ground_truth"]
    assert "Revenue grew by 4%." in first["ground_truth"]
    assert first["metadata"]["gt_tables"] == [
        "<table><tr><td>Q1</td></tr></table>"
    ]
    assert first["metadata"]["page_number"] == 0
    assert first["metadata"]["image_path"] == "docs/page_0.jpg"
    assert first["metadata"]["license_research_only"] is True


def test_omnidocbench_respects_sample_size(omnidocbench_path):
    samples = load_ocr_gt_dataset(
        "omnidocbench", dataset_path=omnidocbench_path, sample_size=1
    )
    assert len(samples) == 1


def test_omnidocbench_requires_path():
    with pytest.raises(ValueError, match="omnidocbench"):
        load_ocr_gt_dataset("omnidocbench", dataset_path=None)


# ---------------------------------------------------------------------------
# DP-Bench
# ---------------------------------------------------------------------------


def test_dp_bench_loads_markdown_and_html_gt(dp_bench_path):
    samples = load_ocr_gt_dataset("dp-bench", dataset_path=dp_bench_path)

    assert len(samples) == 2
    first = samples[0]
    assert first["metadata"]["document_id"] == "doc-1"
    assert "# A Study of Tables" in first["ground_truth"]
    assert first["metadata"]["gt_format"] == "markdown"
    assert first["metadata"]["gt_html"].startswith("<h1>")

    second = samples[1]
    assert second["metadata"]["document_id"] == "doc-2"
    assert "Nur HTML" in second["ground_truth"]
    assert second["metadata"]["gt_format"] == "html"


# ---------------------------------------------------------------------------
# olmOCR-Bench
# ---------------------------------------------------------------------------


def test_olmocr_bench_marks_gt_supported_rows(olmocr_bench_path):
    samples = load_ocr_gt_dataset("olmocr-bench", dataset_path=olmocr_bench_path)

    assert len(samples) == 2
    first = samples[0]
    assert first["ground_truth"] == "Expected plain text"
    assert first["metadata"]["gt_supported"] is True
    assert first["metadata"]["unit_test"] == "old_scan"

    second = samples[1]
    assert second["ground_truth"] == ""
    assert second["metadata"]["gt_supported"] is False


def test_olmocr_bench_capability_statement_is_explicit():
    statement = olmocr_bench_capability_statement()

    assert statement["gt_scoreable"] is True
    assert statement["unit_test_key"] == "unit_test"
    assert statement["gt_key"] == "expected_text"
    assert "expected_text" in statement["capability"]
    assert "expected_text" in statement["limitation"]
    assert "gt_supported=false" in statement["limitation"]


# ---------------------------------------------------------------------------
# Dispatch through the existing dataset layer
# ---------------------------------------------------------------------------


def test_load_benchmark_data_dispatches_ocr_datasets(
    omnidocbench_path, dp_bench_path, olmocr_bench_path
):
    samples = load_benchmark_data(
        dataset_name="dp-bench", dataset_path=dp_bench_path, sample_size=50
    )
    assert samples[0]["metadata"]["dataset_license"] == "MIT"


def test_ocr_datasets_reject_missing_file(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        load_ocr_gt_dataset(
            "dp-bench", dataset_path=str(tmp_path / "missing.json")
        )
