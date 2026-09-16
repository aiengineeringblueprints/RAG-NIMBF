"""Ground-truth dataset loaders for the v1 document benchmarks (OCR-05).

Loads OmniDocBench, DP-Bench, and olmOCR-Bench from local files into the
normalized ``{question, ground_truth, context, metadata}`` benchmark sample
format. Downloads are out of scope: callers point ``dataset_path`` at the
downloaded annotation file, so tests and CI run on tiny local fixtures.

License metadata from the dataset adapter registry is attached to every
loaded sample (``dataset_license`` / ``license_research_only``) and to the
``BenchmarkConfig`` (``dataset_license`` / ``dataset_research_only``), which
flows into the reproducibility manifest in run metadata.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from benchmark.dataset_adapters import OCR_GT_DATASET_NAMES, get_adapter
from benchmark.dataset import normalize_sample

# OmniDocBench layout_dets categories whose text counts as page ground truth.
# Tables carry their GT as HTML and are collected separately.
_OMNIDOCBENCH_TEXT_CATEGORIES = {
    "text",
    "text block",
    "title",
    "header",
    "footer",
}

OLMOCR_BENCH_CAPABILITIES: dict[str, Any] = {
    "gt_scoreable": True,
    "gt_key": "expected_text",
    "unit_test_key": "unit_test",
    "capability": (
        "olmOCR-Bench rows carrying an 'expected_text' field provide "
        "plain-text ground truth and can be scored with text metrics."
    ),
    "limitation": (
        "Rows without 'expected_text' (detection-style unit tests such as "
        "header_footer or reading_order) have no plain-text ground truth and "
        "cannot be GT-scored; they are loaded with gt_supported=false and an "
        "empty ground_truth."
    ),
}


def olmocr_bench_capability_statement() -> dict[str, Any]:
    """Return the explicit GT-scoring capability statement for olmOCR-Bench."""
    return dict(OLMOCR_BENCH_CAPABILITIES)


def load_ocr_gt_dataset(
    dataset_name: str,
    dataset_path: str | None = None,
    sample_size: int | None = None,
) -> list[dict[str, Any]]:
    """Load a document ground-truth benchmark into normalized samples."""
    if dataset_name == "omnidocbench":
        rows = _read_json_or_jsonl(_resolve_path(dataset_name, dataset_path))
        samples = _load_omnidocbench(rows)
    elif dataset_name == "dp-bench":
        rows = _read_json_or_jsonl(_resolve_path(dataset_name, dataset_path))
        samples = _load_dp_bench(rows)
    elif dataset_name == "olmocr-bench":
        rows = _read_json_or_jsonl(_resolve_path(dataset_name, dataset_path))
        samples = _load_olmocr_bench(rows)
    else:
        raise ValueError(
            f"Unknown OCR ground-truth dataset '{dataset_name}'. "
            f"Available: {', '.join(OCR_GT_DATASET_NAMES)}"
        )

    license_meta = _license_metadata(dataset_name)
    for sample in samples:
        sample["metadata"] = {**license_meta, **sample["metadata"]}

    if sample_size and sample_size < len(samples):
        samples = samples[:sample_size]
    return samples


def ocr_dataset_license_metadata(dataset_name: str) -> dict[str, Any]:
    """Return the license metadata dict for an OCR GT dataset name."""
    if dataset_name not in OCR_GT_DATASET_NAMES:
        return {}
    return _license_metadata(dataset_name)


def _license_metadata(dataset_name: str) -> dict[str, Any]:
    adapter = get_adapter(dataset_name)
    return {
        "dataset_license": adapter.license,
        "license_research_only": adapter.license_research_only,
    }


def _resolve_path(dataset_name: str, dataset_path: str | None) -> Path:
    path_value = dataset_path or os.getenv("DATASET_PATH")
    if not path_value:
        raise ValueError(
            f"DATASET_PATH is required when DATASET_NAME={dataset_name}"
        )
    path = Path(path_value)
    if not path.exists():
        raise ValueError(f"DATASET_PATH does not exist: {path}")
    return path


# ---------------------------------------------------------------------------
# OmniDocBench v1.6: per-page JSON with per-element GT in layout_dets
# ---------------------------------------------------------------------------


def _load_omnidocbench(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for index, page in enumerate(pages):
        page_info = page.get("page_info") or {}
        image_path = str(page_info.get("image_path", "")) or f"page_{index}"
        texts: list[str] = []
        tables: list[str] = []
        for det in page.get("layout_dets") or []:
            category = str(det.get("category_type", "")).strip().lower()
            html = det.get("html")
            if html or category == "table":
                if html:
                    tables.append(str(html))
                continue
            text = det.get("text")
            if category in _OMNIDOCBENCH_TEXT_CATEGORIES and text:
                texts.append(str(text))
        samples.append(
            normalize_sample(
                {
                    "question": image_path,
                    "ground_truth": "\n\n".join(texts),
                    "context": "",
                    "metadata": {
                        "document_id": image_path,
                        "page_number": page_info.get("page_number", index),
                        "image_path": image_path,
                        "gt_tables": tables,
                    },
                },
                source=f"omnidocbench[{index}]",
            )
        )
    return samples


# ---------------------------------------------------------------------------
# DP-Bench: documents with HTML and/or Markdown ground truth
# ---------------------------------------------------------------------------


def _load_dp_bench(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for index, doc in enumerate(docs):
        document_id = str(
            doc.get("paper_id")
            or doc.get("document_id")
            or doc.get("paper_title")
            or f"doc_{index}"
        )
        markdown = doc.get("gt_markdown") or ""
        html = doc.get("gt_html") or ""
        if markdown:
            ground_truth = str(markdown)
            gt_format = "markdown"
        elif html:
            ground_truth = str(html)
            gt_format = "html"
        else:
            ground_truth = ""
            gt_format = "none"
        samples.append(
            normalize_sample(
                {
                    "question": document_id,
                    "ground_truth": ground_truth,
                    "context": "",
                    "metadata": {
                        "document_id": document_id,
                        "language": doc.get("language"),
                        "filetype": doc.get("filetype"),
                        "gt_format": gt_format,
                        "gt_html": str(html) or None,
                    },
                },
                source=f"dp-bench[{index}]",
            )
        )
    return samples


# ---------------------------------------------------------------------------
# olmOCR-Bench: unit-test specs, GT-scoreable only with expected_text
# ---------------------------------------------------------------------------


def _load_olmocr_bench(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        unit_test = str(row.get("unit_test", f"unit_{index}"))
        expected = row.get("expected_text")
        gt_supported = expected is not None and str(expected) != ""
        samples.append(
            normalize_sample(
                {
                    "question": unit_test,
                    "ground_truth": str(expected) if gt_supported else "",
                    "context": "",
                    "metadata": {
                        "document_id": str(
                            row.get("pdf_path") or f"unit_{index}"
                        ),
                        "unit_test": unit_test,
                        "gt_supported": gt_supported,
                    },
                },
                source=f"olmocr-bench[{index}]",
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Shared file readers
# ---------------------------------------------------------------------------


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array or a JSONL file into a list of dict rows."""
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        data = json.loads(stripped)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON array of pages")
        return [row for row in data if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(stripped.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {path}:{line_number} must be an object")
        rows.append(row)
    return rows
