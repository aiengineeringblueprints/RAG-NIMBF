"""Parser evaluation runner (OCR-06).

Executes one parsing benchmark cell: a configured parser adapter runs over a
document ground-truth dataset and is scored with the aligned text metrics
(CER/WER/normalized edit distance via the vendored OmniDocBench
``quick_match``) and the TEDS table metrics.

Each parsed document is checkpointed into the run dir so a rerun continues
where the previous attempt stopped. Aggregated per-page details live in the
cell directory; the cross-cell leaderboard is produced by
:func:`generate_parsing_leaderboard`.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.ocr_datasets import load_ocr_gt_dataset
from benchmark.parsing import (
    alignment_run_metadata,
    compute_aligned_parsing_text_metrics,
    compute_parsing_table_metrics,
    get_parser_adapter,
)

__all__ = [
    "generate_parsing_leaderboard",
    "run_parsing_benchmark",
]


def run_parsing_benchmark(config: Any, run_dir: Path) -> dict[str, Any]:
    """Run one parsing cell resumably and return its cell summary."""
    parser = get_parser_adapter(config)
    if parser is None:
        raise ValueError(
            "Parsing benchmarks require a parser adapter "
            "(PARSER_ADAPTER or PARSER_PLUGIN_MODULE)"
        )

    samples = load_ocr_gt_dataset(
        dataset_name=config.dataset_name,
        dataset_path=config.dataset_path,
        sample_size=config.dataset_sample_size or None,
    )

    safe_name = str(config.name).replace(":", "_").replace("/", "_")
    cell_dir = Path(run_dir) / "configs" / f"{safe_name}_parsing"
    cell_dir.mkdir(parents=True, exist_ok=True)

    documents: list[dict[str, Any]] = []
    for sample in samples:
        metadata = sample.get("metadata") or {}
        document_id = str(
            metadata.get("document_id") or sample.get("question") or ""
        )
        checkpoint_path = cell_dir / f"{_safe_filename(document_id)}.json"
        if checkpoint_path.exists():
            documents.append(
                json.loads(checkpoint_path.read_text(encoding="utf-8"))
            )
            continue
        document = _parse_and_score_one(
            parser, config, sample, document_id, metadata
        )
        checkpoint_path.write_text(
            json.dumps(document, indent=2, default=str), encoding="utf-8"
        )
        documents.append(document)

    summary = _summarize_cell(config, parser, documents)
    (cell_dir / "page_details.json").write_text(
        json.dumps(
            {"documents": documents, "metrics": summary["metrics"]},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return summary


def _parse_and_score_one(
    parser: Any,
    config: Any,
    sample: dict[str, Any],
    document_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Parse one document, score it, and return its checkpoint record."""
    page_number = int(metadata.get("page_number", 1) or 1)
    document = {
        "document_id": document_id,
        "pages": [
            {
                "page_number": page_number,
                "image_path": metadata.get("image_path"),
                "image_base64": None,
                "text": None,
            }
        ],
        "metadata": metadata,
    }
    parse_result = parser.parse(document, config)

    gt_supported = bool(metadata.get("gt_supported", True)) and bool(
        str(sample.get("ground_truth") or "").strip()
    )
    per_page: list[dict[str, Any]] = []
    text_scores: dict[str, float] = {}
    table_scores: dict[str, float] = {}

    if gt_supported:
        text = compute_aligned_parsing_text_metrics(
            parse_result, {page_number: str(sample["ground_truth"])}
        )
        text_scores = dict(text.metric_means)
        page_text = {p.page_number: p for p in text.per_page}
        gt_tables = list(metadata.get("gt_tables") or [])
        if not gt_tables and metadata.get("gt_html"):
            gt_tables = [str(metadata["gt_html"])]
        tables = None
        if gt_tables:
            tables = compute_parsing_table_metrics(
                parse_result,
                {page_number: "\n\n".join(gt_tables)},
            )
            table_scores = dict(tables.metric_means)
        page_table = (
            {p.page_number: p for p in tables.per_page} if tables else {}
        )
        for number in sorted(page_text):
            entry: dict[str, Any] = {"page_number": number}
            entry.update(
                {
                    key: getattr(page_text[number], key)
                    for key in ("cer", "wer", "normalized_edit_distance")
                }
            )
            if number in page_table:
                entry["teds"] = page_table[number].teds
                entry["teds_s"] = page_table[number].teds_s
            per_page.append(entry)

    overall = _overall_score(text_scores, table_scores)
    return {
        "document_id": document_id,
        "category": str(metadata.get("category") or "default"),
        "page_number": page_number,
        "gt_supported": gt_supported,
        "parser": parse_result.parser_name or parser.name,
        "parser_version": parse_result.parser_version,
        "per_page": per_page,
        "text_metrics": text_scores,
        "table_metrics": table_scores,
        "overall": overall,
        "parse_seconds": parse_result.total_seconds,
        "status": "completed",
    }


def _overall_score(
    text_scores: dict[str, float], table_scores: dict[str, float]
) -> float | None:
    """Mean of (1 - normalized edit distance) and TEDS over available scores."""
    parts: list[float] = []
    if "normalized_edit_distance" in text_scores:
        parts.append(1.0 - text_scores["normalized_edit_distance"])
    if "teds" in table_scores:
        parts.append(table_scores["teds"])
    return sum(parts) / len(parts) if parts else None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _summarize_cell(
    config: Any, parser: Any, documents: list[dict[str, Any]]
) -> dict[str, Any]:
    scored = [doc for doc in documents if doc.get("gt_supported")]

    def mean_metric(name: str) -> float | None:
        values = [
            doc[source][name]
            for doc in scored
            for source in ("text_metrics", "table_metrics")
            if isinstance(doc.get(source, {}).get(name), (int, float))
        ]
        return _mean(values) if values else None

    metrics = {
        name: value
        for name in (
            "cer",
            "wer",
            "normalized_edit_distance",
            "teds",
            "teds_s",
        )
        if (value := mean_metric(name)) is not None
    }
    overall_values = [
        doc["overall"] for doc in scored if doc.get("overall") is not None
    ]
    if overall_values:
        metrics["overall"] = _mean(overall_values)

    per_category: dict[str, dict[str, Any]] = {}
    for doc in scored:
        bucket = per_category.setdefault(doc["category"], {"num_documents": 0})
        bucket["num_documents"] += 1
        for name, value in {**doc["text_metrics"], **doc["table_metrics"]}.items():
            bucket.setdefault("_accum", {}).setdefault(name, []).append(value)
        if doc.get("overall") is not None:
            bucket.setdefault("_accum", {}).setdefault("overall", []).append(
                doc["overall"]
            )
    category_table = {
        category: {
            "num_documents": bucket["num_documents"],
            "metrics": {
                name: _mean(values)
                for name, values in bucket.get("_accum", {}).items()
            },
        }
        for category, bucket in per_category.items()
    }

    return {
        "config_name": config.name,
        "benchmark_stage": "parsing",
        "parser": getattr(parser, "name", "") or "",
        "parser_version": getattr(parser, "parser_version", None),
        "dataset": {
            "name": config.dataset_name,
            "path": config.dataset_path,
            "license": config.dataset_license,
            "research_only": bool(config.dataset_research_only),
        },
        "match_algorithm": alignment_run_metadata(),
        "num_documents": len(documents),
        "num_scored_documents": len(scored),
        "metrics": metrics,
        "per_category": category_table,
    }


def generate_parsing_leaderboard(
    cell_summaries: list[dict[str, Any]],
    run_dir: Path,
    timestamp: str | None = None,
) -> Path:
    """Aggregate per-cell summaries into a leaderboard report in ``run_dir``."""
    timestamp = timestamp or datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    parsers: dict[str, dict[str, Any]] = {}
    categories: dict[str, dict[str, Any]] = {}
    for summary in cell_summaries:
        parser_name = summary["parser"] or summary["config_name"]
        row = parsers.setdefault(
            parser_name,
            {
                "parser": parser_name,
                "parser_version": summary.get("parser_version"),
                "configs": [],
                "_accum": {},
            },
        )
        row["configs"].append(summary["config_name"])
        for name, value in summary["metrics"].items():
            row["_accum"].setdefault(name, []).append(value)
        for category, entry in summary.get("per_category", {}).items():
            bucket = categories.setdefault(category, {})
            for name, value in entry.get("metrics", {}).items():
                bucket.setdefault(parser_name, {}).setdefault(
                    "_accum", {}
                ).setdefault(name, []).append(value)

    parser_rows = []
    for row in parsers.values():
        metrics = {
            name: _mean(values) for name, values in row.pop("_accum").items()
        }
        parser_rows.append({**row, "metrics": metrics})
    parser_rows.sort(
        key=lambda row: row["metrics"].get("overall", 0.0), reverse=True
    )

    category_table = {
        category: {
            parser_name: {
                "metrics": {
                    name: _mean(values)
                    for name, values in bucket.pop("_accum").items()
                }
            }
            for parser_name, bucket in parsers_map.items()
        }
        for category, parsers_map in categories.items()
    }

    first = cell_summaries[0]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "type": "parsing_leaderboard",
        "match_algorithm": first.get("match_algorithm"),
        "dataset": first.get("dataset"),
        "parsers": parser_rows,
        "per_category": category_table,
    }

    path = Path(run_dir) / f"parsing_leaderboard_{timestamp}.json"
    path.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return path


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "document"
