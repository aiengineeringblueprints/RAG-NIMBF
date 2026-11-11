"""Gold-doc retrieval and answer match per question category and run variant.

The runner averages gold-doc metrics over all samples only. This module joins
the per-sample results of one or more run directories with the dataset rows
(by question text) and reports recall@k, complete@k (all gold documents in the
top k) and the answer match rate per category, e.g. MultiHop-RAG's
``question_type`` (inference, comparison, temporal, null).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from benchmark.reporting.exports import load_json_report

ALL = "ALL"


def _load_dataset(path: Path, category_field: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metadata = row.get("metadata") or {}
        category = metadata.get(category_field) or metadata.get("category") or "unknown"
        rows[row["question"]] = {
            "category": str(category),
            "gold": list(metadata.get("gold_doc_ids") or []),
        }
    return rows


def variant_label(config_name: str) -> str:
    """Short label from a config name: retrieval strategy and top-k."""
    hybrid = re.search(r"_hybrid-a([0-9.]+)", config_name)
    strategy = f"hybrid-a{hybrid.group(1)}" if hybrid else "similarity"
    top_k = re.search(r"_k(\d+)(?:_|$)", config_name)
    return f"{strategy} k{top_k.group(1) if top_k else 5}"


def _top_docs(retrieved: list, k: int) -> set[str]:
    return {str(doc_id) for doc_id in retrieved[:k]}


def _load_results(run_dirs: list[Path]) -> dict[str, list[dict]]:
    """Per-sample results by config name; later reports win on duplicates."""
    results: dict[str, list[dict]] = {}
    for run_dir in run_dirs:
        for report_path in sorted(Path(run_dir).glob("benchmark_*.json")):
            report = load_json_report(report_path)
            for result in report.get("results", []):
                if result.get("per_sample"):
                    results[result["config_name"]] = result["per_sample"]
    return results


def summarize_by_category(
    run_dirs: list[Path],
    dataset_path: Path,
    ks: list[int] = (1, 3, 5, 10),
    category_field: str = "question_type",
) -> list[dict]:
    """Return one row per (variant, category) plus an ``ALL`` row per variant."""
    dataset = _load_dataset(dataset_path, category_field)
    rows: list[dict] = []
    for config_name, samples in _load_results(run_dirs).items():
        variant = variant_label(config_name)
        groups: dict[str, list[dict]] = defaultdict(list)
        for sample in samples:
            question = sample["question"]
            if question not in dataset:
                raise ValueError(f"Question not in dataset: {question[:80]!r}")
            entry = dataset[question]
            gold = {str(g) for g in (sample.get("ground_truth_doc_ids") or entry["gold"])}
            retrieved = list(sample.get("retrieved_doc_ids") or [])
            scores: dict[str, float] = {}
            if gold:
                for k in ks:
                    top = _top_docs(retrieved, k)
                    scores[f"recall@{k}"] = len(gold & top) / len(gold)
                    scores[f"complete@{k}"] = float(gold <= top)
            match = (sample.get("adapter_diagnostics") or {}).get("answer_match")
            if match is not None:
                scores["answer_match"] = float(match)
            groups[entry["category"]].append(scores)
            groups[ALL].append(scores)

        for category in sorted(groups, key=lambda c: (c == ALL, c)):
            scored = groups[category]
            row = {"variant": variant, "config_name": config_name, "category": category, "n": len(scored)}
            keys = [f"{m}@{k}" for m in ("recall", "complete") for k in ks] + ["answer_match"]
            for key in keys:
                values = [s[key] for s in scored if key in s]
                row[key] = sum(values) / len(values) if values else None
            rows.append(row)
    return rows


def to_markdown(rows: list[dict], ks: list[int]) -> str:
    columns = ["variant", "category", "n"]
    columns += [f"recall@{k}" for k in ks] + [f"complete@{k}" for k in ks] + ["answer_match"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in sorted(rows, key=lambda r: (r["category"] == ALL, r["category"], r["variant"])):
        cells = []
        for column in columns:
            value = row.get(column)
            cells.append(f"{value:.3f}" if isinstance(value, float) else ("–" if value is None else str(value)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
