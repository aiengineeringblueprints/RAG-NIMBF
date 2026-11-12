"""LLM-judged answer correctness per question category and run variant, from stored runs.

The deterministic ``answer_match`` misses prose answers to yes/no references
(MultiHop-RAG comparison and temporal queries). This module re-scores the
per-sample answers of finished run directories with an
:data:`~benchmark.answer_match.AnswerJudge` and reports, per variant and
category, the share judged correct, incorrect and declined next to the stored
``answer_match``. A sample with ``expected_behavior: decline`` is correct when
the judge sees a decline. Run directories are only read.

Each distinct (question, answer) pair is judged once. With ``cache_path`` the
verdicts are appended to a JSONL file as they arrive, so an interrupted run
resumes without repeating critic calls; unparseable verdicts are not cached and
are retried next time.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark.answer_match import AnswerJudge
from benchmark.reporting.exports import load_json_report
from benchmark.reporting.retrieval_by_category import ALL, variant_label

VERDICTS = ("correct", "incorrect", "declined")


def _load_dataset(path: Path, category_field: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metadata = row.get("metadata") or {}
        rows[row["question"]] = {
            "category": str(metadata.get(category_field) or metadata.get("category") or "unknown"),
            "reference": str(row.get("ground_truth") or ""),
            "decline": metadata.get("expected_behavior") == "decline",
        }
    return rows


def _load_results(run_dirs: list[Path]) -> list[tuple[str, list[dict]]]:
    results = []
    for run_dir in run_dirs:
        for report_path in sorted(Path(run_dir).glob("benchmark_*.json")):
            for result in load_json_report(report_path).get("results", []):
                if result.get("per_sample"):
                    results.append((result["config_name"], result["per_sample"]))
    return results


def _load_cache(path: Path | None) -> dict[tuple[str, str], str]:
    if path is None or not Path(path).exists():
        return {}
    cache = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            cache[(entry["question"], entry["answer"])] = entry["verdict"]
    return cache


def _judge_pairs(
    pairs: list[tuple[str, str]],
    dataset: dict,
    judge: AnswerJudge,
    cache_path: Path | None,
    max_workers: int,
) -> dict[tuple[str, str], str | None]:
    verdicts: dict[tuple[str, str], str | None] = dict(_load_cache(cache_path))
    pending = [pair for pair in pairs if pair not in verdicts]
    lock = threading.Lock()

    def run(pair: tuple[str, str]) -> None:
        question, answer = pair
        verdict = judge(question, dataset[question]["reference"], answer)
        with lock:
            verdicts[pair] = verdict
            if verdict is not None and cache_path is not None:
                with open(cache_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"question": question, "answer": answer, "verdict": verdict}) + "\n")

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        list(pool.map(run, pending))
    return verdicts


def judge_runs(
    run_dirs: list[Path],
    dataset_path: Path,
    judge: AnswerJudge,
    category_field: str = "question_type",
    cache_path: Path | None = None,
    max_workers: int = 1,
) -> list[dict]:
    """Return one row per (variant, category) plus an ``ALL`` row per variant."""
    dataset = _load_dataset(dataset_path, category_field)
    results = _load_results(run_dirs)
    pairs: dict[tuple[str, str], None] = {}
    for _, samples in results:
        for sample in samples:
            if sample["question"] not in dataset:
                raise ValueError(f"Question not in dataset: {sample['question'][:80]!r}")
            pairs[(sample["question"], str(sample.get("answer") or ""))] = None
    verdicts = _judge_pairs(list(pairs), dataset, judge, cache_path, max_workers)

    rows: list[dict] = []
    for config_name, samples in results:
        rows.extend(_rows_for_config(config_name, samples, dataset, verdicts))
    return rows


def _rows_for_config(config_name: str, samples: list[dict], dataset: dict, verdicts: dict) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        question = sample["question"]
        entry = dataset[question]
        verdict = verdicts.get((question, str(sample.get("answer") or "")))
        scores: dict[str, float] = {}
        if verdict is not None:
            for name in VERDICTS:
                scores[f"judge_{name}"] = float(verdict == name)
            if entry["decline"]:
                scores["judge_correct"] = float(verdict == "declined")
                scores["judge_incorrect"] = float(verdict != "declined")
        match = (sample.get("adapter_diagnostics") or {}).get("answer_match")
        if match is not None:
            scores["answer_match"] = float(match)
        groups[entry["category"]].append(scores)
        groups[ALL].append(scores)

    rows = []
    for category in sorted(groups, key=lambda c: (c == ALL, c)):
        scored = groups[category]
        row = {
            "variant": variant_label(config_name),
            "config_name": config_name,
            "category": category,
            "n": len(scored),
            "judged": sum(1 for s in scored if "judge_correct" in s),
        }
        for key in [f"judge_{name}" for name in VERDICTS] + ["answer_match"]:
            values = [s[key] for s in scored if key in s]
            row[key] = sum(values) / len(values) if values else None
        rows.append(row)
    return rows
