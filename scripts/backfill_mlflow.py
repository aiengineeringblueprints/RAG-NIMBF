#!/usr/bin/env python3
"""Backfill EVAL_MATRIX results into MLflow.

Scans results/ directories, matches to EVAL_MATRIX rows by config parameters,
and logs each as a completed MLflow run with original timestamps.
"""
import json
import re
import sys
from pathlib import Path

import mlflow

# Map EVAL_MATRIX LLM names to result JSON model names
LLM_ALIASES = {
    "Qwen3-32B": ["Qwen/Qwen3-32B-AWQ", "Qwen3-32B"],
    "Qwen3.5-397B": ["Qwen3.5-397B", "Sigurd_Qwen3.5-397B", "Sigurd_Qwne3.5-397B"],
    "GPT-OSS-20B": ["GPT-OSS-20B", "gptoss20b", "SPARK_GPToss:20b", "SPARK-gptoss20"],
    "Qwen3.5-35B": ["Qwen3.5-35B"],
}

# Map EVAL_MATRIX retrieval names
RETRIEVAL_ALIASES = {
    "Similarity": "similarity",
    "MMR": "mmr",
}

# Map EVAL_MATRIX chunking names
CHUNKING_ALIASES = {
    "Recursive": "recursive",
    "Semantic": "semantic",
}

# Map EVAL_MATRIX template names
TEMPLATE_ALIASES = {
    "Concise": "concise",
    "Detailed": "detailed",
}


def parse_eval_matrix(path: Path) -> list[dict]:
    """Parse EVAL_MATRIX.md table rows into dicts."""
    content = path.read_text()
    lines = content.split("\n")

    # Find the main matrix table header
    header_idx = None
    for i, line in enumerate(lines):
        if "| #" in line and "LLM" in line and "Chunking" in line:
            header_idx = i
            break

    if header_idx is None:
        print("ERROR: Could not find EVAL_MATRIX table header")
        return []

    # Parse header
    headers = [h.strip() for h in lines[header_idx].split("|")[1:-1]]
    rows = []
    for line in lines[header_idx + 2:]:
        if not line.strip() or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < len(headers):
            continue
        row = dict(zip(headers, cells))
        status = row.get("Status", "").strip()
        if status in ("Getestet", "Test (N=50)"):
            rows.append(row)

    return rows


def load_result_jsons(results_dir: Path) -> list[tuple[Path, dict]]:
    """Load all benchmark JSON files from results/ subdirectories."""
    results = []
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for json_file in subdir.glob("benchmark_*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                results.append((json_file, data))
            except Exception as e:
                print(f"  Warning: Could not load {json_file}: {e}")
    return results


def match_result_to_row(row: dict, json_data: dict) -> bool:
    """Check if a result JSON matches an EVAL_MATRIX row."""
    for result in json_data.get("results", []):
        try:
            if int(result.get("chunk_size", 0)) != int(row.get("Size", 0)):
                continue
            if int(result.get("chunk_overlap", 0)) != int(row.get("Overlap", 0)):
                continue

            chunking = row.get("Chunking", "")
            result_chunking = result.get("chunking_strategy", "")
            if CHUNKING_ALIASES.get(chunking, chunking.lower()) != result_chunking:
                continue

            template = row.get("Template", "")
            result_template = result.get("prompt_template", "")
            if TEMPLATE_ALIASES.get(template, template.lower()) != result_template:
                continue

            retrieval = row.get("Retrieval", "")
            config_name = result.get("config_name", "").lower()
            retrieval_lower = RETRIEVAL_ALIASES.get(retrieval, retrieval.lower())
            if retrieval_lower == "mmr" and "mmr" not in config_name:
                continue
            if retrieval_lower == "similarity" and "mmr" in config_name:
                continue

            return True
        except (ValueError, TypeError):
            continue
    return False


def log_backfill_run(row: dict, json_path: Path, json_data: dict) -> bool:
    """Log a single backfill run to MLflow."""
    result = None
    for r in json_data.get("results", []):
        if match_result_to_row(row, {"results": [r]}):
            result = r
            break

    if result is None:
        return False

    llm_name = row.get("LLM", "unknown").split("/")[-1]
    chunking = row.get("Chunking", "?")
    size = row.get("Size", "?")
    overlap = row.get("Overlap", "?")
    retrieval = row.get("Retrieval", "?")
    template = row.get("Template", "?")
    run_name = f"{llm_name}_{chunking}_cs{size}_co{overlap}_{retrieval}_{template}"

    ts_match = re.search(r"benchmark_(\d{8})_(\d{6})", json_path.name)
    start_time_ms = None
    if ts_match:
        from datetime import datetime
        dt = datetime.strptime(f"{ts_match.group(1)}_{ts_match.group(2)}", "%Y%m%d_%H%M%S")
        start_time_ms = int(dt.timestamp() * 1000)

    tags = {
        "llm_model": result.get("llm_model", ""),
        "embedding_model": result.get("embedding_model", ""),
        "chunking_strategy": result.get("chunking_strategy", ""),
        "prompt_template": result.get("prompt_template", ""),
        "source": "backfill",
    }

    params = {
        "chunk_size": result.get("chunk_size", 0),
        "chunk_overlap": result.get("chunk_overlap", 0),
        "num_chunks": result.get("num_chunks", 0),
        "num_questions": result.get("num_questions", 0),
    }

    metrics = {}
    val = result.get("faithfulness")
    if val is not None:
        metrics["ragas_faithfulness"] = val

    stats = result.get("stats", {})
    for metric_name, stat_data in stats.items():
        if stat_data and isinstance(stat_data, dict):
            for stat_key in ["mean", "std", "median", "min", "max"]:
                val = stat_data.get(stat_key)
                if val is not None:
                    safe_name = metric_name.replace("@", "_at_")
                    metrics[f"ragas_{safe_name}_{stat_key}"] = val

    custom_keys = [
        "hit@1", "hit@3", "hit@5", "ndcg@1", "ndcg@3", "ndcg@5",
        "recall@1", "recall@3", "recall@5", "context_relevance",
        "rouge_l", "bleu", "meteor",
        "bert_score_precision", "bert_score_recall", "bert_score_f1",
    ]
    for key in custom_keys:
        val = result.get(key)
        if val is not None:
            safe = key.replace("@", "_at_")
            metrics[f"custom_{safe}"] = val

    custom_stats = result.get("custom_stats", {})
    if custom_stats and isinstance(custom_stats, dict):
        for metric_name, stat_data in custom_stats.items():
            if stat_data and isinstance(stat_data, dict):
                safe = metric_name.replace("@", "_at_")
                for stat_key in ["mean", "std", "median", "min", "max"]:
                    val = stat_data.get(stat_key)
                    if val is not None:
                        metrics[f"custom_{safe}_{stat_key}"] = val

    existing = mlflow.search_runs(
        filter_string=f"tags.source = 'backfill' AND tags.mlflow.runName = '{run_name}'",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
    )
    if not existing.empty:
        print(f"  SKIP (already imported): {run_name}")
        return False

    from mlflow import MlflowClient

    experiment = mlflow.get_experiment_by_name("RAG-Benchmark")
    client = MlflowClient()
    run = client.create_run(
        experiment_id=experiment.experiment_id,
        start_time=start_time_ms,
        tags=tags,
        run_name=run_name,
    )
    run_id = run.info.run_id

    client.log_batch(run_id, params=[mlflow.entities.Param(k, str(v)) for k, v in params.items()])
    client.log_batch(run_id, metrics=[mlflow.entities.Metric(k, v, 0, 0) for k, v in metrics.items()])

    per_sample = result.get("per_sample", [])
    if per_sample:
        import csv
        import tempfile

        tmpdir = Path(tempfile.mkdtemp())
        csv_path = tmpdir / "per_sample_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "ground_truth", "answer"])
            for s in per_sample:
                writer.writerow([
                    s.get("question", ""),
                    s.get("ground_truth", ""),
                    s.get("answer", ""),
                ])
        client.log_artifact(run_id, str(csv_path), artifact_path="data")

    client.set_terminated(run_id, status="FINISHED", end_time=(start_time_ms or 0) + 1)

    print(f"  IMPORTED: {run_name} ({len(metrics)} metrics)")
    return True


def main():
    project_root = Path(__file__).parent
    eval_matrix_path = project_root / "EVAL_MATRIX.md"
    results_dir = project_root / "results"

    if not eval_matrix_path.exists():
        print(f"ERROR: {eval_matrix_path} not found")
        sys.exit(1)

    tracking_uri = "http://localhost:5000"
    try:
        import requests
        requests.get(f"{tracking_uri}/api/2.0/mlflow/experiments/search", timeout=2)
    except Exception:
        tracking_uri = "sqlite:///mlflow.db"
        print(f"MLflow server unreachable, using SQLite: {tracking_uri}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("RAG-Benchmark")

    print("Parsing EVAL_MATRIX...")
    rows = parse_eval_matrix(eval_matrix_path)
    print(f"Found {len(rows)} tested rows")

    print("Loading result JSONs...")
    result_jsons = load_result_jsons(results_dir)
    print(f"Found {len(result_jsons)} result files")

    imported = 0
    skipped = 0

    for row in rows:
        row_num = row.get("#", "?")
        llm = row.get("LLM", "")
        chunking = row.get("Chunking", "")
        size = row.get("Size", "")
        overlap = row.get("Overlap", "")
        faith = row.get("Faith", "-")

        if faith == "-" or faith == "":
            print(f"  SKIP row {row_num}: no faithfulness value")
            skipped += 1
            continue

        matched = False
        for json_path, json_data in result_jsons:
            if match_result_to_row(row, json_data):
                if log_backfill_run(row, json_path, json_data):
                    imported += 1
                else:
                    skipped += 1
                matched = True
                break

        if not matched:
            print(f"  SKIP row {row_num} ({llm} {chunking} cs{size}): no matching result file")
            skipped += 1

    print(f"\nDone. Imported: {imported}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
