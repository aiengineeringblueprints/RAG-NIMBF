#!/usr/bin/env python3
"""Query and display all MLflow runs from mlflow.db as a formatted table."""

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "mlflow.db"

# Key metric columns to show by default (main values, not stats)
DEFAULT_METRICS = [
    "ragas_answer_relevancy",
    "ragas_answer_correctness",
    "ragas_context_precision",
    "ragas_context_recall",
    "ragas_faithfulness",
    "avg_tokens_per_second",
    "avg_ttft_seconds",
    "total_time_seconds",
]

TAG_COLUMNS = ["chunking_strategy", "embedding_model", "llm_model", "prompt_template"]
PARAM_COLUMNS = ["chunk_size", "chunk_overlap", "num_chunks", "num_questions"]


def _ts_to_str(ts: int | None) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _fetch_runs(conn: sqlite3.Connection, experiment_id: int | None) -> list[dict]:
    """Fetch all runs with their params, metrics, and tags."""
    where = "" if experiment_id is None else f"WHERE r.experiment_id = {experiment_id}"
    cursor = conn.execute(f"""
        SELECT r.run_uuid, r.name, r.status, r.start_time, r.end_time,
               r.experiment_id
        FROM runs r
        {where}
        ORDER BY r.start_time DESC
    """)

    runs = []
    for row in cursor.fetchall():
        run_uuid, name, status, start_time, end_time, exp_id = row
        runs.append({
            "run_uuid": run_uuid,
            "name": name or "-",
            "status": status,
            "start_time": _ts_to_str(start_time),
            "end_time": _ts_to_str(end_time),
            "experiment_id": exp_id,
        })
    return runs


def _enrich_runs(conn: sqlite3.Connection, runs: list[dict]) -> list[dict]:
    """Add params, metrics, and tags to each run."""
    if not runs:
        return runs

    uuids = [r["run_uuid"] for r in runs]
    placeholders = ",".join("?" for _ in uuids)

    # Fetch all params
    params_rows = conn.execute(
        f"SELECT run_uuid, key, value FROM params WHERE run_uuid IN ({placeholders})",
        uuids,
    ).fetchall()
    params_map: dict[str, dict[str, str]] = {}
    for run_uuid, key, value in params_rows:
        params_map.setdefault(run_uuid, {})[key] = value

    # Fetch all metrics (latest step per key)
    metrics_rows = conn.execute(
        f"SELECT run_uuid, key, value FROM latest_metrics WHERE run_uuid IN ({placeholders})",
        uuids,
    ).fetchall()
    metrics_map: dict[str, dict[str, float]] = {}
    for run_uuid, key, value in metrics_rows:
        metrics_map.setdefault(run_uuid, {})[key] = value

    # Fetch all tags
    tags_rows = conn.execute(
        f"SELECT run_uuid, key, value FROM tags WHERE run_uuid IN ({placeholders})",
        uuids,
    ).fetchall()
    tags_map: dict[str, dict[str, str]] = {}
    for run_uuid, key, value in tags_rows:
        tags_map.setdefault(run_uuid, {})[key] = value

    # Merge into runs
    for run in runs:
        uid = run["run_uuid"]
        run["params"] = params_map.get(uid, {})
        run["metrics"] = metrics_map.get(uid, {})
        run["tags"] = tags_map.get(uid, {})

    return runs


def _format_val(val: str | float | None) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _print_table(runs: list[dict], metrics_keys: list[str]) -> None:
    """Print runs as a formatted terminal table."""
    if not runs:
        print("No runs found.")
        return

    # Build columns: fixed cols + tags + params + metrics
    fixed_headers = ["#", "Name", "Status", "Started"]
    fixed_getters = [
        lambda r, i=[-1]: (i.__setitem__(0, i[0] + 1) or str(i[0])),
        lambda r: r["name"],
        lambda r: r["status"],
        lambda r: r["start_time"],
    ]

    tag_getters = [lambda r, k=key: r["tags"].get(k, "-") for key in TAG_COLUMNS]
    param_getters = [lambda r, k=key: r["params"].get(k, "-") for key in PARAM_COLUMNS]
    metric_getters = [lambda r, k=key: _format_val(r["metrics"].get(k)) for key in metrics_keys]

    headers = fixed_headers + TAG_COLUMNS + PARAM_COLUMNS + metrics_keys
    getters = fixed_getters + tag_getters + param_getters + metric_getters

    # Build rows
    rows: list[list[str]] = []
    for run in runs:
        rows.append([g(run) for g in getters])

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Limit column widths to 50 chars
    col_widths = [min(w, 50) for w in col_widths]

    # Print header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)
    print(header_line)
    print(separator)

    # Print rows
    for row in rows:
        line = " | ".join(cell.ljust(w)[:w] for cell, w in zip(row, col_widths))
        print(line)

    print(f"\nTotal runs: {len(runs)}")


def _print_comparison(runs: list[dict], metrics_keys: list[str]) -> None:
    """Print a compact comparison table sorted by best metric."""
    if not runs:
        print("No runs found.")
        return

    # Only show key identifying columns + metrics
    headers = ["LLM Model", "Embedding", "Strategy", "CS", "CO", "Prompt"] + metrics_keys
    rows: list[list[str]] = []

    for run in runs:
        tags = run["tags"]
        params = run["params"]
        metrics = run["metrics"]
        row = [
            tags.get("llm_model", "-"),
            tags.get("embedding_model", "-"),
            tags.get("chunking_strategy", "-"),
            params.get("chunk_size", "-"),
            params.get("chunk_overlap", "-"),
            tags.get("prompt_template", "-"),
        ]
        row.extend(_format_val(metrics.get(k)) for k in metrics_keys)
        rows.append(row)

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    col_widths = [min(w, 50) for w in col_widths]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)
    print(header_line)
    print(separator)

    for row in rows:
        line = " | ".join(cell.ljust(w)[:w] for cell, w in zip(row, col_widths))
        print(line)

    print(f"\nTotal runs: {len(runs)}")


def _export_csv(runs: list[dict], metrics_keys: list[str], output_path: Path) -> None:
    """Export runs to CSV."""
    import csv

    headers = ["run_uuid", "name", "status", "start_time", "experiment_id"]
    headers += TAG_COLUMNS + PARAM_COLUMNS + metrics_keys

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for run in runs:
            row = [
                run["run_uuid"],
                run["name"],
                run["status"],
                run["start_time"],
                run["experiment_id"],
            ]
            row += [run["tags"].get(k, "") for k in TAG_COLUMNS]
            row += [run["params"].get(k, "") for k in PARAM_COLUMNS]
            row += [_format_val(run["metrics"].get(k)) for k in metrics_keys]
            writer.writerow(row)

    print(f"Exported {len(runs)} runs to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="View MLflow runs from local database")
    parser.add_argument(
        "--db", type=Path, default=DB_PATH, help="Path to mlflow.db (default: ./mlflow.db)"
    )
    parser.add_argument(
        "--experiment", "-e", type=int, default=None, help="Filter by experiment ID"
    )
    parser.add_argument(
        "--compare", "-c", action="store_true",
        help="Compact comparison view (key config + metrics only)",
    )
    parser.add_argument(
        "--csv", type=Path, default=None, help="Export results to CSV file"
    )
    parser.add_argument(
        "--all-metrics", "-a", action="store_true",
        help="Show all metrics instead of default summary metrics",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Database not found: {args.db}")
        raise SystemExit(1)

    conn = sqlite3.connect(args.db)

    # List experiments
    experiments = conn.execute("SELECT experiment_id, name FROM experiments").fetchall()
    print("Experiments:")
    for exp_id, name in experiments:
        print(f"  [{exp_id}] {name}")
    print()

    runs = _fetch_runs(conn, args.experiment)
    runs = _enrich_runs(conn, runs)

    # Determine which metrics to show
    if args.all_metrics:
        all_metric_keys: set[str] = set()
        for run in runs:
            all_metric_keys.update(run["metrics"].keys())
        # Remove stat variants (keep only base metrics)
        metrics_keys = sorted(
            k for k in all_metric_keys
            if not any(k.endswith(suffix) for suffix in ("_max", "_min", "_mean", "_median", "_std"))
        )
    else:
        metrics_keys = DEFAULT_METRICS

    if args.csv:
        _export_csv(runs, metrics_keys, args.csv)
    elif args.compare:
        _print_comparison(runs, metrics_keys)
    else:
        _print_table(runs, metrics_keys)

    conn.close()


if __name__ == "__main__":
    main()
