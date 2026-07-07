#!/usr/bin/env python3
"""Clear all data from the MLflow tracking database while preserving the schema."""

import argparse
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "mlflow.db"

# Tables to clear (order matters for foreign key constraints)
TABLES_TO_CLEAR = [
    "metrics",
    "latest_metrics",
    "params",
    "tags",
    "inputs",
    "input_tags",
    "datasets",
    "logged_models",
    "logged_model_metrics",
    "logged_model_params",
    "logged_model_tags",
    "assessments",
    "spans",
    "trace_metrics",
    "span_metrics",
    "trace_info",
    "trace_tags",
    "trace_request_metadata",
    "runs",
    "experiment_tags",
]


def clear_db(db_path: Path, keep_experiments: bool = True) -> None:
    """Delete all data from the MLflow database."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")

    total_deleted = 0
    for table in TABLES_TO_CLEAR:
        try:
            cursor = conn.execute(f"DELETE FROM {table}")
            total_deleted += cursor.rowcount
        except sqlite3.OperationalError:
            # Table might not exist in this schema version
            pass

    if not keep_experiments:
        try:
            cursor = conn.execute("DELETE FROM experiments")
            total_deleted += cursor.rowcount
        except sqlite3.OperationalError:
            pass

    conn.execute("PRAGMA foreign_keys = ON")
    conn.commit()
    conn.close()
    print(f"Cleared {total_deleted} rows from {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear MLflow tracking database")
    parser.add_argument(
        "--db", type=Path, default=DB_PATH, help="Path to mlflow.db (default: ./mlflow.db)"
    )
    parser.add_argument(
        "--reset-experiments", action="store_true",
        help="Also delete experiments (not just runs/metrics/params)",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Database not found: {args.db}")
        raise SystemExit(1)

    if not args.yes:
        response = input(f"Delete all run data from {args.db}? [y/N] ")
        if response.lower() not in ("y", "yes"):
            print("Aborted.")
            return

    clear_db(args.db, keep_experiments=not args.reset_experiments)
    print("Done.")


if __name__ == "__main__":
    main()
