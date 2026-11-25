"""CLI: ``python -m benchmark.lineage compare --baseline REF --candidate REF``.

REF is a JSON report path (``report.json`` or ``report.json#<config_name>``)
or an MLflow run ID.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmark.lineage.compare import (
    compare_runs,
    default_output_dir,
    log_comparison_to_mlflow,
    write_comparison,
)
from benchmark.lineage.runs import load_run_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmark.lineage")
    sub = parser.add_subparsers(dest="command", required=True)
    cmp = sub.add_parser("compare", help="Paired comparison of two runs")
    cmp.add_argument("--baseline", required=True, help="report path[#config] or MLflow run ID")
    cmp.add_argument("--candidate", required=True, help="report path[#config] or MLflow run ID")
    cmp.add_argument("--out", type=Path, default=None, help="output directory")
    cmp.add_argument("--log-mlflow", action="store_true", help="log result as an MLflow run")
    args = parser.parse_args(argv)

    baseline = load_run_record(args.baseline)
    candidate = load_run_record(args.candidate)
    result = compare_runs(baseline, candidate)
    out_dir = args.out or default_output_dir(baseline, candidate)
    json_path, md_path = write_comparison(result, out_dir)
    print(f"status: {result['status']}")
    for reason in result["reasons"]:
        print(f"  not comparable: {reason}")
    print(f"wrote {md_path} and {json_path}")
    if args.log_mlflow:
        print(f"mlflow run: {log_comparison_to_mlflow(result, json_path, md_path)}")
    return 0 if result["status"] != "not_comparable" else 2


if __name__ == "__main__":
    sys.exit(main())
