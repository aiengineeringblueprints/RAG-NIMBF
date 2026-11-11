#!/usr/bin/env python3
"""Gold-doc retrieval and answer match per question category and run variant.

Usage:
    python scripts/retrieval_by_category.py results/Enterprise-RAG/run7 \
        --dataset datasets/multihop_rag/questions.jsonl \
        [--category-field question_type] [--k 1 3 5 10] [--csv out.csv]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark.reporting.retrieval_by_category import (  # noqa: E402
    summarize_by_category,
    to_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--category-field", default="question_type")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    rows = summarize_by_category(args.run_dirs, args.dataset, ks=args.k, category_field=args.category_field)
    if not rows:
        sys.exit("No per-sample results found in the given run directories.")
    print(to_markdown(rows, args.k))
    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
