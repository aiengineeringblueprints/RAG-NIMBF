#!/usr/bin/env python3
"""Shrink legacy benchmark reports into summary JSON + gzipped samples.

Every ``benchmark_*.json`` (and per-config report) that still inlines
``per_sample`` is rewritten losslessly: the JSON keeps the summary and a
``<stem>.samples.json.gz`` sidecar holds the samples with deduplicated
contexts. Each file is verified by reloading it before the original text is
dropped. Already compact files are skipped.

Usage:
    python scripts/compact_results.py [results_dir ...]   # default: results
    python scripts/compact_results.py --dry-run results/run95
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark.reporting.exports import compact_json_report, samples_path  # noqa: E402


def _reports(roots: list[Path]) -> list[Path]:
    files = set()
    for root in roots:
        files.update(root.rglob("benchmark_*.json"))
        files.update(p for p in root.rglob("configs/*.json")
                     if not p.name.endswith(("_qa.json", "_checkpoint.json")))
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("results")])
    parser.add_argument("--dry-run", action="store_true", help="only list candidates")
    args = parser.parse_args()

    before = after = converted = 0
    for path in _reports(args.roots):
        size = path.stat().st_size
        if args.dry_run:
            if '"per_sample"' in path.read_text(encoding="utf-8"):
                print(f"would compact {path} ({size / 1e6:.1f} MB)")
            continue
        if not compact_json_report(path):
            continue
        new = path.stat().st_size + samples_path(path).stat().st_size
        before, after, converted = before + size, after + new, converted + 1
        print(f"{path}: {size / 1e6:.1f} MB -> {new / 1e6:.2f} MB")

    if not args.dry_run:
        print(f"compacted {converted} reports: {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
