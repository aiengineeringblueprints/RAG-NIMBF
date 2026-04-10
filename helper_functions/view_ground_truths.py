#!/usr/bin/env python3
"""Extract and display ground truth answers from benchmark JSON files."""

import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def find_json_files(run_name: str | None = None) -> list[Path]:
    """Find benchmark JSON files, optionally filtered by run name."""
    if run_name:
        pattern = f"{run_name}/benchmark_*.json"
    else:
        pattern = "run*/benchmark_*.json"

    files = sorted(RESULTS_DIR.glob(pattern))
    if not files:
        print(f"Keine Benchmark-Dateien gefunden in: {RESULTS_DIR / (run_name or '.')}")
        sys.exit(1)
    return files


def load_ground_truths(file_path: Path) -> dict:
    """Load ground truths grouped by config from a benchmark JSON."""
    with open(file_path) as f:
        data = json.load(f)

    configs = {}
    for result in data.get("results", []):
        config_name = result.get("config_name", "unknown")
        samples = []
        for sample in result.get("per_sample", []):
            samples.append({
                "question": sample.get("question", ""),
                "ground_truth": sample.get("ground_truth", ""),
            })
        configs[config_name] = samples

    return {
        "file": file_path.name,
        "run": file_path.parent.name,
        "dataset": data.get("dataset", "unknown"),
        "configs": configs,
    }


def print_ground_truths(data: dict, show_index: bool = True):
    """Print ground truths in a readable format."""
    print(f"\n{'=' * 80}")
    print(f"  Run: {data['run']}  |  Datei: {data['file']}")
    print(f"  Dataset: {data['dataset']}")
    print(f"{'=' * 80}")

    for config_name, samples in data["configs"].items():
        print(f"\n{'─' * 60}")
        print(f"  Config: {config_name}")
        print(f"  Anzahl Questions: {len(samples)}")
        print(f"{'─' * 60}")

        for i, sample in enumerate(samples, 1):
            prefix = f"  [{i}] " if show_index else "  "
            question = sample["question"]
            gt = str(sample["ground_truth"])

            print(f"\n{prefix}Q: {question}")
            print(f"{' ' * len(prefix)}A: {gt}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Ground Truth Antworten aus Benchmark-JSONs anzeigen",
    )
    parser.add_argument(
        "run",
        nargs="?",
        default=None,
        help="Run-Ordner (z.B. run1, run5). Ohne Angabe werden alle Runs angezeigt.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Als JSON ausgeben statt formatiertem Text.",
    )

    args = parser.parse_args()
    files = find_json_files(args.run)

    if args.json:
        output = []
        for f in files:
            output.append(load_ground_truths(f))
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        for f in files:
            data = load_ground_truths(f)
            print_ground_truths(data)


if __name__ == "__main__":
    main()
