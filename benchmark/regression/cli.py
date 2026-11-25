"""CLI: ``python -m benchmark.regression freeze | verify | compare``."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Sequence

from benchmark.regression.compare import compare_runs
from benchmark.regression.criteria import RegressionCriteria
from benchmark.regression.metrics import summarize_run
from benchmark.regression.question_set import (
    SPLITS,
    freeze_question_set,
    jsonl_lines,
    read_jsonl,
    split_status,
)
from benchmark.regression.report import render_report
from benchmark.regression.runs import check_run_configs, find_run_config, load_run_results


def _cmd_freeze(args: argparse.Namespace) -> int:
    from benchmark.dataset import load_benchmark_data

    out = Path(args.out)
    manifest_path = out / "manifest.json"
    existing = sorted(p.name for p in out.iterdir()) if out.exists() else []
    # A committed version carries only manifest.json (the split files are
    # gitignored); freeze then rebuilds them and must reproduce its hashes.
    rebuild = existing == ["manifest.json"]
    if existing and not rebuild:
        print(f"{out} already holds a frozen version; it is immutable — use a new version dir", file=sys.stderr)
        return 1
    # Over-fetch so holdout rows that repeat a dev question can be skipped.
    # A seeded shuffle + prefix select keeps the dev rows identical to runs
    # that loaded ``dev_size`` samples with the same dataset settings.
    requested = args.dev_size + args.holdout_size + args.slack
    samples = load_benchmark_data(
        dataset_name=args.dataset,
        subset=args.subset,
        split=args.split,
        dataset_path=args.dataset_path,
        sample_size=requested,
    )
    if args.dataset_path is None and len(samples) < requested:
        # The HF loader only shuffles when sample_size < split size; a larger
        # request returns source order and the dev rows would not match runs
        # that used the seeded shuffle.
        print(
            f"loaded {len(samples)} < requested {requested}: the loader skipped the seeded "
            f"shuffle; lower --slack so the total is below {len(samples)}",
            file=sys.stderr,
        )
        return 1
    source = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "dataset_path": args.dataset_path,
        "selection": f"loader order (HF: shuffle(seed=42).select(range({len(samples)})))",
        "frozen_on": dt.date.today().isoformat(),
    }
    frozen = freeze_question_set(samples, dev_size=args.dev_size, holdout_size=args.holdout_size, source=source)
    if rebuild:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))["splits"]
        if recorded != frozen.manifest["splits"]:
            print(
                f"rebuilt splits do not match {manifest_path} (source data or freeze arguments changed); "
                "files not written",
                file=sys.stderr,
            )
            return 1
    out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (out / f"{split}.jsonl").write_text(jsonl_lines(getattr(frozen, split)), encoding="utf-8")
    if not rebuild:
        manifest_path.write_text(json.dumps(frozen.manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(frozen.manifest["splits"], indent=2))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    ok = True
    for split in SPLITS:
        status, actual = split_status(args.question_set, split)
        if status == "MISSING":
            print(f"{split}: MISSING ({Path(args.question_set) / f'{split}.jsonl'}); "
                  "rebuild the splits with the freeze command")
        else:
            print(f"{split}: {status} ({actual})")
        ok &= status == "ok"
    return 0 if ok else 1


def _cmd_compare(args: argparse.Namespace) -> int:
    status, actual = split_status(args.question_set, args.split)
    if status != "ok":
        print(
            f"{args.question_set}/{args.split}.jsonl: {status} (sha256 {actual}) does not match the "
            "manifest; the comparison would not be on the frozen question set",
            file=sys.stderr,
        )
        return 1
    question_set = read_jsonl(Path(args.question_set) / f"{args.split}.jsonl")
    criteria = RegressionCriteria.load(args.criteria)
    runs = {"baseline": args.baseline, "best": args.best, "candidate": args.candidate}

    config_check = check_run_configs({name: find_run_config(path) for name, path in runs.items()})
    if config_check["status"] == "invalid":
        for field, values in config_check["mismatches"].items():
            print(f"evaluation setting {field} differs between runs: {values}", file=sys.stderr)
        print("runs were scored differently; refusing to compare", file=sys.stderr)
        return 1
    if config_check["status"] == "unverified":
        print(
            f"warning: no single recorded config for {', '.join(config_check['missing'])}; "
            "judge/dataset consistency not verified",
            file=sys.stderr,
        )

    results = {name: load_run_results(path) for name, path in runs.items()}
    report = {
        "split": args.split,
        "question_set_sha256": actual,
        "runs": runs,
        "criteria": criteria.as_dict(),
        "config_check": config_check,
        "summaries": {name: summarize_run(res, question_set) for name, res in results.items()},
        "comparisons": {
            ref: compare_runs(results[ref], results["candidate"], criteria=criteria, question_set=question_set)
            for ref in ("baseline", "best")
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_report(report), encoding="utf-8")
    for ref, comparison in report["comparisons"].items():
        print(f"candidate vs {ref}: {comparison['verdict']['decision']}")
    print(f"report: {out / 'report.md'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.regression",
        description="Fixed regression benchmark for iterative Blueprint evaluation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze", help="write a versioned dev/holdout question set")
    freeze.add_argument("--dataset", required=True)
    freeze.add_argument("--subset")
    freeze.add_argument("--split")
    freeze.add_argument("--dataset-path")
    freeze.add_argument("--dev-size", type=int, required=True)
    freeze.add_argument("--holdout-size", type=int, required=True)
    freeze.add_argument("--slack", type=int, default=50, help="extra rows loaded to replace holdout duplicates")
    freeze.add_argument("--out", required=True)
    freeze.set_defaults(func=_cmd_freeze)

    verify = sub.add_parser("verify", help="check a frozen question set against its manifest hashes")
    verify.add_argument("--question-set", required=True)
    verify.set_defaults(func=_cmd_verify)

    compare = sub.add_parser("compare", help="judge a candidate run against the baseline and current best")
    compare.add_argument("--question-set", required=True)
    compare.add_argument("--split", choices=SPLITS, default="dev")
    compare.add_argument("--criteria", required=True, help="criteria.yaml (decision policy + bootstrap settings)")
    compare.add_argument("--baseline", required=True, help="immutable reference baseline run report")
    compare.add_argument("--best", required=True, help="current best configuration run report")
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--out", required=True)
    compare.set_defaults(func=_cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)
