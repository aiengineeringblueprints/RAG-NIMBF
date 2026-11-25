"""Fixed regression benchmark for iterative Blueprint evaluation (issue #64).

Three steps, each a pure function so they can be tested without a model:

- :func:`freeze_question_set` turns loaded benchmark samples into a versioned
  dev split (the shared comparison set) and a disjoint holdout split (final
  confirmation only), with a content-hashed manifest.
- :func:`summarize_run` reduces one run's per-sample results to the issue's
  evaluation dimensions, reporting a dimension as ``unavailable`` instead of
  guessing when its labels or scores are missing.
- :func:`compare_runs` pairs a candidate with a reference run question by
  question and judges it against explicit improvement / trade-off criteria.

CLI (guide: ``doc/blueprint-regression-benchmark.md``)::

    python -m benchmark.regression freeze --dataset ragbench --subset techqa \
        --split train --dev-size 500 --holdout-size 200 \
        --out datasets/blueprint_regression/v1
    python -m benchmark.regression verify --question-set datasets/blueprint_regression/v1
    python -m benchmark.regression compare --question-set datasets/blueprint_regression/v1 \
        --criteria experiments/blueprint-regression/criteria.yaml \
        --baseline <report.json> [--best <report.json>] --candidate <report.json> \
        --out results/<run>/regression
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from benchmark.answer_match import declines

# Loader-stamped document IDs. They are re-derived from the row position when
# the frozen file is loaded again (``dataset-doc-{index}``), so storing them
# would only risk a stale value.
_LOADER_METADATA_KEYS = ("doc_id", "gold_doc_id")


@dataclass(frozen=True)
class FrozenQuestionSet:
    dev: list[dict[str, Any]]
    holdout: list[dict[str, Any]]
    manifest: dict[str, Any]


def _category(metadata: Mapping[str, Any]) -> str:
    answerable = metadata.get("answerable")
    if answerable is None:
        return "unlabeled"
    return "answerable" if answerable else "unanswerable"


def _frozen_row(sample: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        key: value
        for key, value in (sample.get("metadata") or {}).items()
        if key not in _LOADER_METADATA_KEYS
    }
    metadata["category"] = _category(metadata)
    return {
        "question": sample["question"],
        "ground_truth": sample["ground_truth"],
        "context": sample["context"],
        "metadata": metadata,
    }


def jsonl_lines(rows: Sequence[Mapping[str, Any]]) -> str:
    """Canonical JSONL text of ``rows``; the hashed and written form."""
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def _split_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "size": len(rows),
        "categories": dict(Counter(row["metadata"]["category"] for row in rows)),
        "sha256": hashlib.sha256(jsonl_lines(rows).encode("utf-8")).hexdigest(),
    }


def freeze_question_set(
    samples: Sequence[Mapping[str, Any]],
    *,
    dev_size: int,
    holdout_size: int,
    source: Mapping[str, Any],
) -> FrozenQuestionSet:
    """Split loaded samples into a fixed dev set and a disjoint holdout set.

    ``dev`` is the first ``dev_size`` samples, so a frozen set built from a
    seeded shuffle keeps the questions earlier runs with the same seed and
    sample size already used. ``holdout`` takes the following samples whose
    question text does not occur in ``dev``.
    """
    if len(samples) < dev_size:
        raise ValueError(f"need {dev_size} dev samples, source has {len(samples)}")
    dev = [_frozen_row(sample) for sample in samples[:dev_size]]
    dev_questions = {row["question"] for row in dev}

    holdout: list[dict[str, Any]] = []
    for sample in samples[dev_size:]:
        if len(holdout) == holdout_size:
            break
        if sample["question"] in dev_questions:
            continue
        holdout.append(_frozen_row(sample))
    if len(holdout) < holdout_size:
        raise ValueError(
            f"need {holdout_size} holdout samples disjoint from dev, "
            f"source has {len(holdout)}; load more samples"
        )

    manifest = {
        "source": dict(source),
        "splits": {"dev": _split_manifest(dev), "holdout": _split_manifest(holdout)},
    }
    return FrozenQuestionSet(dev=dev, holdout=holdout, manifest=manifest)


# --------------------------------------------------------------------------
# Per-sample metrics and run summary
# --------------------------------------------------------------------------

# Gold-label retrieval scores in ``custom_scores`` (benchmark.gold_retrieval_metrics
# and the RAGBench relevant-sentence recall).
_RETRIEVAL_PREFIXES = (
    "hit@",
    "recall@",
    "ndcg@",
    "mrr",
    "relevant_sentence_recall",
    "supporting_fact_coverage@",
)

# Per-sample metrics where a lower value is better; every other metric is
# higher-is-better.
LOWER_IS_BETTER = frozenset(
    {"false_refusal", "total_seconds", "ttft_seconds", "estimated_cost_usd", "total_tokens"}
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)) or math.isnan(value):
        return None
    return float(value)


SampleKey = tuple[str, str, int]


def sample_keys(records: Sequence[Mapping[str, Any]]) -> list[SampleKey]:
    """Identity of each record: question, reference answer, occurrence.

    The question text alone is not unique: RAGBench TechQA lists every
    question twice with the same documents but a different reference answer
    (and sometimes a different answerable label). The occurrence number
    keeps exact duplicates apart. Works for frozen rows and result records.
    """
    seen: Counter[tuple[str, str]] = Counter()
    keys = []
    for record in records:
        base = (record["question"], str(record.get("ground_truth") or ""))
        keys.append((*base, seen[base]))
        seen[base] += 1
    return keys


def question_categories(question_set: Sequence[Mapping[str, Any]] | None) -> dict[SampleKey, str]:
    """Map each frozen sample to its category (answerable/unanswerable)."""
    rows = list(question_set or ())
    return {
        key: (row.get("metadata") or {}).get("category", "unlabeled")
        for key, row in zip(sample_keys(rows), rows)
    }


def sample_metrics(result: Mapping[str, Any], category: str = "unlabeled") -> dict[str, float | None]:
    """Per-sample metric values of one result record; ``None`` = not scored.

    Retrieval metrics count only when the sample has gold document IDs and
    is not known to be unanswerable. Abstention is split by category:
    ``abstained`` (declining an unanswerable question is correct) and
    ``false_refusal`` (declining an answerable one is an error).
    """
    ragas = result.get("ragas_scores") or {}
    custom = result.get("custom_scores") or {}
    metrics: dict[str, float | None] = {
        "answer_correctness": _number(ragas.get("answer_correctness")),
        "bert_score_f1": _number(custom.get("bert_score_f1")),
        "faithfulness": _number(ragas.get("faithfulness")),
    }
    if result.get("ground_truth_doc_ids") and category != "unanswerable":
        for name, value in custom.items():
            if name.startswith(_RETRIEVAL_PREFIXES):
                metrics[name] = _number(value)
    refused = float(declines(result.get("answer") or ""))
    metrics["abstained"] = refused if category == "unanswerable" else None
    metrics["false_refusal"] = refused if category == "answerable" else None
    metrics["total_seconds"] = _number(result.get("total_seconds"))
    metrics["ttft_seconds"] = _number(result.get("ttft_seconds"))
    metrics["estimated_cost_usd"] = _number(result.get("estimated_cost_usd"))
    # Adapters that do not report usage write 0 tokens; that is "unknown".
    metrics["total_tokens"] = _number(result.get("total_tokens")) or None
    return metrics


def _values(per_sample: list[dict[str, float | None]], name: str) -> list[float]:
    return [m[name] for m in per_sample if m.get(name) is not None]


def _p95(values: list[float]) -> float:
    """Nearest-rank 95th percentile."""
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _dimension(metrics: dict[str, float], reason: str) -> dict[str, Any]:
    if metrics:
        return {"status": "ok", "metrics": metrics}
    return {"status": "unavailable", "reason": reason, "metrics": {}}


def summarize_run(
    results: Sequence[Mapping[str, Any]],
    question_set: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reduce one run's per-sample results to the evaluation dimensions."""
    categories = question_categories(question_set)
    per_sample = [
        sample_metrics(result, categories.get(key, "unlabeled"))
        for key, result in zip(sample_keys(results), results)
    ]

    def means(names: Sequence[str]) -> dict[str, float]:
        return {
            name: statistics.fmean(values)
            for name in names
            if (values := _values(per_sample, name))
        }

    retrieval_names = sorted({n for m in per_sample for n in m if n.startswith(_RETRIEVAL_PREFIXES)})

    abstention: dict[str, float] = {}
    if abstained := _values(per_sample, "abstained"):
        abstention["abstention_rate"] = statistics.fmean(abstained)
        abstention["n_unanswerable"] = len(abstained)
    if false_refusals := _values(per_sample, "false_refusal"):
        abstention["false_refusal_rate"] = statistics.fmean(false_refusals)

    latency: dict[str, float] = {}
    if seconds := _values(per_sample, "total_seconds"):
        latency = {
            "total_seconds_mean": statistics.fmean(seconds),
            "total_seconds_median": statistics.median(seconds),
            "total_seconds_p95": _p95(seconds),
        }
        latency.update({f"{k}_mean": v for k, v in means(["ttft_seconds"]).items()})

    cost: dict[str, float] = {}
    if costs := _values(per_sample, "estimated_cost_usd"):
        cost["estimated_cost_usd_total"] = sum(costs)
    if tokens := _values(per_sample, "total_tokens"):
        cost["total_tokens_mean"] = statistics.fmean(tokens)

    return {
        "n": len(results),
        "dimensions": {
            "correctness": _dimension(
                means(["answer_correctness", "bert_score_f1"]),
                "no answer_correctness (RAGAS) or bert_score_f1 scores",
            ),
            "faithfulness": _dimension(
                means(["faithfulness"]), "RAGAS faithfulness not scored (RAGAS disabled?)"
            ),
            "retrieval": _dimension(
                means(retrieval_names),
                "no gold evidence labels (ground_truth_doc_ids) on answerable samples",
            ),
            "abstention": _dimension(
                abstention, "no unanswerable questions labelled in the question set"
            ),
            "latency": _dimension(latency, "no total_seconds recorded"),
            "cost": _dimension(cost, "adapter reports neither token usage nor cost"),
        },
    }


# --------------------------------------------------------------------------
# Paired comparison and verdict
# --------------------------------------------------------------------------


def _bootstrap_ci(
    diffs: "np.ndarray", clusters: Sequence[str], n_boot: int, seed: int
) -> tuple[float, float]:
    """Paired cluster bootstrap: resample questions, not rows.

    TechQA repeats each question with a second reference answer; both copies
    share documents and retrieval, so they are not independent evidence.
    Resampling whole clusters keeps the CI from shrinking with duplicates.
    """
    groups: dict[str, list[int]] = {}
    for i, cluster in enumerate(clusters):
        groups.setdefault(cluster, []).append(i)
    sums = np.array([diffs[ix].sum() for ix in groups.values()])
    counts = np.array([len(ix) for ix in groups.values()])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    boot_means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)


def _mcnemar(baseline: list[float], candidate: list[float]) -> dict[str, Any]:
    """Exact two-sided McNemar test on paired 0/1 outcomes."""
    only_candidate = sum(1 for b, c in zip(baseline, candidate) if not b and c)
    only_baseline = sum(1 for b, c in zip(baseline, candidate) if b and not c)
    n = only_candidate + only_baseline
    k = min(only_candidate, only_baseline)
    p_value = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return {"only_candidate": only_candidate, "only_baseline": only_baseline, "p_value": p_value}


def _metric_comparison(
    baseline: list[float],
    candidate: list[float],
    clusters: Sequence[str],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    diffs = np.asarray(candidate) - np.asarray(baseline)
    base_mean = statistics.fmean(baseline)
    entry: dict[str, Any] = {
        "n": len(diffs),
        "baseline": base_mean,
        "candidate": statistics.fmean(candidate),
        "delta": float(diffs.mean()),
        "delta_pct": (float(diffs.mean()) / base_mean * 100) if base_mean else None,
        "ci95": _bootstrap_ci(diffs, clusters, n_boot, seed),
    }
    if set(baseline) | set(candidate) <= {0.0, 1.0}:
        entry["mcnemar"] = _mcnemar(baseline, candidate)
    return entry


def _worsening(name: str, value: float) -> float:
    """Signed change in the *worse* direction (positive = worse)."""
    return value if name in LOWER_IS_BETTER else -value


def _verdict(metrics: Mapping[str, Mapping[str, Any]], criteria: Mapping[str, Any]) -> dict[str, Any]:
    primary = next((m for m in criteria.get("primary", ()) if m in metrics), None)
    reasons: list[str] = []
    skipped: list[str] = []

    breached = False
    for name, limits in (criteria.get("guardrails") or {}).items():
        entry = metrics.get(name)
        if entry is None:
            skipped.append(name)
            continue
        worse = _worsening(name, entry["delta"])
        if "max_worsening" in limits and worse > limits["max_worsening"]:
            breached = True
            reasons.append(
                f"{name} worsened by {worse:.4f} (> {limits['max_worsening']})"
            )
        if "max_worsening_pct" in limits and entry["delta_pct"] is not None:
            worse_pct = _worsening(name, entry["delta_pct"])
            if worse_pct > limits["max_worsening_pct"]:
                breached = True
                reasons.append(
                    f"{name} worsened by {worse_pct:.1f}% (> {limits['max_worsening_pct']}%)"
                )

    if primary is None:
        return {
            "decision": "reject" if breached else "inconclusive",
            "primary": None,
            "primary_fallback": False,
            "reasons": reasons + ["no primary metric available in both runs"],
            "skipped_guardrails": skipped,
        }
    primary_fallback = primary != next(iter(criteria.get("primary", ())), None)
    if primary_fallback:
        reasons.append(
            f"primary metric fell back to {primary} "
            f"({criteria['primary'][0]} not scored in both runs)"
        )

    entry = metrics[primary]
    # Orient delta and CI so that positive = better; a CI entirely above
    # zero is a significant gain, entirely below a significant loss.
    sign = -1.0 if primary in LOWER_IS_BETTER else 1.0
    gain = sign * entry["delta"]
    gain_low, gain_high = sorted(sign * bound for bound in entry["ci95"])
    min_improvement = criteria.get("min_improvement", 0.0)

    if breached:
        decision = "reject"
    elif gain_high < 0:
        decision = "reject"
        reasons.append(f"{primary} significantly worse (gain {gain:.4f}, 95% CI {gain_low:.4f}..{gain_high:.4f})")
    elif gain_low > 0 and gain >= min_improvement:
        decision = "accept"
        reasons.append(f"{primary} improved by {gain:.4f} (95% CI {gain_low:.4f}..{gain_high:.4f})")
    else:
        decision = "inconclusive"
        reasons.append(
            f"{primary} gain {gain:.4f} (95% CI {gain_low:.4f}..{gain_high:.4f}) "
            f"not a significant improvement of at least {min_improvement}"
        )
    return {
        "decision": decision,
        "primary": primary,
        "primary_fallback": primary_fallback,
        "reasons": reasons,
        "skipped_guardrails": skipped,
    }


def compare_runs(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    criteria: Mapping[str, Any],
    question_set: Sequence[Mapping[str, Any]] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired comparison of ``candidate`` against ``baseline`` plus a verdict.

    Both runs must cover the same questions; a comparison on different
    question sets is not fair and raises. Each metric is compared on the
    questions both runs scored: mean delta (candidate - baseline), paired
    bootstrap 95% CI, and an exact McNemar test for 0/1 metrics.
    """
    base_keys, cand_keys = sample_keys(baseline), sample_keys(candidate)
    if set(base_keys) != set(cand_keys):
        missing = len(set(base_keys) - set(cand_keys))
        extra = len(set(cand_keys) - set(base_keys))
        raise ValueError(
            f"runs cover different question sets: candidate lacks {missing} "
            f"and adds {extra} baseline questions; rerun on the frozen set"
        )

    if question_set is not None:
        frozen_keys = sample_keys(question_set)
        if set(frozen_keys) != set(base_keys):
            raise ValueError(
                "runs do not cover the frozen question set "
                f"({len(base_keys)} run questions vs {len(frozen_keys)} frozen); "
                "rerun on the frozen dev or holdout split"
            )

    categories = question_categories(question_set)

    def by_key(results: Sequence[Mapping[str, Any]], keys: list[SampleKey]) -> dict:
        return {
            key: sample_metrics(result, categories.get(key, "unlabeled"))
            for key, result in zip(keys, results)
        }

    base, cand = by_key(baseline, base_keys), by_key(candidate, cand_keys)
    names = sorted({name for m in base.values() for name in m})
    metrics: dict[str, dict[str, Any]] = {}
    for name in names:
        # Per-sample metrics differ by category (no retrieval scores on
        # unanswerable questions), so compare on the questions both scored.
        pairs = [
            (base[key].get(name), cand[key].get(name), key[0])
            for key in base_keys
            if base[key].get(name) is not None and cand[key].get(name) is not None
        ]
        if pairs:
            metrics[name] = _metric_comparison(
                [b for b, _, _ in pairs],
                [c for _, c, _ in pairs],
                [q for _, _, q in pairs],
                n_boot,
                seed,
            )

    return {"n": len(base_keys), "metrics": metrics, "verdict": _verdict(metrics, criteria)}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

SPLITS = ("dev", "holdout")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_run_results(path: str | Path) -> list[dict[str, Any]]:
    """Per-sample records of a single-config run.

    Accepts the config report ``<config>.json`` or its
    ``<config>.samples.json.gz`` sidecar.
    """
    path = Path(path)
    if path.name.endswith(".samples.json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            per_result = json.load(fh)["results"]
    else:
        from benchmark.reporting.exports import load_json_report

        per_result = [r.get("per_sample") for r in load_json_report(path)["results"]]
    filled = [samples for samples in per_result if samples]
    if len(filled) != 1:
        raise ValueError(f"{path}: expected exactly one config with per-sample data, found {len(filled)}")
    return filled[0]


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
    root = Path(args.question_set)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    ok = True
    for split in SPLITS:
        expected = manifest["splits"][split]["sha256"]
        path = root / f"{split}.jsonl"
        if not path.exists():
            print(f"{split}: MISSING ({path}); rebuild the splits with the freeze command")
            ok = False
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        status = "ok" if actual == expected else "MODIFIED"
        ok &= actual == expected
        print(f"{split}: {status} ({actual})")
    return 0 if ok else 1


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return str(value) if isinstance(value, int) else f"{value:.{digits}f}"


def render_report(report: Mapping[str, Any]) -> str:
    lines = [f"# Blueprint regression report ({report['split']} split)", ""]
    for name, path in report["runs"].items():
        lines.append(f"- **{name}**: `{path}`")
    lines.append("")
    for ref, comparison in report["comparisons"].items():
        verdict = comparison["verdict"]
        lines += [
            f"## Candidate vs {ref}: **{verdict['decision']}**",
            "",
            f"Primary metric: `{verdict['primary']}`" + (" (fallback)" if verdict.get("primary_fallback") else ""),
            "",
            *[f"- {reason}" for reason in verdict["reasons"]],
            *[f"- guardrail `{g}` skipped: not scored in both runs" for g in verdict["skipped_guardrails"]],
            "",
            f"| metric | {ref} | candidate | delta | 95% CI | McNemar p | n |",
            "|---|---|---|---|---|---|---|",
        ]
        for metric, entry in comparison["metrics"].items():
            p_value = entry.get("mcnemar", {}).get("p_value")
            low, high = entry["ci95"]
            lines.append(
                f"| {metric} | {_fmt(entry['baseline'])} | {_fmt(entry['candidate'])} | "
                f"{_fmt(entry['delta'])} | {_fmt(low)}..{_fmt(high)} | {_fmt(p_value, 3)} | {entry['n']} |"
            )
        lines.append("")
    lines += ["## Dimensions per run", ""]
    for name, summary in report["summaries"].items():
        lines.append(f"### {name} (n={summary['n']})")
        for dim, info in summary["dimensions"].items():
            if info["status"] == "ok":
                values = ", ".join(f"{k}={_fmt(v)}" for k, v in info["metrics"].items())
                lines.append(f"- {dim}: {values}")
            else:
                lines.append(f"- {dim}: unavailable — {info['reason']}")
        lines.append("")
    return "\n".join(lines)


def _cmd_compare(args: argparse.Namespace) -> int:
    question_set = read_jsonl(Path(args.question_set) / f"{args.split}.jsonl")
    criteria = yaml.safe_load(Path(args.criteria).read_text(encoding="utf-8"))
    runs = {"baseline": args.baseline, "best": args.best, "candidate": args.candidate}
    runs = {name: path for name, path in runs.items() if path}
    results = {name: load_run_results(path) for name, path in runs.items()}

    report = {
        "split": args.split,
        "runs": runs,
        "criteria": criteria,
        "summaries": {name: summarize_run(res, question_set) for name, res in results.items()},
        "comparisons": {
            ref: compare_runs(results[ref], results["candidate"], criteria=criteria,
                              question_set=question_set, n_boot=args.n_boot, seed=args.seed)
            for ref in ("baseline", "best")
            if ref in results
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
    parser = argparse.ArgumentParser(prog="python -m benchmark.regression", description=__doc__.split("\n")[0])
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

    compare = sub.add_parser("compare", help="judge a candidate run against the baseline / current best")
    compare.add_argument("--question-set", required=True)
    compare.add_argument("--split", choices=SPLITS, default="dev")
    compare.add_argument("--criteria", required=True)
    compare.add_argument("--baseline", required=True, help="immutable reference baseline run report")
    compare.add_argument("--best", help="current best configuration run report")
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--out", required=True)
    compare.add_argument("--n-boot", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=0)
    compare.set_defaults(func=_cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
