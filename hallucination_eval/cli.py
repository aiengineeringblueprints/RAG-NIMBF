"""Hallucination evaluation CLI: compare detection methods on RAG benchmark results.

Usage:
    python -m hallucination_eval --results-dir results
    python -m hallucination_eval --methods ragas,hhem,deberta
    python -m hallucination_eval --judge-model ollama:gemma3:12b
    python -m hallucination_eval --datasets squad,t2-ragbench
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

from hallucination_eval.detectors import (
    DETECTOR_REGISTRY,
    HallucinationSample,
)

logger = logging.getLogger(__name__)


# ── Data loading ─────────────────────────────────────────────────────

def load_per_sample_data(
    results_dir: Path,
    datasets: list[str] | None = None,
    llm_models: list[str] | None = None,
    max_samples: int | None = None,
) -> list[HallucinationSample]:
    """Scan benchmark JSON files and extract per-sample records."""
    results_dir = Path(results_dir)
    json_files = sorted(results_dir.glob("*/benchmark_*.json"))

    if not json_files:
        logger.error("No benchmark_*.json files found in %s", results_dir)
        return []

    samples: list[HallucinationSample] = []
    for jf in json_files:
        run_name = jf.parent.name
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Skipping %s: %s", jf, e)
            continue

        dataset_info = data.get("dataset", {})
        dataset_name = dataset_info.get("name", "unknown")
        if dataset_info.get("subset"):
            dataset_name = f"{dataset_name}/{dataset_info['subset']}"

        if datasets and dataset_name not in datasets and dataset_info.get("name") not in datasets:
            continue

        for entry in data.get("results", []):
            config_name = entry.get("config_name", "")
            llm_model = entry.get("llm_model", "")

            if llm_models and llm_model not in llm_models:
                continue

            for ps in entry.get("per_sample", []):
                ragas = ps.get("ragas_scores", {})
                faith = ragas.get("faithfulness")
                if faith is not None and not isinstance(faith, (int, float)):
                    faith = None

                samples.append(HallucinationSample(
                    run_name=run_name,
                    dataset_name=dataset_name,
                    config_name=config_name,
                    llm_model=llm_model,
                    question=ps.get("question", ""),
                    ground_truth=ps.get("ground_truth", ""),
                    answer=ps.get("answer", ""),
                    contexts=ps.get("contexts", []),
                    ragas_faithfulness=faith,
                    ragas_context_recall=ragas.get("context_recall"),
                    ragas_semantic_similarity=ragas.get("semantic_similarity"),
                    custom_scores=ps.get("custom_scores", {}),
                ))

    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]
        logger.info("Truncated to %d samples", max_samples)

    logger.info("Loaded %d samples from %d files", len(samples), len(json_files))
    return samples


def group_by_dataset(
    samples: list[HallucinationSample],
) -> dict[str, list[HallucinationSample]]:
    """Group samples by dataset name."""
    groups: dict[str, list[HallucinationSample]] = {}
    for s in samples:
        groups.setdefault(s.dataset_name, []).append(s)
    return groups


# ── Detector factory ─────────────────────────────────────────────────

def build_detectors(
    method_names: list[str],
    device: str = "cpu",
    judge_model: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
) -> list:
    """Instantiate requested detectors."""
    import hallucination_eval.detectors.ragas_faithfulness  # noqa: F401
    import hallucination_eval.detectors.hhem  # noqa: F401
    import hallucination_eval.detectors.deberta_nli  # noqa: F401
    import hallucination_eval.detectors.llm_judge  # noqa: F401

    detectors = []
    for name in method_names:
        if name == "ragas_faithfulness" or name == "ragas":
            cls = DETECTOR_REGISTRY["ragas_faithfulness"]
            detectors.append(cls())
        elif name == "hhem":
            cls = DETECTOR_REGISTRY["hhem"]
            detectors.append(cls(device=device))
        elif name == "deberta_nli" or name == "deberta":
            cls = DETECTOR_REGISTRY["deberta_nli"]
            detectors.append(cls(device=device))
        elif name == "llm_judge" or name == "llm":
            if not judge_model:
                logger.warning("Skipping llm_judge: --judge-model not specified")
                continue
            from benchmark.providers import parse_model_id
            provider, model_name = parse_model_id(judge_model)
            base_url = judge_base_url or ""
            cls = DETECTOR_REGISTRY["llm_judge"]
            detectors.append(cls(
                provider=provider,
                model_name=model_name,
                base_url=base_url,
                api_key=judge_api_key,
            ))
        else:
            logger.warning("Unknown method: %s", name)

    return detectors


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hallucination evaluation: compare detection methods",
    )
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="hallucination_results")
    parser.add_argument("--methods", type=str, default="ragas,hhem,deberta_nli")
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--faithfulness-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--judge-base-url", type=str, default=None)
    parser.add_argument("--judge-api-key", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--formats", type=str, default="markdown,csv,latex")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_names = [m.strip() for m in args.methods.split(",")]
    datasets_filter = [d.strip() for d in args.datasets.split(",")] if args.datasets else None
    formats = [f.strip() for f in args.formats.split(",")]

    # Load data
    print(f"Loading benchmark data from {results_dir}...")
    samples = load_per_sample_data(
        results_dir,
        datasets=datasets_filter,
        max_samples=args.max_samples,
    )
    if not samples:
        print("No samples found. Exiting.")
        sys.exit(1)

    grouped = group_by_dataset(samples)
    print(f"Found {len(samples)} samples across {len(grouped)} datasets: {list(grouped.keys())}")

    # Build detectors
    print(f"\nInitializing detectors: {method_names}")
    detectors = build_detectors(
        method_names,
        device=args.device,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_api_key=args.judge_api_key,
    )

    if not detectors:
        print("No detectors initialized. Exiting.")
        sys.exit(1)

    # Run detectors per dataset
    from hallucination_eval.scoring import compute_comparison, compute_threshold_sensitivity
    from hallucination_eval.table_generator import (
        generate_markdown_table,
        generate_latex_table,
        generate_csv_table,
    )

    comparison_results: dict[str, dict[str, object]] = {}
    all_scores: dict[str, dict[str, list[float]]] = {}
    sensitivity_data: dict[str, list[tuple[float, float | None]]] = {}

    for ds_name, ds_samples in sorted(grouped.items()):
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({len(ds_samples)} samples)")
        print(f"{'='*60}")

        comparison_results[ds_name] = {}
        all_scores[ds_name] = {}

        for detector in detectors:
            print(f"  Running {detector.name}...")
            outputs = detector.score_samples(ds_samples)
            scores = [o.score for o in outputs]
            all_scores[ds_name][detector.name] = scores

            ragas_scores = [
                s.ragas_faithfulness if s.ragas_faithfulness is not None else float("nan")
                for s in ds_samples
            ]

            metrics = compute_comparison(
                scores, ragas_scores,
                detector_name=detector.name,
                dataset_name=ds_name,
                threshold=args.faithfulness_threshold,
            )
            comparison_results[ds_name][detector.name] = metrics

            if detector.name != "ragas_faithfulness":
                sens = compute_threshold_sensitivity(scores, ragas_scores)
                sensitivity_data[detector.name] = sens

            print(f"    Mean: {metrics.mean_score:.3f}, AUROC: {metrics.auroc}, Pearson r: {metrics.pearson_r}")

    # Generate tables
    print(f"\n{'='*60}")
    print("Generating output...")
    print(f"{'='*60}")

    if "markdown" in formats:
        path = generate_markdown_table(comparison_results, output_dir / "comparison_table.md")
        print(f"  Markdown: {path}")

    if "latex" in formats:
        path = generate_latex_table(comparison_results, output_dir / "comparison_table.tex")
        print(f"  LaTeX: {path}")

    if "csv" in formats:
        path = generate_csv_table(comparison_results, output_dir / "comparison_table.csv")
        print(f"  CSV: {path}")

    # Save raw scores
    scores_path = output_dir / "scores.json"
    serializable = {}
    for ds, methods in all_scores.items():
        serializable[ds] = {}
        for method, vals in methods.items():
            serializable[ds][method] = [
                v if not math.isnan(v) else None for v in vals
            ]
    scores_path.write_text(json.dumps(serializable, indent=2))
    print(f"  Raw scores: {scores_path}")

    # Generate plots
    if not args.skip_plots:
        from hallucination_eval.plotting import generate_all_plots

        plots = generate_all_plots(
            comparison_results, all_scores, sensitivity_data, output_dir,
        )
        print(f"\nGenerated {len(plots)} plots:")
        for p in plots:
            print(f"  {p.name}")

    print(f"\nDone. Output in {output_dir}/")


if __name__ == "__main__":
    main()
