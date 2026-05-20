"""Table generation: Markdown, LaTeX, CSV output for hallucination comparison."""
from __future__ import annotations

import csv
from pathlib import Path

from hallucination_eval.scoring import ComparisonMetrics


def _sig_marker(p: float | None, level1: float = 0.05, level2: float = 0.01) -> str:
    if p is None:
        return ""
    if p < level2:
        return "**"
    if p < level1:
        return "*"
    return ""


def _fmt(val: float | None, fmt: str = ".3f") -> str:
    if val is None:
        return "—"
    return f"{val:{fmt}}"


def generate_markdown_table(
    results: dict[str, dict[str, ComparisonMetrics]],
    output_path: Path,
) -> Path:
    """Generate a Markdown comparison table.

    Parameters
    ----------
    results
        {dataset_name: {method_name: ComparisonMetrics}}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    methods = sorted({m for per_ds in results.values() for m in per_ds})
    datasets = sorted(results.keys())

    lines = []
    lines.append("# Hallucination Detection — Method Comparison")
    lines.append("")

    for ds in datasets:
        lines.append(f"## {ds}")
        lines.append("")
        lines.append("| Method | N | Mean Score | AUROC | Pearson r | Spearman r | RMSE vs RAGAS | Hal. % | Sig. |")
        lines.append("|--------|---|------------|-------|-----------|------------|---------------|--------|------|")

        for method in methods:
            m = results[ds].get(method)
            if m is None or m.n_samples == 0:
                lines.append(f"| {method} | — | — | — | — | — | — | — | — |")
                continue
            sig = _sig_marker(m.mann_whitney_p)
            lines.append(
                f"| {method} | {m.n_samples} | {_fmt(m.mean_score)} | {_fmt(m.auroc)} | "
                f"{_fmt(m.pearson_r)}{sig} | {_fmt(m.spearman_r)} | {_fmt(m.rmse_vs_ragas)} | "
                f"{_fmt(m.hallucinated_pct, '.1f')}% | {sig} |"
            )
        lines.append("")

    lines.append("## Aggregate (Mean Across Datasets)")
    lines.append("")
    lines.append("| Method | Datasets | Mean Score | Mean AUROC | Mean Pearson r | Mean RMSE |")
    lines.append("|--------|----------|------------|-----------|----------------|-----------|")

    for method in methods:
        metrics_list = [
            results[ds][method]
            for ds in datasets
            if method in results[ds] and results[ds][method].n_samples > 0
        ]
        if not metrics_list:
            lines.append(f"| {method} | 0 | — | — | — | — |")
            continue

        n_ds = len(metrics_list)
        mean_score = sum(m.mean_score for m in metrics_list) / n_ds
        aurocs = [m.auroc for m in metrics_list if m.auroc is not None]
        mean_auroc = sum(aurocs) / len(aurocs) if aurocs else None
        pearsons = [m.pearson_r for m in metrics_list if m.pearson_r is not None]
        mean_pearson = sum(pearsons) / len(pearsons) if pearsons else None
        rmses = [m.rmse_vs_ragas for m in metrics_list if m.rmse_vs_ragas is not None]
        mean_rmse = sum(rmses) / len(rmses) if rmses else None

        lines.append(
            f"| {method} | {n_ds} | {_fmt(mean_score)} | {_fmt(mean_auroc)} | "
            f"{_fmt(mean_pearson)} | {_fmt(mean_rmse)} |"
        )
    lines.append("")

    output_path.write_text("\n".join(lines))
    return output_path


def generate_latex_table(
    results: dict[str, dict[str, ComparisonMetrics]],
    output_path: Path,
) -> Path:
    """Generate a LaTeX table with booktabs styling."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    methods = sorted({m for per_ds in results.values() for m in per_ds})
    datasets = sorted(results.keys())

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Hallucination Detection Method Comparison}")
    lines.append("\\label{tab:hallucination_comparison}")
    lines.append("")

    col_spec = "ll" + "ll" * len(methods)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header_parts = ["Dataset", "N"]
    for m in methods:
        header_parts.extend([f"{m} Score", f"{m} AUROC"])
    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    for ds in datasets:
        row_parts = [ds.replace("_", "\\_")]
        n_val = ""
        for method in methods:
            m = results[ds].get(method)
            if m is not None and m.n_samples > 0:
                if not n_val:
                    n_val = str(m.n_samples)
                row_parts.append(_fmt(m.mean_score))
                row_parts.append(_fmt(m.auroc))
            else:
                row_parts.append("—")
                row_parts.append("—")
        row_parts.insert(1, n_val or "—")
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    output_path.write_text("\n".join(lines))
    return output_path


def generate_csv_table(
    results: dict[str, dict[str, ComparisonMetrics]],
    output_path: Path,
) -> Path:
    """Generate a flat CSV table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in sorted(results.keys()):
        for method in sorted(results[ds].keys()):
            m = results[ds][method]
            rows.append({
                "dataset": ds,
                "method": method,
                "n_samples": m.n_samples,
                "mean_score": _fmt(m.mean_score),
                "std_score": _fmt(m.std_score),
                "auroc": _fmt(m.auroc),
                "pearson_r": _fmt(m.pearson_r),
                "spearman_r": _fmt(m.spearman_r),
                "rmse_vs_ragas": _fmt(m.rmse_vs_ragas),
                "hallucinated_pct": _fmt(m.hallucinated_pct, ".1f"),
                "mann_whitney_p": _fmt(m.mann_whitney_p),
            })

    if rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return output_path
