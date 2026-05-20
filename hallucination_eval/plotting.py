"""Paper-ready comparison plots for hallucination evaluation."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font="sans-serif", palette="deep")


def _paper_rc() -> dict:
    return {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
    }


def plot_auroc_by_dataset(
    comparison_results: dict[str, dict[str, object]],
    output_dir: Path,
) -> Path | None:
    """Grouped bar chart: AUROC per dataset per method."""
    from hallucination_eval.scoring import ComparisonMetrics

    datasets = sorted(comparison_results.keys())
    methods = sorted({m for per_ds in comparison_results.values() for m in per_ds})

    auroc_data = {}
    for ds in datasets:
        for method in methods:
            m = comparison_results[ds].get(method)
            if m and m.auroc is not None:
                auroc_data.setdefault(method, {})[ds] = m.auroc

    if not auroc_data:
        return None

    n_datasets = len(datasets)
    n_methods = len(auroc_data)
    x = np.arange(n_datasets)
    width = 0.8 / max(n_methods, 1)

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(max(6, n_datasets * 1.8), 5))
        palette = sns.color_palette("deep", n_colors=n_methods)

        for i, (method, ds_scores) in enumerate(sorted(auroc_data.items())):
            vals = [ds_scores.get(ds, 0) for ds in datasets]
            offset = (i - n_methods / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, vals, width, label=method,
                color=palette[i], edgecolor="white", linewidth=0.5,
            )
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=9, rotation=20, ha="right")
        ax.set_ylabel("AUROC", fontsize=11, fontweight="bold")
        ax.set_title(
            "Hallucination Detection AUROC by Dataset",
            fontsize=13, fontweight="bold", pad=10,
        )
        ax.set_ylim(0, 1.05)
        ax.legend(title="Method", fontsize=9, title_fontsize=9)
        ax.grid(axis="y", alpha=0.2, linestyle="--")

        fig.tight_layout()
        path = output_dir / "auroc_by_dataset.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path


def plot_score_distributions(
    all_scores: dict[str, list[float]],
    output_dir: Path,
) -> Path | None:
    """Violin/box plot of score distributions per method."""
    methods = [m for m, scores in all_scores.items() if any(not np.isnan(s) for s in scores)]
    if len(methods) < 2:
        return None

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.5), 5))

        data = [all_scores[m] for m in methods]
        clean_data = [[s for s in d if not np.isnan(s)] for d in data]

        palette = sns.color_palette("deep", n_colors=len(methods))
        parts = ax.violinplot(clean_data, positions=range(len(methods)), showmedians=True)

        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(palette[i])
            pc.set_alpha(0.6)

        bp = ax.boxplot(
            clean_data, positions=range(len(methods)),
            widths=0.15, patch_artist=True, showfliers=False,
        )
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(palette[i])
            box.set_alpha(0.8)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=10)
        ax.set_ylabel("Hallucination Score (0=hal, 1=faithful)", fontsize=10, fontweight="bold")
        ax.set_title("Score Distributions by Method", fontsize=13, fontweight="bold", pad=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.2, linestyle="--")

        fig.tight_layout()
        path = output_dir / "score_distributions.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path


def plot_correlation_scatter(
    method_scores: dict[str, list[float]],
    ragas_scores: list[float],
    output_dir: Path,
) -> Path | None:
    """Per-method scatter: detector score vs RAGAS faithfulness."""
    methods = [m for m in method_scores if m != "ragas_faithfulness"]
    if not methods:
        return None

    n = len(methods)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols

    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        palette = sns.color_palette("deep", n_colors=n)

        for idx, method in enumerate(methods):
            ax = axes[idx]
            d_scores = method_scores[method]
            pairs = [
                (d, r)
                for d, r in zip(d_scores, ragas_scores)
                if not (np.isnan(d) or np.isnan(r))
            ]
            if not pairs:
                ax.set_visible(False)
                continue

            ds = [p[0] for p in pairs]
            rs = [p[1] for p in pairs]

            ax.scatter(ds, rs, alpha=0.4, s=20, color=palette[idx], edgecolors="white", linewidths=0.3)

            try:
                z = np.polyfit(ds, rs, 1)
                p_line = np.poly1d(z)
                x_line = np.linspace(min(ds), max(ds), 100)
                ax.plot(x_line, p_line(x_line), "--", color=palette[idx], alpha=0.7, linewidth=1.5)
            except Exception:
                pass

            try:
                from scipy.stats import pearsonr
                r_val, _ = pearsonr(ds, rs)
                ax.text(0.05, 0.95, f"r={r_val:.3f}", transform=ax.transAxes, fontsize=9, va="top")
            except Exception:
                pass

            ax.set_xlabel(f"{method} Score", fontsize=9)
            ax.set_ylabel("RAGAS Faithfulness", fontsize=9)
            ax.set_title(method, fontsize=11, fontweight="bold")
            ax.grid(alpha=0.2, linestyle="--")

        for idx in range(len(methods), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Detector Scores vs RAGAS Faithfulness", fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()

        path = output_dir / "correlation_scatter.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path


def plot_threshold_sensitivity(
    sensitivity_data: dict[str, list[tuple[float, float | None]]],
    output_dir: Path,
) -> Path | None:
    """Line plot: AUROC across faithfulness thresholds per method."""
    if not sensitivity_data:
        return None

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(7, 5))
        palette = sns.color_palette("deep", n_colors=len(sensitivity_data))

        for i, (method, data) in enumerate(sensitivity_data.items()):
            thresholds = [d[0] for d in data]
            aurocs = [d[1] if d[1] is not None else float("nan") for d in data]
            ax.plot(thresholds, aurocs, "o-", label=method, color=palette[i], markersize=5)

        ax.set_xlabel("RAGAS Faithfulness Threshold (hallucination label)", fontsize=11, fontweight="bold")
        ax.set_ylabel("AUROC", fontsize=11, fontweight="bold")
        ax.set_title("AUROC Sensitivity to Threshold Choice", fontsize=13, fontweight="bold", pad=10)
        ax.legend(fontsize=9)
        ax.set_ylim(0.3, 1.0)
        ax.grid(alpha=0.2, linestyle="--")

        fig.tight_layout()
        path = output_dir / "threshold_sensitivity.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path


def generate_all_plots(
    comparison_results: dict[str, dict[str, object]],
    all_scores: dict[str, dict[str, list[float]]],
    sensitivity_data: dict[str, list[tuple[float, float | None]]],
    output_dir: Path,
) -> list[Path]:
    """Generate all hallucination comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    p = plot_auroc_by_dataset(comparison_results, output_dir)
    if p:
        generated.append(p)

    agg_scores: dict[str, list[float]] = {}
    for ds in all_scores:
        for method, scores in all_scores[ds].items():
            agg_scores.setdefault(method, []).extend(scores)

    p = plot_score_distributions(agg_scores, output_dir)
    if p:
        generated.append(p)

    first_ds = next(iter(all_scores), None)
    if first_ds and "ragas_faithfulness" in all_scores.get(first_ds, {}):
        ragas = all_scores[first_ds]["ragas_faithfulness"]
        method_scores = {m: s for m, s in all_scores[first_ds].items() if m != "ragas_faithfulness"}
        p = plot_correlation_scatter(method_scores, ragas, output_dir)
        if p:
            generated.append(p)

    p = plot_threshold_sensitivity(sensitivity_data, output_dir)
    if p:
        generated.append(p)

    return generated
