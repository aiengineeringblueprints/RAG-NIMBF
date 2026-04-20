from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from benchmark.reporting.models import BenchmarkResultExtended
from benchmark.reporting.analysis import RankTable


_METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}

_METRIC_KEYS = list(_METRIC_LABELS.keys())
_PALETTE = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#EECA3B", "#B279A2", "#FF9DA6"]


def _short_labels(results: list[BenchmarkResultExtended], max_len: int = 25) -> list[str]:
    labels = []
    for r in results:
        name = r.config_name
        if len(name) > max_len:
            name = name[: max_len - 2] + ".."
        labels.append(name)
    return labels


def generate_plots(
    results: list[BenchmarkResultExtended],
    rankings: RankTable,
    output_dir: Path,
) -> list[Path]:
    if not results:
        return []

    plots_dir = output_dir / "results_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    paths = [
        _plot_ragas_comparison(results, plots_dir),
        _plot_performance_metrics(results, plots_dir),
    ]

    radar = _plot_radar(results, plots_dir)
    if radar:
        paths.append(radar)

    heatmap = _plot_heatmap(results, plots_dir)
    if heatmap:
        paths.append(heatmap)

    if len(results) > 1:
        rank_path = _plot_ranking(results, rankings, plots_dir)
        if rank_path:
            paths.append(rank_path)

    scatter = _plot_vector_distance_scatter(results, plots_dir)
    if scatter:
        paths.append(scatter)

    generated = [p for p in paths if p is not None]
    return generated


def _plot_ragas_comparison(
    results: list[BenchmarkResultExtended], plots_dir: Path
) -> Path | None:
    labels = _short_labels(results)
    n_configs = len(results)
    n_metrics = len(_METRIC_KEYS)
    x = np.arange(n_configs)
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(10, n_configs * 2.5), 6))

    has_data = False
    for i, key in enumerate(_METRIC_KEYS):
        vals = [getattr(r, f"ragas_{key}") for r in results]
        clean = [v if v is not None else 0.0 for v in vals]
        stds = []
        for r in results:
            stats = getattr(r, f"ragas_{key}_stats")
            stds.append(stats.std if stats and stats.count > 1 else 0.0)
        if any(v is not None for v in vals):
            has_data = True
        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, clean, width, label=_METRIC_LABELS[key],
               color=_PALETTE[i % len(_PALETTE)], yerr=stds, capsize=3)

    if not has_data:
        plt.close(fig)
        return None

    ax.set_ylabel("Score")
    ax.set_title("RAGAS Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    path = plots_dir / "ragas_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_performance_metrics(
    results: list[BenchmarkResultExtended], plots_dir: Path
) -> Path | None:
    labels = _short_labels(results)
    n = len(results)

    fig, axes = plt.subplots(2, 2, figsize=(max(10, n * 2), 8))

    panels = [
        ("TTFT (seconds)", "avg_ttft_seconds", "ttft_stats"),
        ("Throughput (tok/s)", "avg_tokens_per_second", "tps_stats"),
        ("GPU Utilization (%)", "avg_gpu_utilization_pct", "gpu_util_stats"),
        ("Total Time (s)", "total_time_seconds", None),
    ]

    has_data = False
    for ax, (title, avg_attr, stats_attr) in zip(axes.flat, panels):
        vals = [getattr(r, avg_attr) for r in results]
        clean = [v if v is not None else 0.0 for v in vals]
        stds = []
        if stats_attr:
            for r in results:
                stats = getattr(r, stats_attr)
                stds.append(stats.std if stats and stats.count > 1 else 0.0)

        colors = [_PALETTE[i % len(_PALETTE)] for i in range(n)]
        bars = ax.bar(range(n), clean, color=colors,
                      yerr=stds if stds else None, capsize=3)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)

        if any(v is not None for v in vals):
            has_data = True

    if not has_data:
        plt.close(fig)
        return None

    fig.suptitle("Performance Metrics", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = plots_dir / "performance_metrics.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_radar(
    results: list[BenchmarkResultExtended], plots_dir: Path
) -> Path | None:
    labels = list(_METRIC_LABELS.values())
    n_metrics = len(labels)

    # Check if we have any data
    has_any = False
    for r in results:
        for key in _METRIC_KEYS:
            if getattr(r, f"ragas_{key}") is not None:
                has_any = True
                break
    if not has_any:
        return None

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    for i, r in enumerate(results):
        values = []
        for key in _METRIC_KEYS:
            val = getattr(r, f"ragas_{key}")
            values.append(val if val is not None else 0.0)
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=1.5,
                label=_short_labels([r])[0], color=_PALETTE[i % len(_PALETTE)])
        ax.fill(angles, values, alpha=0.15, color=_PALETTE[i % len(_PALETTE)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("RAGAS Radar Chart", pad=20, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)

    path = plots_dir / "ragas_radar.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_heatmap(
    results: list[BenchmarkResultExtended], plots_dir: Path
) -> Path | None:
    if not results:
        return None

    config_labels = _short_labels(results)
    metric_labels = list(_METRIC_LABELS.values())

    data = np.full((len(results), len(_METRIC_KEYS)), np.nan)
    for i, r in enumerate(results):
        for j, key in enumerate(_METRIC_KEYS):
            val = getattr(r, f"ragas_{key}")
            if val is not None:
                data[i, j] = val

    if np.all(np.isnan(data)):
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(_METRIC_KEYS) * 1.5), max(4, len(results) * 0.8)))
    nan_mask = np.isnan(data)
    display_data = np.where(nan_mask, 0, data)

    im = ax.imshow(display_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(_METRIC_KEYS)))
    ax.set_xticklabels(metric_labels, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(config_labels, fontsize=8)

    # Add text annotations
    for i in range(len(results)):
        for j in range(len(_METRIC_KEYS)):
            if not nan_mask[i, j]:
                val = data[i, j]
                color = "white" if val < 0.4 or val > 0.85 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=8, fontweight="bold")
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        color="gray", fontsize=8)

    ax.set_title("RAGAS Heatmap", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    path = plots_dir / "ragas_heatmap.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_ranking(
    results: list[BenchmarkResultExtended],
    rankings: RankTable,
    plots_dir: Path,
) -> Path | None:
    if not rankings.ranks:
        return None

    sorted_ranks = sorted(rankings.ranks, key=lambda cr: cr.composite_score)

    labels = [_short_labels([results[0]])[0] for _ in sorted_ranks]
    # Find actual result for each rank
    name_to_result = {r.config_name: r for r in results}
    labels = []
    scores = []
    for cr in sorted_ranks:
        r = name_to_result.get(cr.config_name)
        labels.append(_short_labels([r])[0] if r else cr.config_name)
        scores.append(cr.composite_score)

    # Color by rank position
    n = len(scores)
    rank_colors = []
    for i, cr in enumerate(sorted_ranks):
        if cr.rank == 1:
            rank_colors.append("#FFD700")  # gold
        elif cr.rank == 2:
            rank_colors.append("#C0C0C0")  # silver
        elif cr.rank == 3:
            rank_colors.append("#CD7F32")  # bronze
        else:
            rank_colors.append(_PALETTE[i % len(_PALETTE)])

    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), max(4, n * 0.6)))
    y_pos = range(n)
    ax.barh(y_pos, scores, color=rank_colors, edgecolor="gray", linewidth=0.5)

    # Add score labels
    for i, score in enumerate(scores):
        ax.text(score + 0.01, i, f"{score:.3f}", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Composite Score")
    ax.set_title("Configuration Ranking", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(scores) * 1.15 if scores else 1)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = plots_dir / "ranking_chart.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_vector_distance_scatter(
    results: list[BenchmarkResultExtended], plots_dir: Path
) -> Path | None:
    has_data = False
    for r in results:
        for s in r.per_sample:
            if s.custom_scores and s.custom_scores.get("vec_dist_q_gt") is not None:
                has_data = True
                break
        if has_data:
            break
    if not has_data:
        return None

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, r in enumerate(results):
        x_vals, y_vals = [], []
        for s in r.per_sample:
            if not s.custom_scores:
                continue
            dx = s.custom_scores.get("vec_dist_q_gt")
            dy = s.custom_scores.get("vec_dist_q_answer")
            if dx is not None and dy is not None:
                x_vals.append(dx)
                y_vals.append(dy)

        if not x_vals:
            continue
        label = _short_labels([r])[0]
        ax.scatter(x_vals, y_vals, alpha=0.6, s=30,
                   color=_PALETTE[i % len(_PALETTE)], label=label, edgecolors="white", linewidths=0.3)

    # Diagonal reference line y = x
    all_vals = []
    for r in results:
        for s in r.per_sample:
            if s.custom_scores:
                for key in ("vec_dist_q_gt", "vec_dist_q_answer"):
                    v = s.custom_scores.get(key)
                    if v is not None:
                        all_vals.append(v)
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        margin = (hi - lo) * 0.05 if hi > lo else 0.1
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "--", color="gray", alpha=0.5, linewidth=1, label="y = x (parity)")
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)

    ax.set_xlabel("Distance  Q ↔ Ground Truth", fontsize=10)
    ax.set_ylabel("Distance  Q ↔ Answer", fontsize=10)
    ax.set_title("Vector Space Distance Comparison", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    path = plots_dir / "vector_distance_scatter.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
