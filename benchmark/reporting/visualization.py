from __future__ import annotations

import logging
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from benchmark.reporting.models import BenchmarkResultExtended
from benchmark.reporting.analysis import RankTable

logger = logging.getLogger(__name__)


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

    # Interactive Plotly charts
    for plot_fn in [
        _plot_metrics_over_time_html,
        _plot_llm_comparison_radar_html,
        _plot_parameter_heatmap_html,
        _plot_scatter_matrix_html,
        _plot_correlation_heatmap_html,
        _plot_metric_boxplots_html,
        _plot_metric_violin_html,
        _plot_parallel_coordinates_html,
        _plot_metric_ranking_html,
        _plot_parameter_importance_html,
    ]:
        try:
            result = plot_fn(results, plots_dir)
            if result:
                generated.append(result)
        except Exception as e:
            logger.warning("Plotly chart %s failed: %s", plot_fn.__name__, e)

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


def _plot_metrics_over_time_html(results: list, output_dir: Path) -> str | None:
    """Interactive metrics-over-time line chart saved as HTML."""
    if len(results) < 2:
        return None

    metric_keys = ["ragas_faithfulness", "custom_hit_at_1", "custom_rouge_l", "custom_bert_score_f1"]
    metric_labels = ["Faithfulness", "Hit@1", "ROUGE-L", "BERTScore F1"]

    records = []
    for r in results:
        rec = {
            "config": r.config_name,
            "llm": r.llm_model.split("/")[-1],
            "chunking": r.chunking_strategy,
        }
        if r.ragas_faithfulness is not None:
            rec["ragas_faithfulness"] = r.ragas_faithfulness
        if r.custom_metric_means:
            for k in ["hit@1", "rouge_l", "bert_score_f1"]:
                if k in r.custom_metric_means:
                    safe = k.replace("@", "_at_")
                    rec[f"custom_{safe}"] = r.custom_metric_means[k]
        records.append(rec)

    fig = make_subplots(rows=2, cols=2, subplot_titles=metric_labels)
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (row, col), key in zip(positions, metric_keys):
        for llm in sorted(set(r["llm"] for r in records)):
            y_vals = [r.get(key) for r in records if r["llm"] == llm and key in r]
            x_vals = list(range(len(y_vals)))
            if y_vals:
                fig.add_trace(
                    go.Scatter(x=x_vals, y=y_vals, mode="lines+markers", name=llm, showlegend=(row == 1 and col == 1)),
                    row=row, col=col,
                )

    fig.update_layout(title_text="Metrics Across Configurations", height=700)
    path = output_dir / "metrics_over_time.html"
    fig.write_html(str(path))
    return str(path)


def _plot_llm_comparison_radar_html(results: list, output_dir: Path) -> str | None:
    """Interactive radar chart comparing LLMs across metrics."""
    llm_data: dict[str, list] = {}

    for r in results:
        llm = r.llm_model.split("/")[-1]
        if llm not in llm_data:
            llm_data[llm] = []
        metrics = {}
        if r.ragas_faithfulness is not None:
            metrics["Faithfulness"] = r.ragas_faithfulness
        if r.custom_metric_means:
            for k, label in [("hit@1", "Hit@1"), ("rouge_l", "ROUGE-L"), ("meteor", "METEOR"), ("bert_score_f1", "BERTScore F1"), ("context_relevance", "Ctx Rel")]:
                if k in r.custom_metric_means:
                    metrics[label] = r.custom_metric_means[k]
        llm_data[llm].append(metrics)

    if len(llm_data) < 2:
        return None

    avg_data = {}
    for llm, metrics_list in llm_data.items():
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        avg_data[llm] = {k: sum(m.get(k, 0) for m in metrics_list) / len(metrics_list) for k in all_keys}

    categories = sorted(set(k for v in avg_data.values() for k in v))
    if not categories:
        return None

    fig = go.Figure()
    for llm, metrics in avg_data.items():
        values = [metrics.get(c, 0) for c in categories]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=llm,
        ))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="LLM Comparison")
    path = output_dir / "llm_comparison_radar.html"
    fig.write_html(str(path))
    return str(path)


def _plot_parameter_heatmap_html(results: list, output_dir: Path) -> str | None:
    """Interactive heatmap of faithfulness by chunk_size x overlap per LLM."""
    if not results:
        return None

    llm_groups: dict[str, list] = {}
    for r in results:
        llm = r.llm_model.split("/")[-1]
        if r.ragas_faithfulness is not None:
            llm_groups.setdefault(llm, []).append(r)

    if not llm_groups:
        return None

    paths = []
    for llm, runs in llm_groups.items():
        sizes = sorted(set(r.chunk_size for r in runs))
        overlaps = sorted(set(r.chunk_overlap for r in runs))

        z = [[None] * len(overlaps) for _ in sizes]
        annotations = [[None] * len(overlaps) for _ in sizes]

        for r in runs:
            si = sizes.index(r.chunk_size)
            oi = overlaps.index(r.chunk_overlap)
            z[si][oi] = r.ragas_faithfulness
            annotations[si][oi] = f"{r.ragas_faithfulness:.3f}"

        fig = go.Figure(data=go.Heatmap(
            z=z, x=[str(o) for o in overlaps], y=[str(s) for s in sizes],
            text=annotations, texttemplate="%{text}",
            colorscale="RdYlGn", zmin=0.7, zmax=1.0,
        ))
        fig.update_layout(title=f"Faithfulness: {llm}", xaxis_title="Overlap", yaxis_title="Chunk Size")

        safe_llm = llm.replace("/", "_").replace(":", "_")
        path = output_dir / f"heatmap_{safe_llm}.html"
        fig.write_html(str(path))
        paths.append(str(path))

    return paths[0] if paths else None


def _plot_scatter_matrix_html(results: list, output_dir: Path) -> str | None:
    """Interactive scatter matrix of custom metrics colored by LLM."""
    if len(results) < 2:
        return None

    records = []
    for r in results:
        rec = {"llm": r.llm_model.split("/")[-1]}
        if r.custom_metric_means:
            for k in ["hit@1", "ndcg@1", "rouge_l", "meteor", "bert_score_f1", "context_relevance"]:
                if k in r.custom_metric_means:
                    safe = k.replace("@", "_at_")
                    rec[safe] = r.custom_metric_means[k]
        if len(rec) > 2:
            records.append(rec)

    if len(records) < 2:
        return None

    import pandas as pd
    df = pd.DataFrame(records)
    numeric_cols = [c for c in df.columns if c != "llm"]
    if len(numeric_cols) < 2:
        return None

    fig = px.scatter_matrix(df, dimensions=numeric_cols, color="llm", title="Custom Metrics Scatter Matrix")
    fig.update_layout(height=800)
    path = output_dir / "scatter_matrix.html"
    fig.write_html(str(path))
    return str(path)


def _plot_correlation_heatmap_html(results: list, output_dir: Path) -> str | None:
    """Interactive correlation heatmap of all metrics."""
    if len(results) < 2:
        return None

    import pandas as pd

    records = []
    for r in results:
        rec = {}
        # RAGAS metrics
        for key in ["faithfulness", "answer_relevancy", "answer_correctness", "context_precision", "context_recall"]:
            val = getattr(r, f"ragas_{key}", None)
            if val is not None:
                rec[f"ragas_{key}"] = val
        # Custom metrics
        if r.custom_metric_means:
            for k, v in r.custom_metric_means.items():
                safe = k.replace("@", "_at_")
                rec[f"custom_{safe}"] = v
        if rec:
            records.append(rec)

    if len(records) < 2:
        return None

    df = pd.DataFrame(records)
    corr_matrix = df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[c.replace("ragas_", "").replace("custom_", "") for c in corr_matrix.columns],
        y=[c.replace("ragas_", "").replace("custom_", "") for c in corr_matrix.index],
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 9},
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
    ))
    fig.update_layout(title="Metric Correlation Matrix", width=700, height=700)
    path = output_dir / "correlation_heatmap.html"
    fig.write_html(str(path))
    return str(path)


def _plot_metric_boxplots_html(results: list, output_dir: Path) -> str | None:
    """Interactive box plots of per-sample metrics grouped by LLM."""
    records = []

    for r in results:
        llm = r.llm_model.split("/")[-1]
        if not r.per_sample:
            continue
        for sample in r.per_sample:
            rec = {"llm": llm}
            if sample.ragas_scores and sample.ragas_scores.get("faithfulness") is not None:
                rec["faithfulness"] = sample.ragas_scores["faithfulness"]
            if sample.custom_scores:
                for k in ["rouge_l", "meteor", "bert_score_f1"]:
                    if k in sample.custom_scores and sample.custom_scores[k] is not None:
                        rec[k] = sample.custom_scores[k]
            if len(rec) > 1:
                records.append(rec)

    if not records:
        return None

    import pandas as pd
    df = pd.DataFrame(records)

    # Melt for box plotting
    metrics_to_plot = ["faithfulness", "rouge_l", "meteor", "bert_score_f1"]
    available = [m for m in metrics_to_plot if m in df.columns]
    if not available:
        return None

    df_melted = df.melt(id_vars=["llm"], value_vars=available, var_name="metric", value_name="value")

    fig = make_subplots(rows=2, cols=2, subplot_titles=available)
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (row, col), metric in zip(positions, available):
        for llm in sorted(df_melted["llm"].unique()):
            llm_data = df_melted[(df_melted["llm"] == llm) & (df_melted["metric"] == metric)]["value"].dropna()
            if len(llm_data) > 0:
                fig.add_trace(go.Box(
                    y=llm_data,
                    name=llm,
                    showlegend=(row == 1 and col == 1),
                ), row=row, col=col)

    fig.update_layout(title_text="Per-Sample Metric Distributions by LLM", height=700)
    path = output_dir / "metric_boxplots.html"
    fig.write_html(str(path))
    return str(path)


def _plot_metric_violin_html(results: list, output_dir: Path) -> str | None:
    """Interactive violin plots of per-sample metrics grouped by chunking strategy."""
    records = []

    for r in results:
        chunking = r.chunking_strategy
        if not r.per_sample:
            continue
        for sample in r.per_sample:
            rec = {"chunking": chunking}
            if sample.ragas_scores and sample.ragas_scores.get("faithfulness") is not None:
                rec["faithfulness"] = sample.ragas_scores["faithfulness"]
            if sample.custom_scores:
                for k in ["rouge_l", "meteor", "bert_score_f1"]:
                    if k in sample.custom_scores and sample.custom_scores[k] is not None:
                        rec[k] = sample.custom_scores[k]
            if len(rec) > 1:
                records.append(rec)

    if not records:
        return None

    import pandas as pd
    df = pd.DataFrame(records)

    metrics_to_plot = ["faithfulness", "rouge_l", "meteor", "bert_score_f1"]
    available = [m for m in metrics_to_plot if m in df.columns]
    if not available:
        return None

    df_melted = df.melt(id_vars=["chunking"], value_vars=available, var_name="metric", value_name="value")

    fig = make_subplots(rows=2, cols=2, subplot_titles=available)
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (row, col), metric in zip(positions, available):
        for chunking in sorted(df_melted["chunking"].unique()):
            chunk_data = df_melted[(df_melted["chunking"] == chunking) & (df_melted["metric"] == metric)]["value"].dropna()
            if len(chunk_data) > 0:
                fig.add_trace(go.Violin(
                    y=chunk_data,
                    name=chunking,
                    showlegend=(row == 1 and col == 1),
                ), row=row, col=col)

    fig.update_layout(title_text="Metric Distribution by Chunking Strategy", height=700)
    path = output_dir / "metric_violin.html"
    fig.write_html(str(path))
    return str(path)


def _plot_parallel_coordinates_html(results: list, output_dir: Path) -> str | None:
    """Interactive parallel coordinates plot showing parameter impact on metrics."""
    records = []

    for r in results:
        rec = {
            "chunk_size": r.chunk_size,
            "chunk_overlap": r.chunk_overlap,
        }
        if r.ragas_faithfulness is not None:
            rec["faithfulness"] = r.ragas_faithfulness
        if r.custom_metric_means:
            for k in ["hit@1", "rouge_l", "bert_score_f1"]:
                if k in r.custom_metric_means:
                    safe = k.replace("@", "_at_")
                    rec[safe] = r.custom_metric_means[k]
        if len(rec) >= 4:
            records.append(rec)

    if len(records) < 2:
        return None

    import pandas as pd
    df = pd.DataFrame(records)

    fig = px.parallel_coordinates(
        df,
        dimensions=["chunk_size", "chunk_overlap", "faithfulness", "hit_at_1", "rouge_l", "bert_score_f1"],
        color="faithfulness",
        title="Parameter Impact on Metrics",
        color_continuous_scale=px.colors.diverging.RdBu_r,
    )
    path = output_dir / "parallel_coordinates.html"
    fig.write_html(str(path))
    return str(path)


def _plot_metric_ranking_html(results: list, output_dir: Path) -> str | None:
    """Interactive horizontal bar chart ranking configs by faithfulness."""
    if not results:
        return None

    records = []
    for r in results:
        if r.ragas_faithfulness is not None:
            records.append({
                "config": r.config_name[:40],
                "llm": r.llm_model.split("/")[-1],
                "faithfulness": r.ragas_faithfulness,
            })

    if not records:
        return None

    import pandas as pd
    df = pd.DataFrame(records)
    df = df.sort_values("faithfulness", ascending=True)

    # Add rank markers
    colors = []
    for idx, row in df.iterrows():
        rank = len(df) - list(df.index).index(idx)
        if rank == 1:
            colors.append("#FFD700")  # gold
        elif rank == 2:
            colors.append("#C0C0C0")  # silver
        elif rank == 3:
            colors.append("#CD7F32")  # bronze
        else:
            colors.append("#4C78A8")

    fig = go.Figure()
    for i, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["faithfulness"]],
            y=[row["config"]],
            orientation="h",
            name=row["llm"],
            marker_color=colors[list(df.index).index(i)],
            showlegend=False,
            text=f"{row['faithfulness']:.3f}",
            textposition="outside",
        ))

    fig.update_layout(
        title="Faithfulness Ranking",
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Faithfulness Score",
        height=max(400, len(df) * 30),
        margin=dict(l=200, r=50, t=50, b=50),
    )
    path = output_dir / "metric_ranking.html"
    fig.write_html(str(path))
    return str(path)


def _plot_parameter_importance_html(results: list, output_dir: Path) -> str | None:
    """Bar chart showing parameter importance based on faithfulness variance."""
    if not results:
        return None

    import pandas as pd

    params_to_check = ["chunk_size", "chunk_overlap", "chunking_strategy", "retrieval_strategy", "prompt_template", "llm_model"]
    importance_scores = []

    for param in params_to_check:
        groups = {}
        for r in results:
            if r.ragas_faithfulness is None:
                continue
            val = getattr(r, param, None)
            if val is None:
                continue
            if isinstance(val, str):
                val = val.split("/")[-1]  # Shorten model names
            groups.setdefault(val, []).append(r.ragas_faithfulness)

        if len(groups) < 2:
            continue

        # Compute variance of group means
        group_means = [np.mean(v) for v in groups.values()]
        variance = np.var(group_means) if len(group_means) > 1 else 0
        importance_scores.append({
            "parameter": param.replace("_", " ").title(),
            "importance": variance,
            "num_groups": len(groups),
        })

    if not importance_scores:
        return None

    df = pd.DataFrame(importance_scores).sort_values("importance", ascending=True)

    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["importance"]],
            y=[row["parameter"]],
            orientation="h",
            name=row["parameter"],
            text=f"{row['num_groups']} groups",
            textposition="inside",
            showlegend=False,
        ))

    fig.update_layout(
        title="Parameter Importance (Faithfulness Variance)",
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Variance of Group Means",
        height=400,
        margin=dict(l=150, r=50, t=50, b=50),
    )
    path = output_dir / "parameter_importance.html"
    fig.write_html(str(path))
    return str(path)


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
