"""Publication-quality chart generator for paper and LLM performance tracking.

Generates PNG + PDF charts from cross-run benchmark data.
Output goes to Paper/figures/ and results/cross_run_plots/.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from benchmark.reporting.run_tracker import scan_all_results

logger = logging.getLogger(__name__)

# ── Style ──────────────────────────────────────────────────────────────────

PALETTE = sns.color_palette("deep")
ACCENT = "#2563EB"
GRID_ALPHA = 0.25

_RAGAS_METRICS = [
    ("ragas_faithfulness", "Faithfulness"),
    ("ragas_answer_relevancy", "Answer Relevancy"),
    ("ragas_answer_correctness", "Answer Correctness"),
    ("ragas_context_precision", "Context Precision"),
    ("ragas_context_recall", "Context Recall"),
    ("ragas_semantic_similarity", "Semantic Similarity"),
]

_SPEED_METRICS = [
    ("avg_ttft_seconds", "TTFT (s)", True),
    ("avg_tokens_per_second", "Throughput (tok/s)", False),
    ("total_time_seconds", "Total Time (s)", True),
]


def _paper_style() -> dict:
    return {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "axes.grid": True,
        "grid.alpha": GRID_ALPHA,
        "grid.linestyle": "--",
    }


def _save(fig, path_stem: Path) -> Path:
    """Save as PNG and PDF."""
    for ext in (".png", ".pdf"):
        p = path_stem.with_suffix(ext)
        fig.savefig(p, bbox_inches="tight")
        logger.info("Saved %s", p)
    plt.close(fig)
    return path_stem.with_suffix(".png")


def _llm_label(model: str) -> str:
    return model.split("/")[-1].replace(":", " ")


def _clean_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
    sub = df.dropna(subset=[col]).copy()
    if col.startswith("ragas_"):
        sub = sub[sub[col] > 0]
    return sub


# ── 1. LLM Quality Heatmap ────────────────────────────────────────────────

def plot_llm_metric_heatmap(df: pd.DataFrame, out: Path) -> Path | None:
    """Heatmap: LLM models x RAGAS metrics, values = mean score."""
    cols = [(c, l) for c, l in _RAGAS_METRICS if c in df.columns]
    if not cols:
        return None

    heatmap_data = {}
    for llm in sorted(df["llm_short"].unique()):
        sub = df[df["llm_short"] == llm]
        row = {}
        for col, _ in cols:
            vals = sub[col].dropna()
            vals = vals[vals > 0] if col.startswith("ragas_") else vals
            row[col] = vals.mean() if len(vals) > 0 else np.nan
        heatmap_data[llm] = row

    heat_df = pd.DataFrame(heatmap_data).T
    heat_df.columns = [l for _, l in cols]

    with plt.rc_context(_paper_style()):
        fig, ax = plt.subplots(figsize=(8, max(3, len(heat_df) * 0.7)))
        sns.heatmap(
            heat_df, annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=0, vmax=1, linewidths=0.8, linecolor="white",
            cbar_kws={"label": "Score", "shrink": 0.8},
            ax=ax, annot_kws={"fontsize": 10, "fontweight": "bold"},
        )
        ax.set_title("RAGAS Metrics by LLM Model")
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.yticks(rotation=0)

    return _save(fig, out / "paper_llm_metric_heatmap")


# ── 2. Quality vs Speed Scatter ───────────────────────────────────────────

def plot_quality_vs_speed(df: pd.DataFrame, out: Path) -> Path | None:
    """Scatter: faithfulness vs throughput, sized by TTFT, colored by LLM."""
    sub = df.dropna(subset=["ragas_faithfulness", "avg_tokens_per_second"]).copy()
    sub = sub[sub["ragas_faithfulness"] > 0]
    sub = sub[sub["avg_tokens_per_second"] > 0]
    if len(sub) < 2:
        return None

    with plt.rc_context(_paper_style()):
        fig, ax = plt.subplots(figsize=(9, 6))

        llms = sorted(sub["llm_short"].unique())
        colors = dict(zip(llms, PALETTE[:len(llms)]))

        for llm in llms:
            mask = sub["llm_short"] == llm
            d = sub[mask]
            ax.scatter(
                d["avg_tokens_per_second"], d["ragas_faithfulness"],
                s=80, alpha=0.75, label=llm,
                color=colors[llm], edgecolors="white", linewidths=0.5,
            )

        # Pareto frontier line
        sub_sorted = sub.sort_values("avg_tokens_per_second")
        pareto_y = sub_sorted["ragas_faithfulness"].cummax()
        ax.plot(sub_sorted["avg_tokens_per_second"], pareto_y,
                "--", color="gray", alpha=0.5, linewidth=1, label="Pareto frontier")

        ax.set_xlabel("Throughput (tokens/s)")
        ax.set_ylabel("Faithfulness")
        ax.set_title("Quality vs. Speed Trade-off")
        ax.legend(loc="lower right", frameon=True)
        ax.set_ylim(0, 1.05)

    return _save(fig, out / "paper_quality_vs_speed")


# ── 3. Grouped Bar Charts (per-metric LLM comparison) ─────────────────────

def plot_grouped_bars(df: pd.DataFrame, out: Path) -> list[Path]:
    """One grouped bar chart per RAGAS metric, bars = LLM models."""
    paths = []
    cols = [(c, l) for c, l in _RAGAS_METRICS if c in df.columns]

    for col, label in cols:
        sub = _clean_df(df, col)
        if sub.empty:
            continue

        agg = sub.groupby("llm_short")[col].agg(["mean", "std"]).sort_values("mean", ascending=False)

        with plt.rc_context(_paper_style()):
            fig, ax = plt.subplots(figsize=(8, max(3, len(agg) * 0.6)))
            x = range(len(agg))
            bars = ax.bar(
                x, agg["mean"],
                yerr=agg["std"].fillna(0), capsize=4,
                color=PALETTE[:len(agg)], edgecolor="white", linewidth=0.5,
                width=0.6,
            )
            # Value labels
            for i, (idx, row) in enumerate(agg.iterrows()):
                ax.text(i, row["mean"] + row.get("std", 0) + 0.01,
                        f"{row['mean']:.3f}", ha="center", fontsize=9, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(agg.index, rotation=15, ha="right")
            ax.set_ylabel(label)
            ax.set_title(f"{label} by LLM Model")
            ax.set_ylim(0, min(1.1, agg["mean"].max() * 1.25))

        p = _save(fig, out / f"paper_grouped_{col}")
        paths.append(p)

    return paths


# ── 4. Speed/Performance Dashboard ────────────────────────────────────────

def plot_speed_dashboard(df: pd.DataFrame, out: Path) -> Path | None:
    """Multi-panel speed dashboard: TTFT, throughput, total time per LLM."""
    available = [(c, l, lb) for c, l, lb in _SPEED_METRICS if c in df.columns and df[c].notna().any()]
    if not available:
        return None

    n = len(available)
    llms = sorted(df["llm_short"].dropna().unique())
    colors = dict(zip(llms, PALETTE[:len(llms)]))

    with plt.rc_context(_paper_style()):
        fig, axes = plt.subplots(1, n, figsize=(5 * n, max(4, len(llms) * 0.5)))
        if n == 1:
            axes = [axes]

        for ax, (col, label, lower_better) in zip(axes, available):
            sub = df.dropna(subset=[col]).copy()
            agg = sub.groupby("llm_short")[col].agg(["mean", "std"])
            agg = agg.reindex(llms).dropna()

            if agg.empty:
                ax.set_visible(False)
                continue

            sort_asc = lower_better
            agg = agg.sort_values("mean", ascending=sort_asc)

            bars = ax.barh(
                range(len(agg)), agg["mean"],
                xerr=agg["std"].fillna(0), capsize=3,
                color=[colors.get(llm, PALETTE[0]) for llm in agg.index],
                edgecolor="white", linewidth=0.5, height=0.6,
            )

            for i, (idx, row) in enumerate(agg.iterrows()):
                fmt = ".2f" if row["mean"] < 10 else ".1f"
                ax.text(row["mean"] + row.get("std", 0) * 0.1 + agg["mean"].max() * 0.02,
                        i, f"{row['mean']:{fmt}}", va="center", fontsize=9, fontweight="bold")

            ax.set_yticks(range(len(agg)))
            ax.set_yticklabels(agg.index, fontsize=9)
            ax.set_xlabel(label)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.invert_yaxis()
            suffix = " (lower is better)" if lower_better else " (higher is better)"
            ax.set_title(f"{label}{suffix}", fontsize=10, fontweight="bold")

        fig.suptitle("LLM Speed Performance", fontsize=13, fontweight="bold")

    return _save(fig, out / "paper_speed_dashboard")


# ── 5. Faithfulness by Chunking Config ────────────────────────────────────

def plot_faithfulness_by_chunking(df: pd.DataFrame, out: Path) -> Path | None:
    """Grouped bars: faithfulness by chunking strategy x chunk size, per LLM."""
    sub = df.dropna(subset=["ragas_faithfulness", "chunking_strategy", "chunk_size"]).copy()
    sub = sub[sub["ragas_faithfulness"] > 0]
    if len(sub) < 2:
        return None

    # Aggregate: mean faithfulness per (llm, chunking_strategy, chunk_size)
    agg = sub.groupby(["llm_short", "chunking_strategy", "chunk_size"])["ragas_faithfulness"].mean().reset_index()

    llms = sorted(agg["llm_short"].unique())
    n_llms = len(llms)
    if n_llms == 0:
        return None

    strategies = sorted(agg["chunking_strategy"].unique())

    with plt.rc_context(_paper_style()):
        fig, axes = plt.subplots(1, n_llms, figsize=(5 * n_llms, 5), sharey=True)
        if n_llms == 1:
            axes = [axes]

        for ax, llm in zip(axes, llms):
            llm_data = agg[agg["llm_short"] == llm]
            x_labels = []
            x_pos = []
            bar_data = []

            for strat in strategies:
                strat_data = llm_data[llm_data["chunking_strategy"] == strat].sort_values("chunk_size")
                for _, row in strat_data.iterrows():
                    x_labels.append(f"{strat[:3]}|{int(row['chunk_size'])}")
                    bar_data.append(row["ragas_faithfulness"])

            x_pos = range(len(bar_data))
            strat_colors = dict(zip(strategies, PALETTE[:len(strategies)]))
            colors = []
            for strat in strategies:
                strat_data = llm_data[llm_data["chunking_strategy"] == strat].sort_values("chunk_size")
                colors.extend([strat_colors[strat]] * len(strat_data))

            ax.bar(x_pos, bar_data, color=colors, edgecolor="white", linewidth=0.5, width=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.set_title(llm, fontsize=10, fontweight="bold")
            ax.set_ylabel("Faithfulness" if ax == axes[0] else "")

            for i, v in enumerate(bar_data):
                ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, fontweight="bold")

        fig.suptitle("Faithfulness by Chunking Configuration", fontsize=13, fontweight="bold")

    return _save(fig, out / "paper_faithfulness_by_chunking")


# ── 6. Composite Ranking ──────────────────────────────────────────────────

def plot_composite_ranking(df: pd.DataFrame, out: Path, top_k: int = 15) -> Path | None:
    """Horizontal bar: composite score ranking top-K configs."""
    ragas_weights = {
        "ragas_faithfulness": 0.30,
        "ragas_context_recall": 0.25,
        "ragas_answer_relevancy": 0.15,
        "ragas_context_precision": 0.15,
        "ragas_answer_correctness": 0.15,
    }

    avail = {k: v for k, v in ragas_weights.items() if k in df.columns and df[k].notna().any()}
    if not avail:
        return None

    total_w = sum(avail.values())
    avail = {k: v / total_w for k, v in avail.items()}

    score_df = df.copy()
    for col in avail:
        score_df[col] = score_df[col].fillna(0)

    score_df["composite"] = sum(score_df[c] * w for c, w in avail.items())
    score_df = score_df[score_df["composite"] > 0]

    top = score_df.nlargest(top_k, "composite").copy()
    top["label"] = top.apply(
        lambda r: f"{r['llm_short']} | {r.get('chunking_strategy','')[:3]}{int(r.get('chunk_size',0))} | {r.get('prompt_template','')[:1]}",
        axis=1,
    )
    top = top.sort_values("composite", ascending=True)

    with plt.rc_context(_paper_style()):
        fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.35)))

        # Color by LLM
        llms = sorted(top["llm_short"].unique())
        color_map = dict(zip(llms, PALETTE[:len(llms)]))

        ax.barh(
            range(len(top)), top["composite"],
            color=[color_map.get(llm, PALETTE[0]) for llm in top["llm_short"]],
            edgecolor="white", linewidth=0.5, height=0.7,
        )

        for i, (_, row) in enumerate(top.iterrows()):
            ax.text(row["composite"] + 0.003, i, f"{row['composite']:.3f}",
                    va="center", fontsize=8, fontweight="bold")

        # Gold star for #1
        ax.scatter([top.iloc[-1]["composite"]], [len(top) - 1],
                   s=150, color="gold", edgecolors="black", linewidths=1.2,
                   zorder=5, marker="*")

        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["label"], fontsize=8)
        ax.set_xlabel("Composite Score")
        ax.set_title(f"Top-{len(top)} Configuration Ranking")
        ax.set_xlim(0, top["composite"].max() * 1.1)

        # Legend
        import matplotlib.patches as mpatches
        handles = [mpatches.Patch(color=color_map[llm], label=llm) for llm in llms]
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    return _save(fig, out / "paper_composite_ranking")


# ── Bonus: Speed Trend Over Runs ──────────────────────────────────────────

def plot_speed_trend(df: pd.DataFrame, out: Path) -> Path | None:
    """Line chart: TTFT and throughput trend across runs per LLM."""
    speed_cols = [c for c in ["avg_ttft_seconds", "avg_tokens_per_second"] if c in df.columns]
    if not speed_cols:
        return None

    llms = sorted(df["llm_short"].unique())
    colors = dict(zip(llms, PALETTE[:len(llms)]))

    with plt.rc_context(_paper_style()):
        fig, axes = plt.subplots(1, len(speed_cols), figsize=(6 * len(speed_cols), 5))
        if len(speed_cols) == 1:
            axes = [axes]

        labels = {"avg_ttft_seconds": "TTFT (s)", "avg_tokens_per_second": "Throughput (tok/s)"}

        for ax, col in zip(axes, speed_cols):
            for llm in llms:
                sub = df[(df["llm_short"] == llm) & (df[col].notna())].copy()
                if sub.empty:
                    continue
                agg = sub.groupby("run_number")[col].mean().sort_index()
                ax.plot(agg.index, agg.values, "o-", label=llm, color=colors[llm],
                        markersize=5, linewidth=1.5)

            ax.set_xlabel("Run Number")
            ax.set_ylabel(labels[col])
            ax.set_title(f"{labels[col]} Over Runs")
            ax.legend(fontsize=8)

        fig.suptitle("LLM Speed Performance Trend", fontsize=13, fontweight="bold")

    return _save(fig, out / "paper_speed_trend")


# ── Main Pipeline ─────────────────────────────────────────────────────────

def generate_all(
    results_dir: str | Path = "results",
    output_dir: str | Path | None = None,
    paper_dir: str | Path | None = None,
) -> list[Path]:
    """Generate all paper-quality charts."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir / "cross_run_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_dir = Path(paper_dir) if paper_dir else Path("Paper/figures")
    paper_dir.mkdir(parents=True, exist_ok=True)

    df = scan_all_results(results_dir)
    if df.empty:
        logger.error("No data found in %s", results_dir)
        return []

    logger.info("Generating paper charts from %d configs, %d LLMs", len(df), df["llm_short"].nunique())

    generated: list[Path] = []
    chart_fns = [
        ("LLM metric heatmap", plot_llm_metric_heatmap),
        ("Quality vs speed", plot_quality_vs_speed),
        ("Grouped bars", plot_grouped_bars),
        ("Speed dashboard", plot_speed_dashboard),
        ("Faithfulness by chunking", plot_faithfulness_by_chunking),
        ("Composite ranking", plot_composite_ranking),
        ("Speed trend", plot_speed_trend),
    ]

    for name, fn in chart_fns:
        try:
            result = fn(df, output_dir)
            if isinstance(result, list):
                generated.extend(result)
            elif result:
                generated.append(result)
            logger.info("  %s: %s", name, "OK" if result else "skipped (no data)")
        except Exception as e:
            logger.warning("  %s: failed — %s", name, e)

    # Copy to Paper/figures/ for LaTeX inclusion
    import shutil
    for p in generated:
        try:
            # Copy both PNG and PDF
            for ext in (".png", ".pdf"):
                src = p.with_suffix(ext) if p.suffix == ".png" else p
                if src.exists():
                    shutil.copy2(src, paper_dir / src.name)
        except Exception as e:
            logger.warning("Copy to paper dir failed: %s", e)

    print(f"\nGenerated {len(generated)} charts:")
    for p in generated:
        print(f"  {p.name}")
    print(f"  → Also copied to {paper_dir}/")

    return generated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate paper-quality benchmark charts")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots")
    parser.add_argument("--paper-dir", default="Paper/figures", help="Paper figures directory")
    args = parser.parse_args()

    generate_all(results_dir=args.results_dir, output_dir=args.output_dir, paper_dir=args.paper_dir)
