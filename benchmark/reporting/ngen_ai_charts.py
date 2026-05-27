"""NGEN-AI paper chart generator.

Produces publication-quality charts for the LLNCS-format paper.
Output: Paper/NGEN-AI/figures/ (PNG + PDF)

Charts match the reference style: grouped bar charts with error bars,
clean white background, value annotations, subtle grid, bold labels.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Style matching reference image ────────────────────────────────────────

PALETTE = {
    "Qwen3-32B": "#2563EB",       # blue
    "Qwen3.5-397B": "#DC2626",    # red
    "GPT-OSS-20B": "#059669",     # green
    "Qwen3.5-think": "#D97706",   # amber
    "GPT-4o-mini": "#7C3AED",     # purple
}

MODEL_ORDER = ["GPT-OSS-20B", "Qwen3-32B", "GPT-4o-mini", "Qwen3.5-397B", "Qwen3.5-think"]

SHORT_NAMES = {
    "Qwen3-32B-AWQ": "Qwen3-32B",
    "Qwen3.5-397B-A17B": "Qwen3.5-397B",
    "gpt-oss_20b": "GPT-OSS-20B",
    "qwen3.5-think": "Qwen3.5-think",
    "gpt-4o-mini": "GPT-4o-mini",
    "qwen3.5_35b": "Qwen3.5-35b",
}


def _paper_rc() -> dict:
    """LLNCS paper-ready rcParams."""
    return {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.labelweight": "bold",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#CCCCCC",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }


def _save(fig, path_stem: Path) -> Path:
    """Save as PNG and PDF."""
    for ext in (".pdf", ".png"):
        p = path_stem.with_suffix(ext)
        fig.savefig(p, bbox_inches="tight")
        logger.info("Saved %s", p)
    plt.close(fig)
    return path_stem.with_suffix(".pdf")


def _normalize_llm(llm_short: str) -> str:
    return SHORT_NAMES.get(llm_short, llm_short)


def scan_data(results_dir: Path) -> pd.DataFrame:
    """Scan benchmark JSONs and return DataFrame with normalized LLM names."""
    from benchmark.reporting.run_tracker import scan_all_results
    df = scan_all_results(results_dir)
    if df.empty:
        return df
    df["model"] = df["llm_short"].apply(_normalize_llm)
    return df


# ── 1. Model Comparison: Grouped Bar Chart ────────────────────────────────

def plot_model_comparison(df: pd.DataFrame, out: Path) -> Path | None:
    """Grouped bar chart: 4 key metrics x models (matches reference image style)."""
    metrics = [
        ("ragas_faithfulness", "Faithfulness"),
        ("custom_ndcg_at_5", "nDCG@5"),
        ("custom_bert_score_f1", "BERTScore F1"),
        ("custom_meteor", "METEOR"),
    ]

    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    if not models:
        return None

    n_metrics = len(metrics)
    n_models = len(models)
    bar_width = 0.15
    x = np.arange(n_metrics)

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(7, 3.2))

        for i, model in enumerate(models):
            sub = df[df["model"] == model]
            vals = []
            errs = []
            for col, _ in metrics:
                if col in sub.columns:
                    clean = sub[col].dropna()
                    if col.startswith("ragas_"):
                        clean = clean[clean > 0]
                    vals.append(clean.mean() if len(clean) > 0 else 0)
                    errs.append(clean.std() if len(clean) > 1 else 0)
                else:
                    vals.append(0)
                    errs.append(0)

            offset = (i - n_models / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset, vals, bar_width,
                yerr=errs, capsize=2,
                label=model,
                color=PALETTE.get(model, "#888888"),
                edgecolor="white", linewidth=0.4,
                error_kw={"linewidth": 0.8},
            )

            # Value labels on bars
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                            f"{v:.2f}", ha="center", va="bottom",
                            fontsize=6, fontweight="bold", rotation=0)

        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in metrics])
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Score")
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.legend(loc="upper right", ncol=3, columnspacing=1)
        ax.set_title("RQ1: Model Comparison Across Key Metrics")

    return _save(fig, out / "fig_model_comparison")


# ── 2. Retrieval Strategy Comparison ──────────────────────────────────────

def plot_retrieval_comparison(df: pd.DataFrame, out: Path) -> Path | None:
    """Grouped bars: similarity vs MMR per model for faithfulness and nDCG@5."""
    if "retrieval_strategy" not in df.columns:
        return None

    sub = df.dropna(subset=["retrieval_strategy"]).copy()
    models = [m for m in MODEL_ORDER if m in sub["model"].unique()]
    strategies = sorted(sub["retrieval_strategy"].unique())
    if len(strategies) < 2 or not models:
        return None

    metrics = [
        ("ragas_faithfulness", "Faithfulness"),
        ("custom_ndcg_at_5", "nDCG@5"),
    ]

    strat_colors = {"similarity": "#2563EB", "mmr": "#DC2626"}

    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=False)

        for ax, (col, label) in zip(axes, metrics):
            x = np.arange(len(models))
            bar_width = 0.3

            for j, strat in enumerate(strategies):
                vals = []
                errs = []
                for model in models:
                    s = sub[(sub["model"] == model) & (sub["retrieval_strategy"] == strat)]
                    if col in s.columns:
                        clean = s[col].dropna()
                        if col.startswith("ragas_"):
                            clean = clean[clean > 0]
                        vals.append(clean.mean() if len(clean) > 0 else 0)
                        errs.append(clean.std() if len(clean) > 1 else 0)
                    else:
                        vals.append(0)
                        errs.append(0)

                offset = (j - len(strategies) / 2 + 0.5) * bar_width
                bars = ax.bar(
                    x + offset, vals, bar_width,
                    yerr=errs, capsize=2,
                    label=strat.title() if ax == axes[0] else "",
                    color=strat_colors.get(strat, "#888"),
                    edgecolor="white", linewidth=0.4,
                    error_kw={"linewidth": 0.8},
                )

                for bar, v in zip(bars, vals):
                    if v > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                                f"{v:.2f}", ha="center", va="bottom",
                                fontsize=6, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=15, ha="right")
            ax.set_ylabel(label)
            ax.set_ylim(0, 1.08)

        axes[0].legend(title="Retrieval")
        fig.suptitle("RQ2: Retrieval Strategy Comparison", fontsize=10, fontweight="bold")
        fig.tight_layout()

    return _save(fig, out / "fig_retrieval_comparison")


# ── 3. Prompt Template Effect ─────────────────────────────────────────────

def plot_template_effect(df: pd.DataFrame, out: Path) -> Path | None:
    """Grouped bars: concise vs detailed per model for faithfulness and METEOR."""
    if "prompt_template" not in df.columns:
        return None

    sub = df.dropna(subset=["prompt_template"]).copy()
    models = [m for m in MODEL_ORDER if m in sub["model"].unique()]
    templates = sorted(sub["prompt_template"].unique())
    if len(templates) < 2 or not models:
        return None

    metrics = [
        ("ragas_faithfulness", "Faithfulness"),
        ("custom_meteor", "METEOR"),
    ]

    tmpl_colors = {"concise": "#2563EB", "detailed": "#DC2626"}

    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=False)

        for ax, (col, label) in zip(axes, metrics):
            x = np.arange(len(models))
            bar_width = 0.3

            for j, tmpl in enumerate(templates):
                vals = []
                errs = []
                for model in models:
                    s = sub[(sub["model"] == model) & (sub["prompt_template"] == tmpl)]
                    if col in s.columns:
                        clean = s[col].dropna()
                        if col.startswith("ragas_"):
                            clean = clean[clean > 0]
                        vals.append(clean.mean() if len(clean) > 0 else 0)
                        errs.append(clean.std() if len(clean) > 1 else 0)
                    else:
                        vals.append(0)
                        errs.append(0)

                offset = (j - len(templates) / 2 + 0.5) * bar_width
                bars = ax.bar(
                    x + offset, vals, bar_width,
                    yerr=errs, capsize=2,
                    label=tmpl.title() if ax == axes[0] else "",
                    color=tmpl_colors.get(tmpl, "#888"),
                    edgecolor="white", linewidth=0.4,
                    error_kw={"linewidth": 0.8},
                )

                for bar, v in zip(bars, vals):
                    if v > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                                f"{v:.2f}", ha="center", va="bottom",
                                fontsize=6, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=15, ha="right")
            ax.set_ylabel(label)
            ax.set_ylim(0, 1.08)

        axes[0].legend(title="Template")
        fig.suptitle("RQ3: Prompt Template Effect", fontsize=10, fontweight="bold")
        fig.tight_layout()

    return _save(fig, out / "fig_template_effect")


# ── 4. Performance-Quality Tradeoff ───────────────────────────────────────

def plot_perf_quality(df: pd.DataFrame, out: Path) -> Path | None:
    """Scatter: TPS vs faithfulness, bubble = TTFT, per model."""
    sub = df.dropna(subset=["ragas_faithfulness", "avg_tokens_per_second"]).copy()
    sub = sub[sub["ragas_faithfulness"] > 0]
    sub = sub[sub["avg_tokens_per_second"] > 0]
    if len(sub) < 2:
        return None

    models = [m for m in MODEL_ORDER if m in sub["model"].unique()]

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(3.5, 3))

        for model in models:
            d = sub[sub["model"] == model]
            if d.empty:
                continue
            # Aggregate per model
            tps = d["avg_tokens_per_second"].mean()
            faith = d["ragas_faithfulness"].mean()
            ttft = d["avg_ttft_seconds"].dropna().mean() if "avg_ttft_seconds" in d.columns else 1
            size = max(40, min(200, 200 / max(ttft, 0.1)))

            ax.scatter(
                [tps], [faith],
                s=size, color=PALETTE.get(model, "#888"),
                label=model, edgecolors="white", linewidths=0.8,
                zorder=5, alpha=0.9,
            )
            ax.annotate(
                f"{faith:.2f}\n{tps:.0f} tok/s",
                (tps, faith), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=6, fontweight="bold",
            )

        ax.set_xlabel("Throughput (tokens/s)")
        ax.set_ylabel("Faithfulness")
        ax.set_ylim(0.7, 1.0)
        ax.legend(fontsize=6, loc="lower right")
        ax.set_title("RQ6: Quality vs. Speed Trade-off")

    return _save(fig, out / "fig_perf_quality")


# ── 5. Speed Performance Bars ─────────────────────────────────────────────

def plot_speed_bars(df: pd.DataFrame, out: Path) -> Path | None:
    """Side-by-side bars: TTFT and TPS per model."""
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    if not models:
        return None

    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

        # TTFT
        ax = axes[0]
        vals = []
        errs = []
        for model in models:
            s = df[(df["model"] == model) & (df["avg_ttft_seconds"].notna())]
            v = s["avg_ttft_seconds"].mean() if len(s) > 0 else 0
            e = s["avg_ttft_seconds"].std() if len(s) > 1 else 0
            vals.append(v)
            errs.append(e)

        bars = ax.barh(
            range(len(models)), vals,
            xerr=errs, capsize=2,
            color=[PALETTE.get(m, "#888") for m in models],
            edgecolor="white", linewidth=0.4, height=0.6,
            error_kw={"linewidth": 0.8},
        )
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(v + 0.05, i, f"{v:.2f}s", va="center", fontsize=7, fontweight="bold")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=7)
        ax.set_xlabel("Seconds (lower is better)")
        ax.set_title("Time to First Token")
        ax.invert_yaxis()

        # Throughput
        ax = axes[1]
        vals = []
        errs = []
        for model in models:
            s = df[(df["model"] == model) & (df["avg_tokens_per_second"].notna())]
            s = s[s["avg_tokens_per_second"] > 0]
            v = s["avg_tokens_per_second"].mean() if len(s) > 0 else 0
            e = s["avg_tokens_per_second"].std() if len(s) > 1 else 0
            vals.append(v)
            errs.append(e)

        bars = ax.barh(
            range(len(models)), vals,
            xerr=errs, capsize=2,
            color=[PALETTE.get(m, "#888") for m in models],
            edgecolor="white", linewidth=0.4, height=0.6,
            error_kw={"linewidth": 0.8},
        )
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=7, fontweight="bold")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=7)
        ax.set_xlabel("Tokens/s (higher is better)")
        ax.set_title("Generation Throughput")
        ax.invert_yaxis()

        fig.suptitle("RQ6: LLM Speed Performance", fontsize=10, fontweight="bold")
        fig.tight_layout()

    return _save(fig, out / "fig_speed_bars")


# ── 6. Interaction Heatmap ────────────────────────────────────────────────

def plot_interaction_heatmap(df: pd.DataFrame, out: Path) -> Path | None:
    """Heatmap: model x prompt template, value = mean faithfulness."""
    if "prompt_template" not in df.columns:
        return None

    sub = df.dropna(subset=["ragas_faithfulness", "prompt_template"]).copy()
    sub = sub[sub["ragas_faithfulness"] > 0]
    if len(sub) < 4:
        return None

    models = [m for m in MODEL_ORDER if m in sub["model"].unique()]
    templates = sorted(sub["prompt_template"].unique())

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(3.5, 3))

        data = np.full((len(models), len(templates)), np.nan)
        for i, model in enumerate(models):
            for j, tmpl in enumerate(templates):
                s = sub[(sub["model"] == model) & (sub["prompt_template"] == tmpl)]
                if len(s) > 0:
                    data[i, j] = s["ragas_faithfulness"].mean()

        im = ax.imshow(data, cmap="RdYlGn", vmin=0.7, vmax=1.0, aspect="auto")

        ax.set_xticks(range(len(templates)))
        ax.set_xticklabels([t.title() for t in templates])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)

        for i in range(len(models)):
            for j in range(len(templates)):
                if not np.isnan(data[i, j]):
                    v = data[i, j]
                    color = "white" if v < 0.78 or v > 0.95 else "black"
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=color)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Faithfulness")
        ax.set_title("RQ5: Model × Template Interaction")

    return _save(fig, out / "fig_interaction_heatmap")


# ── Main Pipeline ─────────────────────────────────────────────────────────

def generate_all(
    results_dir: str | Path = "results",
    output_dir: str | Path | None = None,
) -> list[Path]:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else Path("Paper/NGEN-AI/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = scan_data(results_dir)
    if df.empty:
        logger.error("No data found in %s", results_dir)
        return []

    logger.info("Generating NGEN-AI charts from %d configs, %d models",
                len(df), df["model"].nunique())

    generated: list[Path] = []
    chart_fns = [
        ("Model comparison", plot_model_comparison),
        ("Retrieval comparison", plot_retrieval_comparison),
        ("Template effect", plot_template_effect),
        ("Perf-quality scatter", plot_perf_quality),
        ("Speed bars", plot_speed_bars),
        ("Interaction heatmap", plot_interaction_heatmap),
    ]

    for name, fn in chart_fns:
        try:
            result = fn(df, output_dir)
            if result:
                generated.append(result)
            logger.info("  %s: %s", name, "OK" if result else "skipped")
        except Exception as e:
            logger.warning("  %s: failed — %s", name, e)

    print(f"\nGenerated {len(generated)} charts:")
    for p in generated:
        print(f"  {p.name}")
    print(f"  → {output_dir}/")

    return generated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate NGEN-AI paper charts")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="Paper/NGEN-AI/figures")
    args = parser.parse_args()

    generate_all(results_dir=args.results_dir, output_dir=args.output_dir)
