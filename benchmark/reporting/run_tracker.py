"""Cross-run metric tracker: scan all benchmark results and generate comparative plots."""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Modern Seaborn theme
sns.set_theme(style="whitegrid", font="sans-serif", palette="deep")

# Metric groupings for organized plotting
RAGAS_METRICS = [
    ("ragas_faithfulness", "Faithfulness"),
    ("ragas_answer_relevancy", "Answer Relevancy"),
    ("ragas_answer_correctness", "Answer Correctness"),
    ("ragas_context_precision", "Context Precision"),
    ("ragas_context_recall", "Context Recall"),
    ("ragas_semantic_similarity", "Semantic Similarity"),
]

PERFORMANCE_METRICS = [
    ("avg_ttft_seconds", "Avg TTFT (s)"),
    ("avg_tokens_per_second", "Avg Throughput (tok/s)"),
    ("total_time_seconds", "Total Time (s)"),
]

CUSTOM_METRICS = [
    ("custom_hit_at_1", "Hit@1"),
    ("custom_ndcg_at_1", "NDCG@1"),
    ("custom_rouge_l", "ROUGE-L"),
    ("custom_meteor", "METEOR"),
    ("custom_bert_score_f1", "BERTScore F1"),
    ("custom_context_relevance", "Context Relevance"),
]


def _extract_run_number(run_name: str) -> int:
    """Extract numeric run ID from directory name like 'run52' or 'run19_Sigurd_...'."""
    m = re.match(r"run(\d+)", run_name)
    return int(m.group(1)) if m else 0


def _merge_csv_custom_metrics(df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Merge custom metrics from results_summary.csv into the DataFrame.

    The benchmark JSONs may not include custom_metric_means, but the CSVs do.
    """
    csv_files = sorted(results_dir.glob("*/results_summary.csv"))
    if not csv_files:
        return df

    custom_dfs = []
    for csv_path in csv_files:
        run_name = csv_path.parent.name
        try:
            csv_df = pd.read_csv(csv_path)
        except Exception:
            continue

        # Find custom metric columns
        custom_cols = [c for c in csv_df.columns if c.startswith("custom_")]
        if not custom_cols:
            continue

        # Normalize column names: custom_hit@1 -> custom_hit_at_1
        rename_map = {c: c.replace("@", "_at_") for c in custom_cols if "@" in c}
        csv_df = csv_df.rename(columns=rename_map)
        custom_cols = [rename_map.get(c, c) for c in custom_cols]

        # Build merge key
        csv_df["run_name"] = run_name
        csv_df["config_name"] = csv_df.get("config_name", "")
        merge_cols = ["run_name", "config_name"] + [rename_map.get(c, c) for c in custom_cols]
        custom_dfs.append(csv_df[merge_cols])

    if not custom_dfs:
        return df

    csv_all = pd.concat(custom_dfs, ignore_index=True)

    # Merge on (run_name, config_name)
    df = df.merge(csv_all, on=["run_name", "config_name"], how="left", suffixes=("", "_csv"))

    # Fill missing custom columns from CSV
    for col in csv_all.columns:
        if col in ("run_name", "config_name"):
            continue
        csv_col = f"{col}_csv"
        if csv_col in df.columns:
            df[col] = df[col].fillna(df[csv_col])
            df = df.drop(columns=[csv_col])

    return df


def scan_all_results(results_dir: Path) -> pd.DataFrame:
    """Scan all benchmark_*.json files in results subdirectories.

    Returns a DataFrame with one row per config-result across all runs.
    """
    results_dir = Path(results_dir)
    json_files = sorted(results_dir.glob("*/benchmark_*.json"))

    if not json_files:
        logger.warning("No benchmark_*.json files found in %s", results_dir)
        return pd.DataFrame()

    records = []
    for jf in json_files:
        run_name = jf.parent.name
        run_number = _extract_run_number(run_name)

        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Skipping %s: %s", jf, e)
            continue

        for entry in data.get("results", []):
            rec = {
                "run_name": run_name,
                "run_number": run_number,
                "timestamp": data.get("timestamp", ""),
                "dataset": data.get("dataset", ""),
            }
            # Config params
            for key in [
                "config_name", "llm_model", "embedding_model",
                "chunk_size", "chunk_overlap", "chunking_strategy",
                "prompt_template", "retrieval_strategy", "retrieval_top_k",
                "reranker_model",
            ]:
                rec[key] = entry.get(key)

            # RAGAS metrics
            for key in [
                "ragas_faithfulness", "ragas_answer_relevancy",
                "ragas_answer_correctness", "ragas_context_precision",
                "ragas_context_recall", "ragas_semantic_similarity",
            ]:
                rec[key] = entry.get(key)

            # Performance metrics
            for key in [
                "avg_ttft_seconds", "avg_tokens_per_second",
                "total_time_seconds", "num_chunks", "num_questions",
                "avg_gpu_utilization_pct", "avg_gpu_memory_used_mb",
            ]:
                rec[key] = entry.get(key)

            # Custom metrics (flattened from dict if present)
            custom = entry.get("custom_metric_means", {})
            if isinstance(custom, dict):
                for k, v in custom.items():
                    safe = k.replace("@", "_at_")
                    rec[f"custom_{safe}"] = v

            records.append(rec)

    df = pd.DataFrame(records)

    if df.empty:
        return df

    # Merge custom metrics from results_summary.csv (JSONs may lack them)
    df = _merge_csv_custom_metrics(df, results_dir)

    # Short LLM label
    df["llm_short"] = df["llm_model"].apply(
        lambda x: x.split("/")[-1].replace(":", "_") if isinstance(x, str) else str(x)
    )

    # Sort by run number then config name
    df = df.sort_values(["run_number", "config_name"]).reset_index(drop=True)

    logger.info("Scanned %d configs across %d runs", len(df), df["run_name"].nunique())
    return df


def _short_config_label(row: pd.Series, max_len: int = 35) -> str:
    """Generate a short readable label for a config row."""
    llm = row.get("llm_short", "")
    cs = row.get("chunk_size", "")
    co = row.get("chunk_overlap", "")
    strat = row.get("chunking_strategy", "")[:3].lower()
    tmpl = row.get("prompt_template", "")[:1]
    label = f"{llm} | {strat}{cs}/{co} | {tmpl}"
    if len(label) > max_len:
        label = label[: max_len - 2] + ".."
    return label


def _paper_rc() -> dict:
    """Paper-ready matplotlib rcParams."""
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


def plot_metric_over_runs(
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    output_dir: Path,
    title_suffix: str = "",
) -> Path | None:
    """Generate a paper-ready horizontal bar chart (Top 15) for one metric.

    Returns path to generated PNG or None if no data.
    """
    subset = df.dropna(subset=[metric_col])
    if subset.empty:
        logger.info("No data for %s, skipping", metric_col)
        return None

    # Filter zero/null-ish for RAGAS
    if metric_col.startswith("ragas_"):
        subset = subset[subset[metric_col] > 0]

    if subset.empty:
        return None

    subset = subset.copy()
    subset["label"] = subset.apply(_short_config_label, axis=1)

    # Lower-is-better metrics: take smallest values
    lower_better = metric_col in ("avg_ttft_seconds", "total_time_seconds")
    top_n = subset.nsmallest(15, metric_col) if lower_better else subset.nlargest(15, metric_col)
    top_n = top_n.sort_values(metric_col, ascending=True)

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(7, max(4, len(top_n) * 0.35)))

        llm_models = sorted(top_n["llm_short"].unique())
        palette = sns.color_palette("deep", n_colors=max(len(llm_models), 1))
        color_map = dict(zip(llm_models, palette))

        bars = ax.barh(
            range(len(top_n)),
            top_n[metric_col],
            color=[color_map.get(llm, "#4C78A8") for llm in top_n["llm_short"]],
            edgecolor="white",
            linewidth=0.5,
            height=0.7,
        )

        # Score labels on bars
        for i, (_, row) in enumerate(top_n.iterrows()):
            val = row[metric_col]
            ax.text(
                val + (top_n[metric_col].max() * 0.005),
                i,
                f"{val:.3f}",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        # Rank badge for #1
        ax.scatter(
            [top_n.iloc[-1][metric_col]],
            [len(top_n) - 1],
            s=120,
            color="gold",
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
            marker="*",
        )

        ax.set_yticks(range(len(top_n)))
        ax.set_yticklabels(top_n["label"], fontsize=9)
        ax.set_xlabel(metric_label, fontsize=11, fontweight="bold")
        lower_suffix = " (lowest is best)" if lower_better else ""
        ax.set_title(f"Top 15 — {metric_label}{lower_suffix}{title_suffix}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlim(0, top_n[metric_col].max() * 1.12)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25, linestyle="--")

        # LLM legend — outside plot to avoid overlapping values
        import matplotlib.patches as mpatches
        handles = [mpatches.Patch(color=color_map[llm], label=llm) for llm in llm_models]
        legend = ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, framealpha=0.9, borderaxespad=0)

        fig.tight_layout()

        safe_name = metric_col.replace(" ", "_")
        path = output_dir / f"metric_{safe_name}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)
        return path


def plot_llm_boxplots(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Generate boxplots showing metric distributions per LLM model."""
    ragas_cols = [(c, l) for c, l in RAGAS_METRICS if c in df.columns and df[c].notna().any() and (df[c] > 0).any()]

    if not ragas_cols:
        return None

    n = len(ragas_cols)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols

    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        llm_models = sorted(df["llm_short"].dropna().unique())
        palette = sns.color_palette("deep", n_colors=len(llm_models))

        for idx, (col, label) in enumerate(ragas_cols):
            ax = axes[idx]
            subset = df.dropna(subset=[col])
            subset = subset[subset[col] > 0]

            if subset.empty:
                ax.set_visible(False)
                continue

            sns.boxplot(
                data=subset,
                x="llm_short",
                y=col,
                order=llm_models,
                hue="llm_short",
                palette=palette,
                ax=ax,
                width=0.5,
                fliersize=3,
                legend=False,
            )
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Score", fontsize=9)
            ax.tick_params(axis="x", rotation=20, labelsize=8)
            ax.set_ylim(bottom=0)

        for idx in range(len(ragas_cols), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("RAGAS Score Distributions by LLM", fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()

        path = output_dir / "llm_boxplots.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)
        return path


def plot_overview_grid(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Generate a paper-ready multi-panel overview grid."""
    key_metrics = [
        ("ragas_faithfulness", "Faithfulness"),
        ("ragas_context_recall", "Context Recall"),
        ("ragas_answer_relevancy", "Answer Relevancy"),
        ("ragas_context_precision", "Context Precision"),
        ("avg_ttft_seconds", "TTFT (s)"),
        ("avg_tokens_per_second", "Throughput (tok/s)"),
    ]

    available = [(c, l) for c, l in key_metrics if c in df.columns and df[c].notna().any()]
    if not available:
        return None

    n_metrics = len(available)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.5 * n_rows))
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (col, label) in enumerate(available):
            ax = axes[idx]
            subset = df.dropna(subset=[col])
            if col.startswith("ragas_"):
                subset = subset[subset[col] > 0]
            if subset.empty:
                ax.set_visible(False)
                continue

            subset = subset.copy()
            subset["label"] = subset.apply(_short_config_label, axis=1)

            lower_better = col in ("avg_ttft_seconds", "total_time_seconds")
            top_n = subset.nsmallest(10, col) if lower_better else subset.nlargest(10, col)
            top_n = top_n.sort_values(col, ascending=True)

            llm_models = sorted(top_n["llm_short"].unique())
            palette = sns.color_palette("deep", n_colors=max(len(llm_models), 1))
            color_map = dict(zip(llm_models, palette))

            ax.barh(
                range(len(top_n)),
                top_n[col],
                color=[color_map.get(llm, "#4C78A8") for llm in top_n["llm_short"]],
                edgecolor="white",
                linewidth=0.3,
                height=0.7,
            )
            ax.set_yticks(range(len(top_n)))
            ax.set_yticklabels(top_n["label"], fontsize=7)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.2, linestyle="--")

        for idx in range(len(available), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Cross-Run Overview (Top 10 per Metric)", fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()

        path = output_dir / "overview_grid.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)
        return path


def plot_ranking(df: pd.DataFrame, output_dir: Path, top_k: int = 20) -> Path | None:
    """Generate a paper-ready composite ranking bar chart."""
    ragas_cols = {
        "ragas_faithfulness": 0.30,
        "ragas_context_recall": 0.25,
        "ragas_answer_relevancy": 0.15,
        "ragas_context_precision": 0.15,
        "ragas_answer_correctness": 0.15,
    }

    available_weights = {k: v for k, v in ragas_cols.items() if k in df.columns and df[k].notna().any()}

    if not available_weights:
        logger.warning("No RAGAS metrics available for ranking")
        return None

    total = sum(available_weights.values())
    available_weights = {k: v / total for k, v in available_weights.items()}

    score_df = df.copy()
    for col in available_weights:
        score_df[col] = score_df[col].fillna(0)

    score_df["composite"] = sum(
        score_df[col] * weight for col, weight in available_weights.items()
    )

    score_df = score_df[score_df["composite"] > 0]
    if score_df.empty:
        return None

    top = score_df.nlargest(top_k, "composite").copy()
    top["label"] = top.apply(_short_config_label, axis=1)
    top = top.sort_values("composite", ascending=True)

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.4)))

        colors = sns.color_palette("YlOrRd_r", n_colors=len(top))

        ax.barh(
            range(len(top)),
            top["composite"],
            color=colors,
            edgecolor="white",
            linewidth=0.5,
        )

        for i, (_, row) in enumerate(top.iterrows()):
            ax.text(
                row["composite"] + 0.005,
                i,
                f"{row['composite']:.3f}",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["label"], fontsize=8)
        ax.set_xlabel("Composite Score", fontsize=11, fontweight="bold")
        ax.set_title(f"Top-{len(top)} Configuration Ranking", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlim(0, top["composite"].max() * 1.1)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25, linestyle="--")

        fig.tight_layout()
        path = output_dir / "ranking_top20.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)
        return path


def main(
    results_dir: str | Path = "results",
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Run the full cross-run tracking pipeline."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir / "cross_run_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = scan_all_results(results_dir)
    if df.empty:
        logger.error("No data found in %s", results_dir)
        return []

    generated: list[Path] = []

    # Filter: only plot metrics present in majority of runs (>= 50%)
    run_names = df["run_name"].unique()
    n_runs = len(run_names)

    def _metric_coverage(col: str) -> int:
        """Return number of runs with data for this metric, or 0 if below threshold."""
        if col not in df.columns:
            return 0
        runs_with_data = df.dropna(subset=[col])["run_name"].nunique()
        return runs_with_data

    # Per-metric plots: RAGAS
    for col, label in RAGAS_METRICS:
        cov = _metric_coverage(col)
        if cov == 0:
            continue
        suffix = f" ({cov}/{n_runs} runs)" if cov < n_runs else ""
        path = plot_metric_over_runs(df, col, label, output_dir, title_suffix=suffix)
        if path:
            generated.append(path)

    # Per-metric plots: Performance
    for col, label in PERFORMANCE_METRICS:
        cov = _metric_coverage(col)
        if cov == 0:
            continue
        suffix = f" ({cov}/{n_runs} runs)" if cov < n_runs else ""
        path = plot_metric_over_runs(df, col, label, output_dir, title_suffix=suffix)
        if path:
            generated.append(path)

    # Per-metric plots: Custom metrics (mathematische Metriken)
    for col, label in CUSTOM_METRICS:
        cov = _metric_coverage(col)
        if cov == 0:
            continue
        suffix = f" ({cov}/{n_runs} runs)" if cov < n_runs else ""
        path = plot_metric_over_runs(df, col, label, output_dir, title_suffix=suffix)
        if path:
            generated.append(path)

    # Overview grid
    overview = plot_overview_grid(df, output_dir)
    if overview:
        generated.append(overview)

    # LLM boxplots (distribution per model)
    boxplots = plot_llm_boxplots(df, output_dir)
    if boxplots:
        generated.append(boxplots)

    # Ranking
    ranking = plot_ranking(df, output_dir)
    if ranking:
        generated.append(ranking)

    logger.info("Generated %d plots in %s", len(generated), output_dir)
    print(f"\nGenerated {len(generated)} plots in {output_dir}/")
    for p in generated:
        print(f"  {p.name}")

    return generated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Cross-run benchmark metric tracker")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing run results (default: results/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: results/cross_run_plots/)",
    )
    args = parser.parse_args()

    main(results_dir=args.results_dir, output_dir=args.output_dir)
