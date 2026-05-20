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

NLG_EXTRA_METRICS = [
    ("bleu", "BLEU"),
    ("bert_score_precision", "BERTScore Precision"),
    ("bert_score_recall", "BERTScore Recall"),
]

QUALITY_METRICS = [
    ("ragas_faithfulness", "Faithfulness", 0.25),
    ("custom_context_relevance", "Context Relevance", 0.20),
    ("custom_ndcg_at_5", "NDCG@5", 0.20),
    ("custom_recall_at_5", "Recall@5", 0.15),
    ("custom_hit_at_1", "Hit@1", 0.10),
    ("custom_bert_score_f1", "BERTScore F1", 0.10),
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

            # IR retrieval metrics at higher k
            for key in [
                "hit@1", "hit@3", "hit@5",
                "ndcg@1", "ndcg@3", "ndcg@5",
                "recall@1", "recall@3", "recall@5",
            ]:
                safe = key.replace("@", "_at_")
                rec[f"custom_{safe}"] = entry.get(key)

            # NLG metrics (flat keys in JSON)
            for key in ["bleu", "rouge_l", "meteor", "bert_score_f1",
                        "bert_score_precision", "bert_score_recall"]:
                rec[key] = entry.get(key)

            # Context relevance + vector distances
            rec["context_relevance"] = entry.get("context_relevance")
            rec["vec_dist_q_answer"] = entry.get("vec_dist_q_answer")
            rec["vec_dist_q_gt"] = entry.get("vec_dist_q_gt")

            # Stage timings
            timings = entry.get("stage_timings", {})
            if isinstance(timings, dict):
                for stage, t in timings.items():
                    rec[f"time_{stage}"] = t

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


def _short_config_label(row: pd.Series, max_len: int = 55) -> str:
    """Generate a readable label for a config row."""
    run = row.get("run_number", "")
    run_prefix = f"Run {run} | " if run else ""
    llm = row.get("llm_short", "")
    strat = row.get("chunking_strategy", "")
    cs = row.get("chunk_size", "")
    co = row.get("chunk_overlap", "")
    parts = [p for p in [run_prefix, llm, strat, f"cs{cs}", f"co{co}"] if p]
    label = " | ".join(parts)
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


def _ci95(series: pd.Series) -> float:
    """Compute 95% confidence interval half-width for a pandas Series."""
    n = len(series)
    if n < 2:
        return 0.0
    std = series.std(ddof=1)
    from scipy import stats as _stats
    t_crit = _stats.t.ppf(0.975, df=n - 1)
    return t_crit * std / np.sqrt(n)



def _coverage_label(df: pd.DataFrame, col: str) -> str:
    """Return compact coverage label for a metric column."""
    return f"n={df[col].notna().sum()}/{len(df)}"


def _compute_quality_scores(df: pd.DataFrame, min_coverage: float = 0.5) -> tuple[pd.DataFrame, list[tuple[str, str, float]]]:
    """Add a quality score from metrics with enough cross-run coverage.

    Timing metrics are intentionally excluded because they are sparsely available.
    Each row must have at least 70% of the selected score weight to avoid ranking
    rows from one or two metrics only.
    """
    n_rows = len(df)
    selected = [
        (col, label, weight)
        for col, label, weight in QUALITY_METRICS
        if col in df.columns and df[col].notna().sum() >= max(1, int(n_rows * min_coverage))
    ]
    if not selected:
        out = df.copy()
        out["quality_score"] = np.nan
        out["quality_weight_available"] = 0.0
        out["quality_metric_count"] = 0
        return out, []

    total_weight = sum(weight for _, _, weight in selected)
    out = df.copy()
    weighted_sum = pd.Series(0.0, index=out.index)
    available_weight = pd.Series(0.0, index=out.index)
    metric_count = pd.Series(0, index=out.index)

    for col, _, weight in selected:
        vals = pd.to_numeric(out[col], errors="coerce")
        mask = vals.notna()
        weighted_sum.loc[mask] += vals.loc[mask] * weight
        available_weight.loc[mask] += weight
        metric_count.loc[mask] += 1

    out["quality_weight_available"] = available_weight
    out["quality_metric_count"] = metric_count
    out["quality_score"] = weighted_sum / available_weight.replace(0, np.nan)
    out.loc[available_weight < total_weight * 0.70, "quality_score"] = np.nan
    return out, selected


def _assign_config_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign compact Cxx IDs for display-heavy paper plots."""
    out = df.copy().reset_index(drop=True)
    out["config_id"] = [f"C{i:02d}" for i in range(1, len(out) + 1)]
    return out


def _config_table_label(row: pd.Series) -> str:
    """Return compact but interpretable config description for legends/tables."""
    return (
        f"{row.get('llm_short', '')}; "
        f"{row.get('prompt_template', '')}; "
        f"{row.get('chunking_strategy', '')}; "
        f"cs={row.get('chunk_size', '')}; co={row.get('chunk_overlap', '')}; "
        f"run={row.get('run_name', '')}"
    )


def plot_paper_quality_ranking(df: pd.DataFrame, output_dir: Path, top_k: int = 10) -> Path | None:
    """Generate poster-ready Top-K quality ranking with metric contributions."""
    scored, selected = _compute_quality_scores(df)
    scored = scored.dropna(subset=["quality_score"])
    if scored.empty or not selected:
        return None

    top = _assign_config_ids(scored.nlargest(top_k, "quality_score"))
    top = top.sort_values("quality_score", ascending=True)

    with plt.rc_context({**_paper_rc(), "font.size": 11, "axes.titlesize": 13}):
        fig, ax = plt.subplots(figsize=(10.5, max(5, len(top) * 0.48)))
        contribution_colors = sns.color_palette("colorblind", n_colors=len(selected))
        left = np.zeros(len(top))

        for i, (col, label, weight) in enumerate(selected):
            vals = pd.to_numeric(top[col], errors="coerce").fillna(0).values * weight
            ax.barh(
                range(len(top)), vals, left=left, label=f"{label} ({weight:.2f})",
                color=contribution_colors[i], edgecolor="white", linewidth=0.4, height=0.7,
            )
            left += vals

        for idx, (_, row) in enumerate(top.iterrows()):
            ax.text(left[idx] + 0.008, idx, f"{row['quality_score']:.3f}", va="center", fontsize=9, fontweight="bold")

        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["config_id"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Weighted Quality Contribution", fontsize=11, fontweight="bold")
        ax.set_title("Top Configurations by Quality Score", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlim(0, max(left) * 1.15)
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=8)

        table_text = "\n".join(
            f"{row['config_id']}: {_config_table_label(row)}"
            for _, row in top.sort_values("quality_score", ascending=False).iterrows()
        )
        fig.text(0.01, -0.02, table_text, ha="left", va="top", fontsize=8)

        coverage = "; ".join(f"{label} {_coverage_label(df, col)}" for col, label, _ in selected)
        fig.text(0.01, 0.01, f"Composite excludes timing metrics. Coverage: {coverage}", ha="left", fontsize=8)

        fig.tight_layout(rect=(0, 0.08, 0.82, 1))
        path = output_dir / "paper_quality_ranking_top10.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)
        return path


def plot_paper_quality_load_proxy(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Scatter quality against chunk-count proxy for computational load."""
    scored, _ = _compute_quality_scores(df)
    subset = scored.dropna(subset=["quality_score", "num_chunks"])
    if len(subset) < 5:
        return None

    with plt.rc_context({**_paper_rc(), "font.size": 11, "axes.titlesize": 13}):
        fig, ax = plt.subplots(figsize=(8.5, 6.2))
        llm_models = sorted(subset["llm_short"].dropna().unique())
        palette = sns.color_palette("colorblind", n_colors=len(llm_models))
        color_map = dict(zip(llm_models, palette))
        markers = {"recursive": "o", "semantic": "s"}

        for (llm, strat), group in subset.groupby(["llm_short", "chunking_strategy"], dropna=False):
            ax.scatter(
                group["num_chunks"], group["quality_score"], label=f"{llm} / {strat}",
                color=color_map.get(llm, "#4C78A8"), marker=markers.get(str(strat), "D"),
                s=55, alpha=0.72, edgecolors="white", linewidths=0.5,
            )

        top = subset.nlargest(4, "quality_score")
        for i, (_, row) in enumerate(top.iterrows(), start=1):
            ax.annotate(f"T{i}", (row["num_chunks"], row["quality_score"]), xytext=(5, 5), textcoords="offset points", fontsize=9, fontweight="bold")

        ax.set_xscale("log")
        ax.set_xlabel("Indexed Chunk Count (log scale; compute/load proxy)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Quality Score (timing excluded)", fontsize=11, fontweight="bold")
        ax.set_title("Quality vs Retrieval Load Proxy", fontsize=14, fontweight="bold", pad=10)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=True)
        fig.tight_layout()

        path = output_dir / "paper_quality_vs_load_proxy.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)
        return path


def plot_paper_llm_heatmap(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Generate LLM x metric heatmap with sample counts."""
    scored, selected = _compute_quality_scores(df)
    metric_cols = [(col, label) for col, label, _ in selected]
    if not metric_cols:
        return None

    cols = [col for col, _ in metric_cols]
    grouped = scored.groupby("llm_short")[cols + ["quality_score"]].mean()
    counts = scored.groupby("llm_short")["quality_score"].count()
    grouped = grouped.dropna(how="all")
    if grouped.empty:
        return None

    display = grouped.rename(columns=dict(metric_cols))
    display["Quality"] = grouped["quality_score"]
    row_labels = [f"{idx} (n={int(counts.get(idx, 0))})" for idx in display.index]

    with plt.rc_context({**_paper_rc(), "font.size": 11, "axes.titlesize": 13}):
        fig, ax = plt.subplots(figsize=(9.5, max(3.5, len(display) * 0.7)))
        sns.heatmap(
            display, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=1,
            linewidths=0.5, linecolor="white", yticklabels=row_labels, ax=ax,
            cbar_kws={"label": "Mean Score"},
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("LLM Mean Metric Profile", fontsize=14, fontweight="bold", pad=10)
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)
        fig.tight_layout()

        path = output_dir / "paper_llm_metric_heatmap.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)
        return path



def plot_paper_faithfulness_by_llm_prompt(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Poster-ready Faithfulness comparison with mean CI and visible outlier note."""
    metric = "ragas_faithfulness"
    if metric not in df.columns or df[metric].notna().sum() < 3:
        return None

    subset = df.dropna(subset=[metric, "llm_short", "prompt_template"]).copy()
    subset = subset[subset[metric] > 0]
    if subset.empty:
        return None

    llm_order = (
        subset.groupby("llm_short")[metric]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    prompt_order = sorted(subset["prompt_template"].dropna().unique())
    summary = (
        subset.groupby(["llm_short", "prompt_template"])[metric]
        .agg(mean="mean", median="median", std="std", count="count", min="min")
        .reset_index()
    )
    summary["ci"] = summary.apply(lambda r: _stats_t_ci(r["std"], r["count"]), axis=1)
    summary["ci"] = summary["ci"].fillna(0)
    x_base = {llm: i for i, llm in enumerate(llm_order)}
    offsets = np.linspace(-0.18, 0.18, len(prompt_order)) if len(prompt_order) > 1 else [0]
    prompt_offset = dict(zip(prompt_order, offsets))
    palette = sns.color_palette("colorblind", n_colors=max(len(prompt_order), 1))
    prompt_color = dict(zip(prompt_order, palette))

    low_outliers = subset[subset[metric] < 0.70]
    y_focus_min = max(0.70, min(0.72, subset[metric].quantile(0.05) - 0.02))

    with plt.rc_context({**_paper_rc(), "font.size": 12, "axes.titlesize": 14}):
        fig, ax = plt.subplots(figsize=(9.5, 5.6))

        for prompt in prompt_order:
            part = summary[summary["prompt_template"] == prompt]
            xs = [x_base[row["llm_short"]] + prompt_offset[prompt] for _, row in part.iterrows()]
            ax.errorbar(
                xs,
                part["mean"],
                yerr=part["ci"],
                fmt="o",
                markersize=8,
                capsize=5,
                elinewidth=1.4,
                markeredgecolor="white",
                markeredgewidth=0.8,
                color=prompt_color[prompt],
                label=prompt,
                zorder=4,
            )
            for x, (_, row) in zip(xs, part.iterrows()):
                ax.text(x, row["mean"] + row["ci"] + 0.008, f"n={int(row['count'])}", ha="center", va="bottom", fontsize=8)

        # Show faint raw points only in the focused range; low outliers are counted in the note.
        visible = subset[subset[metric] >= y_focus_min]
        for prompt in prompt_order:
            part = visible[visible["prompt_template"] == prompt]
            xs = [x_base[row["llm_short"]] + prompt_offset[prompt] for _, row in part.iterrows()]
            jitter = np.random.default_rng(7).normal(0, 0.025, len(xs)) if xs else []
            ax.scatter(
                np.array(xs) + jitter,
                part[metric],
                s=16,
                alpha=0.28,
                color=prompt_color[prompt],
                edgecolors="none",
                zorder=2,
            )

        ax.set_xticks(range(len(llm_order)))
        ax.set_xticklabels(llm_order, rotation=10, ha="right")
        ax.set_ylabel("Faithfulness", fontsize=12, fontweight="bold")
        ax.set_ylim(y_focus_min, 1.01)
        ax.set_title(
            f"Faithfulness by LLM and Prompt Template (mean +/- 95% CI, n={len(subset)}/{len(df)})",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.legend(title="Prompt", frameon=True, fontsize=10, title_fontsize=10, loc="lower right")

        note = "Zoomed y-axis; low outliers below axis: "
        if low_outliers.empty:
            note += "none"
        else:
            grouped = low_outliers.groupby(["llm_short", "prompt_template"])[metric].count()
            note += "; ".join(f"{llm}/{prompt}: {int(n)}" for (llm, prompt), n in grouped.items())
        fig.text(0.01, 0.01, note, ha="left", fontsize=8)
        fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.20)

        path = output_dir / "paper_faithfulness_by_llm_prompt.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        logger.info("Saved %s", path)
        return path


def plot_paper_retrieval_by_strategy(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Poster-ready retrieval metric comparison grouped by chunking strategy."""
    metric_groups = [
        ("Hit@1", "custom_hit_at_1"), ("Hit@3", "custom_hit_at_3"), ("Hit@5", "custom_hit_at_5"),
        ("NDCG@1", "custom_ndcg_at_1"), ("NDCG@3", "custom_ndcg_at_3"), ("NDCG@5", "custom_ndcg_at_5"),
        ("Recall@1", "custom_recall_at_1"), ("Recall@3", "custom_recall_at_3"), ("Recall@5", "custom_recall_at_5"),
    ]
    available = [(label, col) for label, col in metric_groups if col in df.columns and df[col].notna().any()]
    if not available or "chunking_strategy" not in df.columns:
        return None

    subset = df.dropna(subset=[col for _, col in available], how="all").copy()
    if subset.empty:
        return None

    strategies = sorted(subset["chunking_strategy"].dropna().unique())
    if not strategies:
        return None

    means = subset.groupby("chunking_strategy")[[col for _, col in available]].mean().reindex(strategies)
    x = np.arange(len(available))
    width = 0.8 / len(strategies)
    palette = sns.color_palette("colorblind", n_colors=len(strategies))

    with plt.rc_context({**_paper_rc(), "font.size": 11, "axes.titlesize": 13}):
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for i, strat in enumerate(strategies):
            offset = (i - len(strategies) / 2 + 0.5) * width
            ax.bar(
                x + offset, means.loc[strat].values, width,
                label=f"{strat} (n={int((subset['chunking_strategy'] == strat).sum())})",
                color=palette[i], edgecolor="white", linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _ in available], rotation=35, ha="right")
        ax.set_ylabel("Mean Score", fontsize=11, fontweight="bold")
        ax.set_title("Retrieval Quality by Chunking Strategy", fontsize=14, fontweight="bold", pad=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.legend(frameon=True, fontsize=9)
        fig.tight_layout()

        path = output_dir / "paper_retrieval_by_strategy.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)
        return path


def plot_paper_timing_subset(df: pd.DataFrame, output_dir: Path, top_k: int = 8) -> Path | None:
    """Generate separate timing-subset plot; not part of main quality score."""
    scored, _ = _compute_quality_scores(df)
    subset = scored.dropna(subset=["total_time_seconds"])
    if len(subset) < 3:
        return None

    subset = subset.nsmallest(top_k, "total_time_seconds").sort_values("total_time_seconds", ascending=True)
    subset = _assign_config_ids(subset)
    has_quality = subset["quality_score"].notna().any()

    with plt.rc_context({**_paper_rc(), "font.size": 11, "axes.titlesize": 13}):
        fig, ax1 = plt.subplots(figsize=(10, 5.5))
        x = np.arange(len(subset))
        colors = sns.color_palette("colorblind", n_colors=len(subset))
        bars = ax1.bar(x, subset["total_time_seconds"], color=colors, edgecolor="white", linewidth=0.5)
        ax1.set_ylabel("Total Time (s)", fontsize=11, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(subset["config_id"], fontsize=10, fontweight="bold")
        ax1.grid(axis="y", alpha=0.25, linestyle="--")

        if has_quality:
            ax2 = ax1.twinx()
            ax2.plot(x, subset["quality_score"], color="#222222", marker="o", linewidth=1.8, label="Quality")
            ax2.set_ylabel("Quality Score", fontsize=11, fontweight="bold")
            ax2.set_ylim(0, 1.05)

        for bar, val in zip(bars, subset["total_time_seconds"]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.0f}", ha="center", va="bottom", fontsize=8)

        subtitle = "Quality overlay shown where complete metrics are available" if has_quality else "No quality overlay: timing rows lack complete quality metric coverage"
        ax1.set_title(
            f"Runtime Subset Only: Fastest Configurations (n={len(df['total_time_seconds'].dropna())}/{len(df)})\n{subtitle}",
            fontsize=13, fontweight="bold", pad=10,
        )
        table_text = "\n".join(f"{row['config_id']}: {_config_table_label(row)}" for _, row in subset.iterrows())
        fig.text(0.01, -0.02, table_text, ha="left", va="top", fontsize=8)
        fig.tight_layout(rect=(0, 0.10, 1, 1))

        path = output_dir / "paper_timing_subset.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)
        return path


def plot_metric_over_runs(
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    output_dir: Path,
    title_suffix: str = "",
) -> Path | None:
    """Generate a paper-ready horizontal bar chart (Top 15) for one metric.

    Includes 95% CI error bars when multiple observations share a config group.
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

    # Compute CI per row based on same (llm_short, chunking_strategy, chunk_size) group
    group_cols = ["llm_short", "chunking_strategy", "chunk_size"]
    ci_map = subset.groupby(group_cols)[metric_col].agg(_ci95).to_dict()
    top_n["_ci"] = top_n.apply(
        lambda r: ci_map.get((r["llm_short"], r.get("chunking_strategy"), r.get("chunk_size")), 0.0),
        axis=1,
    )

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(9, max(4, len(top_n) * 0.35)))

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

        # CI error bars
        ax.errorbar(
            top_n[metric_col],
            range(len(top_n)),
            xerr=top_n["_ci"],
            fmt="none",
            ecolor="#333333",
            elinewidth=1.0,
            capsize=3,
            capthick=0.8,
            zorder=5,
        )

        # Score labels on bars (with CI annotation)
        for i, (_, row) in enumerate(top_n.iterrows()):
            val = row[metric_col]
            ci = row["_ci"]
            ci_str = f" ±{ci:.3f}" if ci > 0.001 else ""
            ax.text(
                val + (top_n[metric_col].max() * 0.005) + ci,
                i,
                f"{val:.3f}{ci_str}",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

        ax.set_yticks(range(len(top_n)))
        ax.set_yticklabels(top_n["label"], fontsize=9)
        ax.set_xlabel(metric_label, fontsize=11, fontweight="bold")
        lower_suffix = " (lowest is best)" if lower_better else ""
        ax.set_title(f"Top 15 — {metric_label}{lower_suffix}{title_suffix}", fontsize=12, fontweight="bold", pad=10)

        # Tight x-axis for metrics with narrow value range
        val_min = top_n[metric_col].min()
        val_max = top_n[metric_col].max()
        val_range = val_max - val_min
        if not lower_better and val_min > 0 and val_range < 0.3 * val_max:
            ax.set_xlim(val_min * 0.95, val_max * 1.05)
        else:
            ax.set_xlim(0, val_max * 1.12)
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
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
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
            ax.tick_params(axis="x", rotation=0, labelsize=8)
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
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 4.5 * n_rows))
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
            top_n = subset.nsmallest(5, col) if lower_better else subset.nlargest(5, col)
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
            ax.set_yticklabels(top_n["label"], fontsize=9)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.2, linestyle="--")

        for idx in range(len(available), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Cross-Run Overview (Top 5 per Metric)", fontsize=13, fontweight="bold", y=1.01)
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


def plot_retrieval_level_comparison(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Grouped bar chart comparing retrieval metrics at k=1,3,5."""
    metric_groups = [
        ("Hit", "custom_hit_at_1", "custom_hit_at_3", "custom_hit_at_5"),
        ("NDCG", "custom_ndcg_at_1", "custom_ndcg_at_3", "custom_ndcg_at_5"),
        ("Recall", "custom_recall_at_1", "custom_recall_at_3", "custom_recall_at_5"),
    ]

    available_groups = []
    for name, k1, k3, k5 in metric_groups:
        cols_present = [c for c in (k1, k3, k5) if c in df.columns and df[c].notna().any()]
        if len(cols_present) >= 2:
            available_groups.append((name, k1, k3, k5))

    if not available_groups:
        return None

    n_groups = len(available_groups)
    k_levels = ["k=1", "k=3", "k=5"]
    k_palette = sns.color_palette("deep", n_colors=3)

    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 5))
        if n_groups == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for gidx, (name, k1, k3, k5) in enumerate(available_groups):
            ax = axes[gidx]
            means = []
            for col in (k1, k3, k5):
                if col in df.columns:
                    means.append(df[col].mean())
                else:
                    means.append(np.nan)

            x = np.arange(len(means))
            bars = ax.bar(x, means, color=k_palette[:len(means)], edgecolor="white", linewidth=0.5, width=0.6)
            for bar, val in zip(bars, means):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(k_levels[:len(means)], fontsize=10)
            ax.set_title(name, fontsize=12, fontweight="bold")
            ax.set_ylabel("Mean Score", fontsize=10)
            ax.set_ylim(0, min(1.05, max(m for m in means if not np.isnan(m)) * 1.15) if means else 1.0)
            ax.grid(axis="y", alpha=0.2, linestyle="--")

        fig.suptitle("Retrieval Metrics by k (Mean over all configs)", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        path = output_dir / "retrieval_level_comparison.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)
        return path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Generate a correlation heatmap across all numeric metrics."""
    metric_cols = []
    all_metrics = (
        [(c, l) for c, l in RAGAS_METRICS]
        + [(c, l) for c, l in CUSTOM_METRICS]
        + [(c, l) for c, l in NLG_EXTRA_METRICS]
    )
    for col, label in all_metrics:
        if col in df.columns and df[col].notna().sum() >= 5:
            metric_cols.append((col, label))

    if len(metric_cols) < 4:
        return None

    cols = [c for c, _ in metric_cols]
    labels = [l for _, l in metric_cols]
    corr = df[cols].corr()

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.7), max(7, len(cols) * 0.65)))

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1,
            xticklabels=labels, yticklabels=labels,
            square=True, linewidths=0.5, ax=ax,
            annot_kws={"fontsize": 7},
        )
        ax.set_title("Metric Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        fig.tight_layout()
        path = output_dir / "correlation_heatmap.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)
        return path


def plot_radar_per_llm(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Generate radar/spider chart showing mean metric profile per LLM."""
    radar_metrics = [
        ("ragas_faithfulness", "Faithfulness"),
        ("ragas_answer_relevancy", "Ans. Relevancy"),
        ("ragas_context_recall", "Ctx. Recall"),
        ("ragas_context_precision", "Ctx. Precision"),
        ("custom_hit_at_1", "Hit@1"),
        ("custom_ndcg_at_1", "NDCG@1"),
    ]

    available = [(c, l) for c, l in radar_metrics if c in df.columns and df[c].notna().any()]
    if len(available) < 3:
        return None

    cols = [c for c, _ in available]
    labels = [l for _, l in available]

    llm_means = df.groupby("llm_short")[cols].mean().dropna(how="all")
    if llm_means.empty:
        return None

    # Keep only top 4 LLMs by mean metric score
    llm_means["__overall"] = llm_means.mean(axis=1)
    llm_means = llm_means.nlargest(4, "__overall").drop(columns=["__overall"])

    # Normalize to 0-1 for radar
    mins = llm_means.min()
    maxs = llm_means.max()
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    normed = (llm_means - mins) / ranges

    n_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    llm_models = sorted(normed.index)
    palette = sns.color_palette("deep", n_colors=len(llm_models))

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for i, llm in enumerate(llm_models):
            values = normed.loc[llm].tolist()
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=1.5, label=llm, color=palette[i], markersize=4)
            ax.fill(angles, values, alpha=0.08, color=palette[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=7)
        ax.set_title("LLM Metric Profiles (Normalized)", fontsize=13, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

        fig.tight_layout()
        path = output_dir / "radar_llm_profiles.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)
        return path


def plot_vector_distance_scatter(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Scatter plot: vec_dist_q_answer vs vec_dist_q_gt per config."""
    subset = df.dropna(subset=["vec_dist_q_answer", "vec_dist_q_gt"])
    if len(subset) < 5:
        return None

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(8, 7))

        llm_models = sorted(subset["llm_short"].unique())
        palette = sns.color_palette("deep", n_colors=len(llm_models))
        color_map = dict(zip(llm_models, palette))

        for llm in llm_models:
            group = subset[subset["llm_short"] == llm]
            ax.scatter(
                group["vec_dist_q_gt"], group["vec_dist_q_answer"],
                label=llm, color=color_map[llm], alpha=0.6, s=30, edgecolors="white", linewidths=0.3,
            )

        # Diagonal reference line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "--", color="gray", alpha=0.5, linewidth=1, label="Equal distance")

        ax.set_xlabel("Vector Distance (Question → Ground Truth)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Vector Distance (Question → Answer)", fontsize=11, fontweight="bold")
        ax.set_title("Semantic Distance: Answer vs Ground Truth", fontsize=13, fontweight="bold", pad=10)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.2, linestyle="--")

        # Tight axes around data range
        xvals = subset["vec_dist_q_gt"].dropna()
        yvals = subset["vec_dist_q_answer"].dropna()
        all_vals = pd.concat([xvals, yvals])
        margin = (all_vals.max() - all_vals.min()) * 0.08
        ax.set_xlim(all_vals.min() - margin, all_vals.max() + margin)
        ax.set_ylim(all_vals.min() - margin, all_vals.max() + margin)

        fig.tight_layout()
        path = output_dir / "vector_distance_scatter.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)
        return path


def plot_chunking_strategy_comparison(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Grouped bar chart comparing chunking strategies across key metrics."""
    if "chunking_strategy" not in df.columns:
        return None

    strat_metrics = [
        ("ragas_faithfulness", "Faithfulness"),
        ("ragas_answer_relevancy", "Ans. Relevancy"),
        ("custom_hit_at_1", "Hit@1"),
        ("custom_ndcg_at_1", "NDCG@1"),
        ("ragas_context_recall", "Ctx. Recall"),
    ]
    available = [(c, l) for c, l in strat_metrics if c in df.columns and df[c].notna().any()]
    if not available:
        return None

    cols = [c for c, _ in available]
    labels = [l for _, l in available]

    strat_means = df.groupby("chunking_strategy")[cols].mean()
    if len(strat_means) < 2:
        return None

    strategies = strat_means.index.tolist()
    n_metrics = len(cols)
    n_strats = len(strategies)
    x = np.arange(n_metrics)
    width = 0.8 / n_strats

    palette = sns.color_palette("deep", n_colors=n_strats)

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(max(6, n_metrics * 1.5), 5))

        for i, strat in enumerate(strategies):
            vals = strat_means.loc[strat].values
            offset = (i - n_strats / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=strat, color=palette[i], edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Mean Score", fontsize=11, fontweight="bold")
        ax.set_title("Chunking Strategy Comparison", fontsize=13, fontweight="bold", pad=10)
        ax.set_ylim(0, min(1.0, strat_means.max().max() * 1.2))
        ax.legend(title="Strategy", fontsize=9, title_fontsize=9)
        ax.grid(axis="y", alpha=0.2, linestyle="--")

        fig.tight_layout()
        path = output_dir / "chunking_strategy_comparison.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)
        return path


def plot_stage_timing_breakdown(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Stacked bar chart showing time breakdown per pipeline stage."""
    time_cols = [c for c in df.columns if c.startswith("time_") and c != "time_total"]
    if not time_cols:
        return None

    # Filter to rows with at least one timing
    subset = df.dropna(subset=time_cols, how="all")
    if len(subset) < 10:
        logger.info("Stage timing: only %d entries, need >= 10. Skipping.", len(subset))
        return None

    # Use run_name + config as label
    subset = subset.copy()
    subset["label"] = subset.apply(_short_config_label, axis=1)

    # Take top 20 by total time
    subset["total_stage_time"] = subset[time_cols].sum(axis=1)
    subset = subset.nlargest(20, "total_stage_time").sort_values("total_stage_time", ascending=True)

    stage_labels = [c.replace("time_", "").replace("_", " ").title() for c in time_cols]
    palette = sns.color_palette("deep", n_colors=len(time_cols))

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(9, max(5, len(subset) * 0.4)))

        left = np.zeros(len(subset))
        for i, col in enumerate(time_cols):
            vals = subset[col].fillna(0).values
            ax.barh(range(len(subset)), vals, left=left, label=stage_labels[i],
                    color=palette[i], edgecolor="white", linewidth=0.3, height=0.7)
            left += vals

        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels(subset["label"], fontsize=8)
        ax.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
        n_questions = int(subset["num_questions"].mean()) if "num_questions" in subset.columns else "?"
        ax.set_title(f"Pipeline Stage Timing Breakdown (Top 20 Slowest, N={n_questions} questions)", fontsize=12, fontweight="bold", pad=10)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, framealpha=0.9)
        ax.grid(axis="x", alpha=0.2, linestyle="--")

        fig.tight_layout()
        path = output_dir / "stage_timing_breakdown.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)
        return path


def _parse_eval_matrix(path: Path) -> pd.DataFrame:
    """Parse EVAL_MATRIX.md into a DataFrame. Merges main + math-metrics tables."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    main_start = math_start = None
    for i, line in enumerate(lines):
        if "## Matrix" in line:
            main_start = i
        if "## Mathematische Metriken" in line:
            math_start = i

    if main_start is None:
        return pd.DataFrame()

    def _parse_pipe_table(start: int, end: int) -> pd.DataFrame:
        rows = []
        headers = None
        for i in range(start, min(end, len(lines)) if end else len(lines)):
            line = lines[i].strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if all(set(c) <= {"-", " ", ":"} for c in cells):
                continue
            if headers is None:
                headers = cells
                continue
            rows.append(cells)
        if not headers or not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=headers)

    main_end = math_start if math_start else len(lines)
    main_df = _parse_pipe_table(main_start, main_end)
    if main_df.empty:
        return pd.DataFrame()

    if "#" in main_df.columns:
        main_df = main_df.rename(columns={"#": "config_id"})
    main_df["config_id"] = main_df["config_id"].astype(str).str.strip()
    if "Status" in main_df.columns:
        main_df = main_df[main_df["Status"].str.strip() == "Getestet"]
    for col in ["Faith", "Size", "Overlap", "N"]:
        if col in main_df.columns:
            main_df[col] = pd.to_numeric(main_df[col].astype(str).str.strip(), errors="coerce")
    for col in ["Faith", "Rel", "Corr", "Prec", "Rec"]:
        if col in main_df.columns:
            main_df[col] = main_df[col].replace("-", np.nan)
            main_df[col] = pd.to_numeric(main_df[col], errors="coerce")

    if math_start is not None:
        math_df = _parse_pipe_table(math_start, len(lines))
        if not math_df.empty and "ID" in math_df.columns:
            math_df = math_df.rename(columns={"ID": "config_id"})
            math_df["config_id"] = math_df["config_id"].astype(str).str.strip()
            for col in math_df.columns:
                if col == "config_id":
                    continue
                math_df[col] = pd.to_numeric(math_df[col].astype(str).str.strip(), errors="coerce")
            main_df = main_df.merge(math_df, on="config_id", how="left", suffixes=("", "_math"))

    return main_df


def plot_llm_run_comparison(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Generate one large PNG per LLM with subplots for quality metrics.

    Reads curated results from EVAL_MATRIX.md. No performance metrics.
    Single color per LLM, x-axis = config number.
    """
    eval_path = Path("EVAL_MATRIX.md")
    if not eval_path.exists():
        logger.warning("EVAL_MATRIX.md not found, skipping per-LLM comparison")
        return []

    em = _parse_eval_matrix(eval_path)
    if em.empty:
        logger.warning("EVAL_MATRIX.md produced no data, skipping")
        return []

    em["llm_display"] = em["LLM"].str.strip()

    quality_metrics = [
        ("Faith", "Faithfulness"),
        ("ndcg@5", "NDCG@5"),
        ("hit@1", "Hit@1"),
        ("recall@5", "Recall@5"),
        ("context_relevance", "Context Relevance"),
        ("bert_score_f1", "BERTScore F1"),
        ("meteor", "METEOR"),
        ("rouge_l", "ROUGE-L"),
    ]
    available = [(c, l) for c, l in quality_metrics if c in em.columns and em[c].notna().any()]
    if not available:
        return []

    llm_models = sorted(em["llm_display"].dropna().unique())
    generated: list[Path] = []
    bar_color = "#4C78A8"

    for llm in llm_models:
        llm_df = em[em["llm_display"] == llm].copy()
        if llm_df.empty:
            continue

        n_metrics = len(available)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        with plt.rc_context({**_paper_rc(), "font.size": 10, "axes.titlesize": 11, "xtick.labelsize": 7, "ytick.labelsize": 9}):
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows))
            if n_metrics == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            for idx, (col, label) in enumerate(available):
                ax = axes[idx]
                subset = llm_df.dropna(subset=[col]).sort_values("config_id")
                if subset.empty:
                    ax.set_visible(False)
                    continue

                x = np.arange(len(subset))
                vals = subset[col].values.astype(float)

                ax.bar(x, vals, color=bar_color, edgecolor="white", linewidth=0.4, width=0.7, alpha=0.85)

                # Value labels only when few enough bars to avoid overlap
                if len(x) <= 16:
                    for xi, v in zip(x, vals):
                        ax.text(xi, v + (vals.max() - vals.min()) * 0.03, f"{v:.3f}",
                                ha="center", va="bottom", fontsize=6, fontweight="bold")

                labels = subset["config_id"].values
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
                # Thin out labels when many configs
                if len(x) > 15:
                    for i, tick in enumerate(ax.get_xticklabels()):
                        if i % 3 != 0:
                            tick.set_visible(False)
                ax.set_title(label, fontsize=11, fontweight="bold")
                ax.set_ylabel("")
                ax.grid(axis="y", alpha=0.2, linestyle="--")

                vmin, vmax = vals.min(), vals.max()
                vrange = vmax - vmin if vmax > vmin else 0.1
                ax.set_ylim(max(0, vmin - vrange * 0.1), vmax + vrange * 0.15)
                ax.axhline(vals.mean(), color="#999999", linewidth=0.8, linestyle=":", alpha=0.7)

            for idx in range(len(available), len(axes)):
                axes[idx].set_visible(False)

            fig.suptitle(f"{llm} — Config Scores (EVAL_MATRIX, N≥100)", fontsize=13, fontweight="bold", y=1.01)
            fig.tight_layout()

            safe_name = llm.replace(":", "_").replace("/", "_").replace(" ", "_")
            path = output_dir / f"llm_runs_{safe_name}.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved %s", path)
            generated.append(path)

    return generated


def _stats_t_ci(std: float, n: float) -> float:
    """Compute CI half-width from std and count."""
    from scipy import stats as _stats
    if n < 2 or std == 0:
        return 0.0
    t_crit = _stats.t.ppf(0.975, df=n - 1)
    return t_crit * std / np.sqrt(n)


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

    # Retrieval-level comparison (consolidated k=1/3/5)
    retrieval = plot_retrieval_level_comparison(df, output_dir)
    if retrieval:
        generated.append(retrieval)

    # Per-metric plots: Extra NLG metrics (BLEU, BERTScore P/R)
    for col, label in NLG_EXTRA_METRICS:
        cov = _metric_coverage(col)
        if cov == 0:
            continue
        suffix = f" ({cov}/{n_runs} runs)" if cov < n_runs else ""
        path = plot_metric_over_runs(df, col, label, output_dir, title_suffix=suffix)
        if path:
            generated.append(path)

    # Paper/poster-ready summary figures (main evidence figures)
    for paper_plot in (
        plot_paper_quality_ranking,
        plot_paper_quality_load_proxy,
        plot_paper_faithfulness_by_llm_prompt,
        plot_paper_llm_heatmap,
        plot_paper_retrieval_by_strategy,
        plot_paper_timing_subset,
    ):
        path = paper_plot(df, output_dir)
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

    # Correlation heatmap
    heatmap = plot_correlation_heatmap(df, output_dir)
    if heatmap:
        generated.append(heatmap)

    # Radar chart per LLM
    radar = plot_radar_per_llm(df, output_dir)
    if radar:
        generated.append(radar)

    # Vector distance scatter
    scatter = plot_vector_distance_scatter(df, output_dir)
    if scatter:
        generated.append(scatter)

    # Chunking strategy comparison
    chunking = plot_chunking_strategy_comparison(df, output_dir)
    if chunking:
        generated.append(chunking)

    # Stage timing breakdown
    timing = plot_stage_timing_breakdown(df, output_dir)
    if timing:
        generated.append(timing)

    # Per-LLM run comparison plots (one large PNG per LLM)
    llm_run_plots = plot_llm_run_comparison(df, output_dir)
    generated.extend(llm_run_plots)

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
