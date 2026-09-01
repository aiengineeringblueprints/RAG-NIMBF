"""Plotly chart builders for the dashboard (no Streamlit dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.loader import metric_label, per_sample_metric_values

_CATEGORY_COLORS = px.colors.qualitative.Plotly


def metric_comparison_bar(
    df: pd.DataFrame, metric: str, sort: bool = True
) -> go.Figure:
    """Horizontal bar of one metric across configs."""
    plot_df = df[["config_name", metric]].dropna()
    if plot_df.empty:
        return _empty_figure(f"Keine Daten für {metric_label(metric)}")
    plot_df = plot_df.rename(columns={metric: "value"})
    if sort:
        plot_df = plot_df.sort_values("value", ascending=True)
    fig = px.bar(
        plot_df,
        x="value",
        y="config_name",
        orientation="h",
        title=metric_label(metric),
        labels={"value": metric_label(metric), "config_name": "Config"},
        color="value",
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=max(320, 40 * max(len(plot_df), 1)), margin=dict(l=10, r=10, t=50, b=10))
    return fig


def per_sample_box(
    df: pd.DataFrame, metric: str, configs: list[dict[str, Any]]
) -> go.Figure:
    """Box plot of a metric's per-sample distribution per config."""
    fig = go.Figure()
    for color, config in zip(_CATEGORY_COLORS, configs):
        values = per_sample_metric_values(config, metric)
        if not values:
            continue
        fig.add_trace(
            go.Box(
                y=values,
                name=config.get("config_name", "?"),
                boxmean=True,
                marker_color=color,
            )
        )
    fig.update_layout(
        title=metric_label(metric),
        yaxis_title=metric_label(metric),
        height=max(320, 60 * max(len(fig.data), 1)),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    if not fig.data:
        return _empty_figure(f"Keine Per-Sample-Daten für {metric_label(metric)}")
    return fig


def scatter_quality_vs_perf(
    df: pd.DataFrame, quality_metric: str, perf_metric: str
) -> go.Figure:
    """Scatter quality vs. performance, labeled by config."""
    plot_df = df.dropna(subset=[quality_metric, perf_metric])
    if plot_df.empty:
        return _empty_figure(
            f"Keine Daten für {metric_label(quality_metric)} vs. "
            f"{metric_label(perf_metric)} in dieser Auswahl"
        )
    fig = px.scatter(
        plot_df,
        x=perf_metric,
        y=quality_metric,
        text="config_name",
        title=f"{metric_label(quality_metric)} vs. {metric_label(perf_metric)}",
        labels={
            perf_metric: metric_label(perf_metric),
            quality_metric: metric_label(quality_metric),
            "config_name": "Config",
        },
        color="config_name",
    )
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def stage_timing_bar(stage_latency: dict[str, Any]) -> go.Figure:
    """Per-stage mean latency bar chart from a stage_latency dict."""
    rows = []
    for stage, stats in (stage_latency or {}).items():
        if not isinstance(stats, dict):
            continue
        mean = stats.get("mean_s")
        count = stats.get("count")
        if mean is None:
            continue
        rows.append(
            {
                "stage": stage,
                "mean_s": float(mean),
                "p95_s": float(stats.get("p95_s") or 0),
                "count": int(count) if count is not None else 0,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return _empty_figure("Keine Stage-Timings vorhanden")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["stage"], y=df["mean_s"], name="Mean (s)", marker_color="#4C78A8"
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["stage"], y=df["p95_s"], name="P95 (s)", marker_color="#F58518"
        )
    )
    fig.update_layout(
        barmode="group",
        title="Stage-Latenzen",
        yaxis_title="Sekunden",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def resource_trace_plot(csv_path: Path) -> go.Figure:
    """Line chart of GPU power / utilization over a resource trace CSV."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return _empty_figure(f"Trace konnte nicht gelesen werden: {csv_path.name}")
    time_col = None
    for candidate in ("timestamp", "time", "elapsed_s"):
        if candidate in df.columns:
            time_col = candidate
            break
    x = df[time_col] if time_col else df.index
    fig = go.Figure()
    for col in ("gpu_power_w", "gpu_utilization_pct", "gpu_memory_used_mb", "cpu_utilization_pct"):
        if col in df.columns:
            fig.add_trace(go.Scatter(x=x, y=df[col], name=col, mode="lines"))
    fig.update_layout(
        title="Ressourcen-Trace",
        yaxis_title="Wert",
        xaxis_title=time_col or "Sample",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def llm_performance_bar(llm_perf: dict[str, Any]) -> go.Figure:
    """Selected LLM load-test latency metrics as a grouped bar chart."""
    selectable = {
        "latency_mean_s": "Mean (s)",
        "latency_median_s": "Median (s)",
        "latency_p95_s": "P95 (s)",
        "ttft_mean_s": "TTFT Mean (s)",
        "ttft_p95_s": "TTFT P95 (s)",
    }
    rows = []
    for key, label in selectable.items():
        matches = {
            k.split("_", 2)[1]: v
            for k, v in (llm_perf or {}).items()
            if k.endswith(key)
        }
        for mode, value in matches.items():
            rows.append({"mode": mode, "metric": label, "value": float(value)})
    df = pd.DataFrame(rows)
    if df.empty:
        return _empty_figure("Keine LLM-Performance-Metriken vorhanden")
    fig = px.bar(
        df,
        x="mode",
        y="value",
        color="metric",
        barmode="group",
        title="LLM-Performance",
        labels={"mode": "Profil", "value": "Sekunden", "metric": "Metrik"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def metric_correlation_heatmap(df: pd.DataFrame, metrics: list[str]) -> go.Figure:
    """Correlation heatmap of the selected metrics across configs."""
    corr = df[metrics].dropna(axis=1).corr()
    if corr.empty:
        return _empty_figure("Nicht genug Daten für Korrelation")
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=[metric_label(c) for c in corr.columns],
            y=[metric_label(c) for c in corr.columns],
            zmin=-1,
            zmax=1,
            colorscale="RdBu_r",
            colorbar=dict(title="r"),
        )
    )
    fig.update_layout(
        title="Metrik-Korrelation über Configs",
        height=600,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
    return fig