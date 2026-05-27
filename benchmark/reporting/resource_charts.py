"""Resource-utilization paper figures from benchmark trace CSVs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scenario:
    label: str
    trace_path: Path
    markers_path: Path | None = None


def _paper_rc() -> dict:
    return {
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }


def discover_scenarios(trace_dir: Path) -> list[Scenario]:
    traces = sorted(
        p for p in trace_dir.glob("*.csv")
        if not p.name.endswith("_markers.csv")
    )
    scenarios = []
    for trace in traces:
        markers = trace.with_name(f"{trace.stem}_markers.csv")
        scenarios.append(
            Scenario(
                label=trace.stem,
                trace_path=trace,
                markers_path=markers if markers.exists() else None,
            )
        )
    return scenarios


def parse_scenario(value: str) -> Scenario:
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            "scenario must be LABEL:TRACE_CSV or LABEL:TRACE_CSV:MARKERS_CSV"
        )
    label, trace = parts[:2]
    markers = parts[2] if len(parts) == 3 else None
    return Scenario(
        label=label,
        trace_path=Path(trace),
        markers_path=Path(markers) if markers else None,
    )


def plot_resource_breakdown(
    scenarios: list[Scenario],
    output_stem: Path,
    *,
    max_scenarios: int | None = None,
) -> Path:
    if max_scenarios is not None:
        scenarios = scenarios[:max_scenarios]
    if not scenarios:
        raise ValueError("No resource trace CSVs provided")

    panels = [
        ("cpu", "CPU Utilization (%)"),
        ("gpu_util", "GPU Utilization (%)"),
        ("host_mem", "Host Memory (GB)"),
        ("gpu_mem", "GPU Memory (GB)"),
        ("pcie", "PCIe Throughput (MB/s)"),
        ("disk", "Disk I/O Bandwidth (GB/s)"),
        ("gpu_dram", "GPU DRAM Bandwidth\nUtilization (%)"),
        ("power", "GPU Power (W)"),
    ]

    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(
            len(scenarios),
            len(panels),
            figsize=(16, 2.15 * len(scenarios)),
            squeeze=False,
        )

        for row_idx, scenario in enumerate(scenarios):
            df = _read_trace(scenario.trace_path)
            markers = _read_markers(scenario.markers_path)
            x = _time_axis(df)
            max_x = float(np.nanmax(x)) if x.size else 0.0
            unit = "min" if max_x >= 180 else "sec"
            if unit == "min":
                x = x / 60.0
                if not markers.empty:
                    markers["elapsed_s"] = markers["elapsed_s"] / 60.0

            for col_idx, (kind, ylabel) in enumerate(panels):
                ax = axes[row_idx, col_idx]
                _plot_panel(ax, df, x, kind)
                _decorate_panel(ax, markers, ylabel, unit, show_xlabel=row_idx == len(scenarios) - 1)
                if col_idx == 0:
                    ax.text(
                        -0.42,
                        0.5,
                        scenario.label,
                        transform=ax.transAxes,
                        ha="right",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        rotation=90,
                    )

        handles, labels = _combined_legend(axes)
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=min(8, len(labels)),
                frameon=False,
                bbox_to_anchor=(0.5, 1.02),
            )
        fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.96))

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(output_stem.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)
    return output_stem.with_suffix(".pdf")


def _read_trace(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "elapsed_s" not in df.columns:
        raise ValueError(f"{path} does not contain an elapsed_s column")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("elapsed_s")


def _read_markers(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["stage", "event", "elapsed_s"])
    df = pd.read_csv(path)
    if not {"stage", "event", "elapsed_s"}.issubset(df.columns):
        return pd.DataFrame(columns=["stage", "event", "elapsed_s"])
    df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")
    return df.dropna(subset=["elapsed_s"])


def _time_axis(df: pd.DataFrame) -> np.ndarray:
    return df["elapsed_s"].to_numpy(dtype=float)


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    return df[col]


def _plot_panel(ax: plt.Axes, df: pd.DataFrame, x: np.ndarray, kind: str) -> None:
    if kind == "cpu":
        user = _series(df, "cpu_user_pct").fillna(0)
        system = _series(df, "cpu_system_pct").fillna(0)
        ax.stackplot(x, user, system, labels=["User", "System"], colors=["#56B4E9", "#E69F00"], alpha=0.9)
        ax.set_ylim(0, 100)
    elif kind == "gpu_util":
        ax.plot(x, _series(df, "gpu_util_pct"), label="SM Utilization", color="#56B4E9", lw=1.6)
        ax.plot(x, _series(df, "gpu_mem_util_pct"), label="SM Occupancy", color="#009E73", lw=1.6)
        ax.set_ylim(0, 100)
    elif kind == "host_mem":
        ax.plot(x, _series(df, "host_mem_used_gb"), label="VectorDB", color="#4B0082", lw=1.8)
    elif kind == "gpu_mem":
        ax.plot(x, _series(df, "gpu_mem_used_gb"), label="Generation Model", color="#008000", lw=1.8)
        total = _series(df, "gpu_mem_total_gb")
        if total.notna().any():
            ax.axhline(total.dropna().max(), color="black", lw=1.0, ls="--")
    elif kind == "pcie":
        ax.plot(x, _series(df, "pcie_rx_mb_s"), label="PCIe to CPU", color="#56B4E9", lw=1.6)
        ax.plot(x, _series(df, "pcie_tx_mb_s"), label="PCIe to GPU", color="#009E73", lw=1.6)
    elif kind == "disk":
        ax.plot(x, _series(df, "disk_read_gb_s"), label="Read", color="#56B4E9", lw=1.6)
        ax.plot(x, _series(df, "disk_write_gb_s"), label="Write", color="#009E73", lw=1.6)
    elif kind == "gpu_dram":
        ax.plot(x, _series(df, "gpu_mem_util_pct"), label="Average", color="#009E73", lw=1.6)
        max_line = _series(df, "gpu_mem_util_pct").expanding().max()
        ax.plot(x, max_line, label="Max", color="#56B4E9", lw=1.6)
        ax.set_ylim(0, 100)
    elif kind == "power":
        ax.plot(x, _series(df, "gpu_power_w"), label="Power", color="#D55E00", lw=1.6)


def _decorate_panel(
    ax: plt.Axes,
    markers: pd.DataFrame,
    ylabel: str,
    unit: str,
    *,
    show_xlabel: bool,
) -> None:
    starts = markers[markers["event"] == "start"] if not markers.empty else markers
    ymax = ax.get_ylim()[1]
    for _, row in starts.iterrows():
        elapsed = float(row["elapsed_s"])
        stage = str(row["stage"])
        ax.axvline(elapsed, color="red", ls="--", lw=1.0, alpha=0.85)
        if stage in {"chunk", "index", "retrieve", "rerank", "generate", "external_rag"}:
            label = "insert" if stage == "index" else stage.replace("_", " ")
            ax.annotate(
                label,
                xy=(elapsed, ymax),
                xytext=(2, 8),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=7,
                arrowprops={"arrowstyle": "-|>", "lw": 0.6, "color": "black"},
            )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(f"Time ({unit})" if show_xlabel else "")


def _combined_legend(axes: np.ndarray) -> tuple[list, list[str]]:
    handles = []
    labels = []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    return handles, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate resource-utilization paper charts")
    parser.add_argument("--trace-dir", type=Path, help="Directory containing trace CSV files")
    parser.add_argument(
        "--scenario",
        action="append",
        type=parse_scenario,
        help="LABEL:TRACE_CSV or LABEL:TRACE_CSV:MARKERS_CSV. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Paper/NGEN-AI/figures/fig_resource_breakdown"),
        help="Output path stem. .pdf and .png are written.",
    )
    parser.add_argument("--max-scenarios", type=int, default=None)
    args = parser.parse_args()

    scenarios = args.scenario or []
    if args.trace_dir:
        scenarios.extend(discover_scenarios(args.trace_dir))
    path = plot_resource_breakdown(scenarios, args.output, max_scenarios=args.max_scenarios)
    print(path)


if __name__ == "__main__":
    main()
