"""Lightweight resource tracing for benchmark runs.

The monitor samples host and GPU counters into CSV files so paper figures can
show time-series resource utilization alongside benchmark stage boundaries.
It intentionally uses stdlib plus ``nvidia-smi`` instead of adding a runtime
dependency such as psutil.
"""
from __future__ import annotations

import csv
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


_SAMPLE_FIELDS = [
    "elapsed_s",
    "wall_time_s",
    "cpu_user_pct",
    "cpu_system_pct",
    "host_mem_used_gb",
    "host_mem_pct",
    "gpu_util_pct",
    "gpu_mem_util_pct",
    "gpu_mem_used_gb",
    "gpu_mem_total_gb",
    "gpu_power_w",
    "pcie_rx_mb_s",
    "pcie_tx_mb_s",
    "disk_read_gb_s",
    "disk_write_gb_s",
]


@dataclass(frozen=True)
class _CpuTimes:
    user: int
    nice: int
    system: int
    idle: int
    iowait: int
    irq: int
    softirq: int
    steal: int

    @property
    def total(self) -> int:
        return (
            self.user
            + self.nice
            + self.system
            + self.idle
            + self.iowait
            + self.irq
            + self.softirq
            + self.steal
        )


@dataclass(frozen=True)
class _DiskCounters:
    read_sectors: int
    write_sectors: int


class ResourceMonitor:
    """Sample process-host resource counters into CSV files."""

    def __init__(
        self,
        trace_path: Path,
        markers_path: Path,
        *,
        interval_seconds: float = 1.0,
        gpu_index: int = 0,
    ) -> None:
        self.trace_path = trace_path
        self.markers_path = markers_path
        self.interval_seconds = max(interval_seconds, 0.1)
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_wall = 0.0
        self._started_perf = 0.0
        self._marker_lock = threading.Lock()
        self._stage_starts: dict[str, float] = {}
        self._prev_cpu: _CpuTimes | None = None
        self._prev_disk: tuple[float, _DiskCounters] | None = None

    def __enter__(self) -> "ResourceMonitor":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    @property
    def elapsed(self) -> float:
        if self._started_perf <= 0:
            return 0.0
        return time.perf_counter() - self._started_perf

    def start(self) -> None:
        if self._thread is not None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.markers_path.parent.mkdir(parents=True, exist_ok=True)
        self._started_wall = time.time()
        self._started_perf = time.perf_counter()
        self._prev_cpu = _read_cpu_times()
        self._prev_disk = (self.elapsed, _read_disk_counters())
        self._write_trace_header()
        self._write_marker_header()
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        self._thread = None

    def stage_start(self, stage: str) -> None:
        if self._thread is None:
            return
        with self._marker_lock:
            self._stage_starts[stage] = self.elapsed
            self._append_marker(stage, "start", self.elapsed)

    def stage_end(self, stage: str) -> None:
        if self._thread is None:
            return
        with self._marker_lock:
            elapsed = self.elapsed
            self._append_marker(stage, "end", elapsed)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_seconds)
        self._sample_once()

    def _write_trace_header(self) -> None:
        with self.trace_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_SAMPLE_FIELDS).writeheader()

    def _write_marker_header(self) -> None:
        with self.markers_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["stage", "event", "elapsed_s"])
            writer.writeheader()

    def _append_marker(self, stage: str, event: str, elapsed: float) -> None:
        with self.markers_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["stage", "event", "elapsed_s"])
            writer.writerow({
                "stage": stage,
                "event": event,
                "elapsed_s": f"{elapsed:.6f}",
            })

    def _sample_once(self) -> None:
        elapsed = self.elapsed
        row = {
            "elapsed_s": elapsed,
            "wall_time_s": self._started_wall + elapsed,
        }
        row.update(_cpu_percentages(self._prev_cpu))
        self._prev_cpu = _read_cpu_times()
        row.update(_host_memory())
        row.update(_gpu_metrics(self.gpu_index))
        row.update(self._disk_rates(elapsed))

        with self.trace_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_SAMPLE_FIELDS)
            writer.writerow({
                key: "" if row.get(key) is None else f"{float(row[key]):.6f}"
                for key in _SAMPLE_FIELDS
            })

    def _disk_rates(self, elapsed: float) -> dict[str, float | None]:
        current = _read_disk_counters()
        if self._prev_disk is None:
            self._prev_disk = (elapsed, current)
            return {"disk_read_gb_s": None, "disk_write_gb_s": None}

        prev_elapsed, prev = self._prev_disk
        delta_s = max(elapsed - prev_elapsed, 1e-9)
        # Linux sectors are conventionally 512 bytes in /proc/diskstats.
        read_gb = max(current.read_sectors - prev.read_sectors, 0) * 512 / 1e9
        write_gb = max(current.write_sectors - prev.write_sectors, 0) * 512 / 1e9
        self._prev_disk = (elapsed, current)
        return {
            "disk_read_gb_s": read_gb / delta_s,
            "disk_write_gb_s": write_gb / delta_s,
        }


def enabled_from_env() -> bool:
    raw = os.getenv("BENCHMARK_RESOURCE_MONITOR", "false").strip().lower()
    return raw in ("1", "true", "yes", "on")


def interval_from_env() -> float:
    return float(os.getenv("BENCHMARK_RESOURCE_MONITOR_INTERVAL_SECONDS", "1.0"))


def gpu_index_from_env() -> int:
    return int(os.getenv("BENCHMARK_RESOURCE_MONITOR_GPU_INDEX", "0"))


def _read_cpu_times() -> _CpuTimes | None:
    try:
        line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
        parts = [int(v) for v in line.split()[1:9]]
        while len(parts) < 8:
            parts.append(0)
        return _CpuTimes(*parts[:8])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def _cpu_percentages(prev: _CpuTimes | None) -> dict[str, float | None]:
    current = _read_cpu_times()
    if prev is None or current is None:
        return {"cpu_user_pct": None, "cpu_system_pct": None}
    total_delta = current.total - prev.total
    if total_delta <= 0:
        return {"cpu_user_pct": None, "cpu_system_pct": None}
    user_delta = (current.user + current.nice) - (prev.user + prev.nice)
    system_delta = (
        current.system
        + current.irq
        + current.softirq
        - prev.system
        - prev.irq
        - prev.softirq
    )
    return {
        "cpu_user_pct": max(user_delta, 0) * 100.0 / total_delta,
        "cpu_system_pct": max(system_delta, 0) * 100.0 / total_delta,
    }


def _host_memory() -> dict[str, float | None]:
    try:
        values: dict[str, float] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            values[key] = float(raw.strip().split()[0])
        total_kb = values.get("MemTotal")
        available_kb = values.get("MemAvailable")
        if not total_kb or available_kb is None:
            return {"host_mem_used_gb": None, "host_mem_pct": None}
        used_kb = total_kb - available_kb
        return {
            "host_mem_used_gb": used_kb / 1024 / 1024,
            "host_mem_pct": used_kb * 100.0 / total_kb,
        }
    except (FileNotFoundError, ValueError, IndexError):
        return {"host_mem_used_gb": None, "host_mem_pct": None}


def _gpu_metrics(gpu_index: int) -> dict[str, float | None]:
    keys = [
        "gpu_util_pct",
        "gpu_mem_util_pct",
        "gpu_mem_used_gb",
        "gpu_mem_total_gb",
        "gpu_power_w",
        "pcie_rx_mb_s",
        "pcie_tx_mb_s",
    ]
    query = (
        "utilization.gpu,utilization.memory,memory.used,memory.total,"
        "power.draw,pcie.rx_throughput,pcie.tx_throughput"
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return dict.fromkeys(keys, None)

    if result.returncode != 0 or not result.stdout.strip():
        return dict.fromkeys(keys, None)

    parts = [p.strip() for p in result.stdout.strip().splitlines()[0].split(",")]
    values: list[float | None] = []
    for raw in parts[:7]:
        try:
            values.append(float(raw))
        except ValueError:
            values.append(None)
    while len(values) < 7:
        values.append(None)

    gpu_util, gpu_mem_util, mem_used_mb, mem_total_mb, power, pcie_rx_kb, pcie_tx_kb = values
    return {
        "gpu_util_pct": gpu_util,
        "gpu_mem_util_pct": gpu_mem_util,
        "gpu_mem_used_gb": None if mem_used_mb is None else mem_used_mb / 1024,
        "gpu_mem_total_gb": None if mem_total_mb is None else mem_total_mb / 1024,
        "gpu_power_w": power,
        "pcie_rx_mb_s": None if pcie_rx_kb is None else pcie_rx_kb / 1024,
        "pcie_tx_mb_s": None if pcie_tx_kb is None else pcie_tx_kb / 1024,
    }


def _read_disk_counters() -> _DiskCounters:
    read_sectors = 0
    write_sectors = 0
    try:
        for line in Path("/proc/diskstats").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 14:
                continue
            name = parts[2]
            if name.startswith(("loop", "ram", "sr")):
                continue
            read_sectors += int(parts[5])
            write_sectors += int(parts[9])
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return _DiskCounters(read_sectors=read_sectors, write_sectors=write_sectors)
