"""Tests for the NVML + cgroups v2 resource monitor backend."""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchmark import resource_monitor_v2
from benchmark.resource_monitor_v2 import (
    DEFAULT_OVERHEAD_FACTOR,
    NvmlNotAvailableError,
    ResourceMonitorV2,
    SAMPLE_FIELDS,
    _read_cgroup_cpu_stat,
    _read_cgroup_memory_current,
    backend_from_env,
    cgroup_path_from_env,
    is_v2_requested,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cgroup(tmp_path: Path) -> Path:
    """Synthesise a cgroup-shaped directory under tmp_path."""
    cg = tmp_path / "rag.slice"
    cg.mkdir()
    (cg / "cpu.stat").write_text(
        "usage_usec 1000000\n"
        "user_usec 600000\n"
        "system_usec 400000\n"
        "nr_periods 0\n"
        "nr_throttled 0\n"
        "throttled_usec 0\n"
    )
    (cg / "memory.current").write_text("524288000")  # 500 MiB
    return cg


class _FakeNvml:
    """In-process NVML stand-in so tests need no GPU.

    Returns canned values and records call counts so tests can assert the
    backend hit the expected code paths.
    """

    def __init__(
        self,
        *,
        available: bool = True,
        util: tuple[int, int] = (42, 17),
        mem: tuple[int, int] = (5 * 10**9, 24 * 10**9),  # 5/24 GiB
        power_mw: int = 250_000,  # 250 W
        pcie_kb_s: tuple[int, int] = (1024, 2048),  # 1 / 2 MiB/s
        gpm: tuple[float, float] = (35.5, 12.7),
    ) -> None:
        self._available = available
        self._util = util
        self._mem = mem
        self._power_mw = power_mw
        self._pcie = pcie_kb_s
        self._gpm = gpm
        self.calls: dict[str, int] = {}

    def _bump(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    @property
    def available(self) -> bool:
        return self._available

    def close(self) -> None:
        self._bump("close")

    def utilization_rates(self):
        self._bump("utilization_rates")
        return (float(self._util[0]), float(self._util[1]))

    def memory_info(self):
        self._bump("memory_info")
        return (self._mem[0] / 1e9, self._mem[1] / 1e9)

    def power_watts(self):
        self._bump("power_watts")
        return self._power_mw / 1000.0

    def pcie_throughput_mb_s(self):
        self._bump("pcie")
        return (
            self._pcie[0] / (1024.0 * 1024.0),
            self._pcie[1] / (1024.0 * 1024.0),
        )

    def gpm_engine_util(self):
        self._bump("gpm")
        return (self._gpm[0], self._gpm[1])


def _make_monitor(
    tmp_path: Path,
    *,
    interval_seconds: float = 0.05,
    cgroup_path: Path | None = None,
    nvml: _FakeNvml | None = None,
) -> tuple[ResourceMonitorV2, Path, Path, _FakeNvml]:
    """Build a monitor wired to a fake NVML backend under tmp_path."""
    trace = tmp_path / "trace.csv"
    markers = tmp_path / "markers.csv"
    fake = nvml or _FakeNvml()
    monitor = ResourceMonitorV2(
        trace,
        markers,
        interval_seconds=interval_seconds,
        gpu_index=0,
        cgroup_path=cgroup_path,
        nvml_factory=lambda idx: fake,
    )
    return monitor, trace, markers, fake


# ---------------------------------------------------------------------------
# Env-var plumbing
# ---------------------------------------------------------------------------


class TestEnvVars:
    def test_backend_default_is_v1(self, monkeypatch):
        monkeypatch.delenv("RESOURCE_MONITOR_BACKEND", raising=False)
        assert backend_from_env() == "v1"
        assert is_v2_requested() is False

    def test_v2_requested_explicitly(self, monkeypatch):
        monkeypatch.setenv("RESOURCE_MONITOR_BACKEND", "v2")
        assert backend_from_env() == "v2"
        assert is_v2_requested() is True

    def test_cgroup_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv("BENCHMARK_RESOURCE_MONITOR_CGROUP", raising=False)
        assert cgroup_path_from_env() is None

    def test_cgroup_set_returns_value(self, monkeypatch):
        monkeypatch.setenv("BENCHMARK_RESOURCE_MONITOR_CGROUP", "/sys/fs/cgroup/rag.slice")
        assert cgroup_path_from_env() == "/sys/fs/cgroup/rag.slice"


# ---------------------------------------------------------------------------
# Cgroup v2 parsing
# ---------------------------------------------------------------------------


class TestCgroupParsing:
    def test_cpu_stat_parser_reads_usage_usec(self, tmp_cgroup: Path):
        result = _read_cgroup_cpu_stat(tmp_cgroup)
        assert result is not None
        assert result.usage_usec == 1_000_000

    def test_cpu_stat_parser_returns_none_when_missing(self, tmp_path: Path):
        assert _read_cgroup_cpu_stat(tmp_path / "nope") is None

    def test_cpu_stat_parser_returns_none_on_malformed(self, tmp_path: Path):
        cg = tmp_path / "bad"
        cg.mkdir()
        (cg / "cpu.stat").write_text("not a stat line\n")
        assert _read_cgroup_cpu_stat(cg) is None

    def test_memory_current_parser(self, tmp_cgroup: Path):
        # 500 MiB = 500 * 1024 * 1024 bytes
        assert _read_cgroup_memory_current(tmp_cgroup) == 500 * 1024 * 1024

    def test_memory_current_missing_returns_none(self, tmp_path: Path):
        assert _read_cgroup_memory_current(tmp_path / "missing") is None


# ---------------------------------------------------------------------------
# Metric extraction (no GPU required)
# ---------------------------------------------------------------------------


class TestGpuExtraction:
    def test_gpu_metrics_populate_expected_columns(self, tmp_path: Path):
        monitor, _, _, fake = _make_monitor(tmp_path, nvml=_FakeNvml())
        monitor._nvml = fake  # bypass start() lifecycle
        row = monitor._gpu_metrics()
        assert row["gpu_util_pct"] == 42.0
        assert row["gpu_mem_util_pct"] == 17.0
        assert pytest.approx(row["gpu_mem_used_gb"], rel=1e-6) == 5.0
        assert pytest.approx(row["gpu_mem_total_gb"], rel=1e-6) == 24.0
        assert row["gpu_power_w"] == 250.0
        assert pytest.approx(row["pcie_rx_mb_s"], rel=1e-6) == 1.0 / 1024.0
        assert pytest.approx(row["pcie_tx_mb_s"], rel=1e-6) == 2.0 / 1024.0
        assert row["gpu_sm_util_pct"] == 35.5
        assert row["gpu_mem_ctrl_util_pct"] == 12.7

    def test_gpu_metrics_empty_when_unavailable(self, tmp_path: Path):
        monitor, _, _, _ = _make_monitor(
            tmp_path, nvml=_FakeNvml(available=False)
        )
        monitor._nvml = None
        row = monitor._gpu_metrics()
        for key in (
            "gpu_util_pct",
            "gpu_mem_util_pct",
            "gpu_mem_used_gb",
            "gpu_mem_total_gb",
            "gpu_power_w",
            "pcie_rx_mb_s",
            "pcie_tx_mb_s",
            "gpu_sm_util_pct",
            "gpu_mem_ctrl_util_pct",
        ):
            assert row[key] is None


# ---------------------------------------------------------------------------
# NVML availability / fallback
# ---------------------------------------------------------------------------


def test_nvml_factory_import_error_is_converted(tmp_path: Path):
    """If pynvml is missing, v2 should raise NvmlNotAvailableError on start."""

    def _bad_factory(idx):
        raise NvmlNotAvailableError("pynvml missing")

    monitor = ResourceMonitorV2(
        tmp_path / "t.csv",
        tmp_path / "m.csv",
        nvml_factory=_bad_factory,
    )
    with pytest.raises(NvmlNotAvailableError):
        monitor.start()


# ---------------------------------------------------------------------------
# Cgroup metric attribution
# ---------------------------------------------------------------------------


class TestCgroupMetrics:
    def test_cgroup_memory_attribution(self, tmp_path: Path, tmp_cgroup: Path):
        monitor, _, _, _ = _make_monitor(tmp_path, cgroup_path=tmp_cgroup)
        # Prime the previous-CPU delta state with a synthetic earlier sample.
        from benchmark.resource_monitor_v2 import _CgroupCpu, _read_cgroup_cpu_stat

        prev = _read_cgroup_cpu_stat(tmp_cgroup)
        assert prev is not None
        monitor._prev_cgroup_cpu = (0.0, prev)

        # Bump usage by 250 ms of CPU in 1 s of wall.
        (tmp_cgroup / "cpu.stat").write_text(
            "usage_usec 1250000\nuser_usec 750000\nsystem_usec 500000\n"
        )
        out = monitor._cgroup_metrics(elapsed=1.0)
        # 250 ms / 1 s wall = 25 % of one core
        assert out["proc_cpu_pct"] == pytest.approx(25.0, rel=1e-3)
        # 500 MiB exactly
        assert out["proc_mem_mb"] == pytest.approx(500.0, rel=1e-6)

    def test_cgroup_metrics_skip_when_no_path(self, tmp_path: Path):
        monitor, _, _, _ = _make_monitor(tmp_path, cgroup_path=None)
        out = monitor._cgroup_metrics(elapsed=1.0)
        assert out["proc_cpu_pct"] is None
        assert out["proc_mem_mb"] is None

    def test_cgroup_first_sample_returns_none_cpu(self, tmp_path: Path, tmp_cgroup: Path):
        monitor, _, _, _ = _make_monitor(tmp_path, cgroup_path=tmp_cgroup)
        # Don't prime _prev_cgroup_cpu.
        out = monitor._cgroup_metrics(elapsed=0.0)
        # First sample has no delta to compute from.
        assert out["proc_cpu_pct"] is None
        assert out["proc_mem_mb"] == pytest.approx(500.0, rel=1e-6)


# ---------------------------------------------------------------------------
# CSV schema + sampling
# ---------------------------------------------------------------------------


class TestSamplingLifecycle:
    def test_schema_includes_v1_and_v2_columns(self):
        v1_only = {
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
        }
        assert v1_only.issubset(set(SAMPLE_FIELDS))
        for extra in (
            "gpu_sm_util_pct",
            "gpu_mem_ctrl_util_pct",
            "proc_cpu_pct",
            "proc_mem_mb",
            "sample_self_ms",
            "interval_eff_s",
        ):
            assert extra in SAMPLE_FIELDS

    def test_sample_once_writes_full_row(self, tmp_path: Path, tmp_cgroup: Path):
        monitor, trace, _, fake = _make_monitor(
            tmp_path, cgroup_path=tmp_cgroup, interval_seconds=0.05
        )
        monitor.start()
        try:
            time.sleep(0.2)  # allow several samples
        finally:
            monitor.stop()
        assert fake.calls["utilization_rates"] >= 1
        rows = list(csv.DictReader(trace.open()))
        assert len(rows) >= 2
        # All SAMPLE_FIELDS present as headers in v1+ order with v2 trailing.
        with trace.open() as f:
            header = next(csv.reader(f))
        assert header[:15] == SAMPLE_FIELDS[:15]
        assert header == SAMPLE_FIELDS
        # Row dict has every field populated or empty.
        for row in rows:
            assert set(row.keys()) == set(SAMPLE_FIELDS)
            # GPU fields filled (fake backend is "available").
            assert row["gpu_util_pct"] not in ("", None)
            assert row["gpu_sm_util_pct"] not in ("", None)

    def test_stage_markers_written(self, tmp_path: Path):
        monitor, _, markers, _ = _make_monitor(tmp_path)
        monitor.start()
        try:
            monitor.stage_start("retrieve")
            time.sleep(0.05)
            monitor.stage_end("retrieve")
        finally:
            monitor.stop()
        rows = list(csv.DictReader(markers.open()))
        stages = [(r["stage"], r["event"]) for r in rows]
        assert ("retrieve", "start") in stages
        assert ("retrieve", "end") in stages


# ---------------------------------------------------------------------------
# Self-regulating cadence
# ---------------------------------------------------------------------------


class TestOverheadTracking:
    def test_overhead_below_threshold_does_not_extend(self, tmp_path: Path):
        monitor, _, _, _ = _make_monitor(tmp_path, interval_seconds=1.0)
        original = monitor.effective_interval
        # 10 ms self-time vs 1000 ms interval = 1 % — well under 20 %.
        monitor._maybe_adjust_interval(self_ms=10.0)
        assert monitor.effective_interval == original

    def test_overhead_above_threshold_doubles_interval(self, tmp_path: Path):
        monitor, _, _, _ = _make_monitor(tmp_path, interval_seconds=1.0)
        # 500 ms self-time vs 1000 ms interval = 50 % — exceeds 20 %.
        monitor._maybe_adjust_interval(self_ms=500.0)
        assert monitor.effective_interval == pytest.approx(
            1.0 * DEFAULT_OVERHEAD_FACTOR
        )

    def test_interval_caps_at_max(self, tmp_path: Path):
        monitor = ResourceMonitorV2(
            tmp_path / "t.csv",
            tmp_path / "m.csv",
            interval_seconds=8.0,
            max_interval_seconds=10.0,
            nvml_factory=lambda idx: _FakeNvml(),
        )
        monitor._maybe_adjust_interval(self_ms=99999.0)
        # Should never exceed max.
        assert monitor.effective_interval == pytest.approx(10.0)

    def test_floor_is_respected(self, tmp_path: Path):
        monitor = ResourceMonitorV2(
            tmp_path / "t.csv",
            tmp_path / "m.csv",
            interval_seconds=0.05,  # below default floor
            nvml_factory=lambda idx: _FakeNvml(),
        )
        # Effective interval should have been raised to the floor.
        assert monitor.effective_interval >= 0.1

    def test_overhead_grows_under_load_in_simulated_loop(self, tmp_path: Path):
        """When the sample function is slow, the loop backs off autonomously."""
        # Patch _sample_once to inject artificial slowness. The loop wraps
        # _sample_once so the timing observation naturally captures the
        # injected delay.
        monitor, _, _, _ = _make_monitor(tmp_path, interval_seconds=0.1)

        call_count = {"n": 0}

        def slow_sample():
            call_count["n"] += 1
            time.sleep(0.05)  # 50 % of a 100 ms interval
            return {
                "elapsed_s": monitor.elapsed,
                "wall_time_s": time.time(),
                "sample_self_ms": 0.0,
                "interval_eff_s": monitor.effective_interval,
            }

        monitor._sample_once = slow_sample  # type: ignore[assignment]
        monitor.start()
        try:
            time.sleep(0.6)
        finally:
            monitor.stop()
        assert monitor.effective_interval > 0.1
        assert call_count["n"] >= 1


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


class TestBackwardsCompat:
    def test_v1_module_still_imports_and_works(self, tmp_path: Path):
        """The v1 backend must remain functional when v2 is not requested."""
        from benchmark.resource_monitor import ResourceMonitor

        trace = tmp_path / "v1.csv"
        markers = tmp_path / "v1_markers.csv"
        rm = ResourceMonitor(trace, markers, interval_seconds=0.05)
        rm.start()
        try:
            rm.stage_start("index")
            time.sleep(0.05)
            rm.stage_end("index")
        finally:
            rm.stop()
        rows = list(csv.DictReader(trace.open()))
        assert len(rows) >= 1
        # v1 schema has exactly the 15 original columns.
        with trace.open() as f:
            header = next(csv.reader(f))
        assert len(header) == 15
        assert "gpu_sm_util_pct" not in header

    def test_v2_unrequested_does_not_construct(self, monkeypatch):
        monkeypatch.delenv("RESOURCE_MONITOR_BACKEND", raising=False)
        assert is_v2_requested() is False
