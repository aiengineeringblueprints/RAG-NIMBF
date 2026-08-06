"""Low-overhead resource tracing via NVML/GPM and cgroups v2.

This module is the v2 backend for ``RESOURCE_MONITOR_BACKEND=v2``. It keeps the
v1 CSV schema (so existing analysis notebooks continue to work) and adds the
extra columns that RAGPerf §3.4 calls out:

* ``gpu_sm_util_pct``        — per-engine SM utilization from the NVML GPM
                                interface (more accurate than the coarse
                                ``utilization.gpu`` polled by ``nvidia-smi``,
                                which only updates ~1/6 s).
* ``gpu_mem_ctrl_util_pct``  — memory-controller utilization from GPM.
* ``proc_cpu_pct``           — per-cgroup CPU % attributed to the RAG pipeline
                                rather than the whole host.
* ``proc_mem_mb``            — per-cgroup RSS-style memory from cgroups v2.

Design goals
------------
1. **No subprocess overhead.** ``nvidia-smi`` shell-out costs ~50–150 ms per
   sample on a busy box; ``pynvml`` calls are sub-millisecond in-process. When
   the user asks for high-frequency sampling (e.g. 10 Hz) the v1 backend
   spends a multiple of the sample budget on fork/exec alone.
2. **Per-component attribution.** ``/proc/`` is system-wide. cgroups v2 lets
   us attribute CPU/memory/IO to the RAG pipeline specifically (or whatever
   cgroup the user pins it into), so background OS noise does not pollute the
   trace.
3. **Self-regulating cadence.** We measure our own sample time. If collection
   overhead grows beyond 20 % of the requested interval we double the
   interval (with a floor and ceiling) so the monitor cannot starve the
   workload it is measuring.
4. **Backwards compatible.** Same CSV schema as v1 plus four new columns at
   the end; old readers ignore trailing columns.

The NVML dependency (``nvidia-ml-py``) is *optional*. If it is missing the v2
backend refuses to start and the caller is expected to fall back to v1 (see
``main.py`` wiring). If the package is present but no GPU is reachable at
``nvmlInit`` time, GPU columns stay empty but the host/cgroup columns still
trace — useful for CPU-only smoke tests.
"""
from __future__ import annotations

import csv
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Optional NVML binding. We lazy-import inside ``_NvmlBackend.__init__`` so the
# module can be imported on hosts without a GPU (e.g. CI) — only constructing a
# monitor that actually needs NVML will surface the ImportError.

logger = logging.getLogger(__name__)


# --- CSV schema -----------------------------------------------------------

# All v1 fields, in v1 order, then the four v2 additions. Existing analysis
# scripts read by column name (csv.DictReader), so trailing columns are safe.
SAMPLE_FIELDS: list[str] = [
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
    # v2 additions ---------------------------------------------------------
    "gpu_sm_util_pct",
    "gpu_mem_ctrl_util_pct",
    "proc_cpu_pct",
    "proc_mem_mb",
    # self-instrumentation (always present in v2 so users can see the
    # auto-adjustment happening) ------------------------------------------
    "sample_self_ms",
    "interval_eff_s",
]


# --- Configuration --------------------------------------------------------

DEFAULT_OVERHEAD_THRESHOLD = 0.20  # if self-time / interval > 20 %, back off
DEFAULT_MIN_INTERVAL_S = 0.1
DEFAULT_MAX_INTERVAL_S = 10.0
DEFAULT_OVERHEAD_FACTOR = 2.0  # multiply interval by this when backing off


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


@dataclass(frozen=True)
class _CgroupCpu:
    usage_usec: int


# --- NVML backend (mockable) ---------------------------------------------

class NvmlNotAvailableError(RuntimeError):
    """Raised when the user asked for v2 but NVML cannot be initialised."""


class _NvmlBackend:
    """Thin wrapper around ``pynvml`` exposing only the calls we need.

    The wrapper exists so unit tests can monkeypatch a single object. It also
    isolates the lazy import so importing this module on a CPU-only host is
    safe.
    """

    def __init__(self, gpu_index: int) -> None:
        try:
            import pynvml  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised by integration
            raise NvmlNotAvailableError(
                "pynvml (nvidia-ml-py) is required for the v2 resource monitor; "
                "install with `pip install nvidia-ml-py` or unset "
                "RESOURCE_MONITOR_BACKEND to use the v1 backend."
            ) from exc
        self._pynvml = pynvml
        self._handle: Any | None = None
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception as exc:  # NVML_ERROR_NO_PERMISSION, _GPU_NOT_FOUND, ...
            logger.warning(
                "NVML init failed (%s); GPU columns will be empty.", exc
            )
            self._handle = None

    @property
    def available(self) -> bool:
        return self._handle is not None

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:  # pragma: no cover - best effort
                pass
            self._handle = None

    # -- Standard NVML queries --------------------------------------------

    def utilization_rates(self) -> tuple[float | None, float | None]:
        if self._handle is None:
            return None, None
        try:
            rates = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            return float(rates.gpu), float(rates.memory)
        except Exception as exc:
            logger.debug("nvmlDeviceGetUtilizationRates failed: %s", exc)
            return None, None

    def memory_info(self) -> tuple[float | None, float | None]:
        """Return ``(used_gb, total_gb)``."""
        if self._handle is None:
            return None, None
        try:
            info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return info.used / 1e9, info.total / 1e9
        except Exception as exc:
            logger.debug("nvmlDeviceGetMemoryInfo failed: %s", exc)
            return None, None

    def power_watts(self) -> float | None:
        if self._handle is None:
            return None
        try:
            # NVML reports milliwatts.
            return self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        except Exception as exc:
            logger.debug("nvmlDeviceGetPowerUsage failed: %s", exc)
            return None

    def pcie_throughput_mb_s(self) -> tuple[float | None, float | None]:
        """Return ``(rx_mb_s, tx_mb_s)`` if the driver supports the query."""
        if self._handle is None:
            return None, None
        nvml = self._pynvml
        rx = tx = None
        try:
            rx_bytes = nvml.nvmlDeviceGetPcieThroughput(
                self._handle, nvml.NVML_PCIE_UTIL_RX_BYTES
            )
            rx = rx_bytes / (1024.0 * 1024.0)  # KB/s -> MB/s
        except Exception as exc:
            logger.debug("nvmlDeviceGetPcieThroughput(rx) failed: %s", exc)
        try:
            tx_bytes = nvml.nvmlDeviceGetPcieThroughput(
                self._handle, nvml.NVML_PCIE_UTIL_TX_BYTES
            )
            tx = tx_bytes / (1024.0 * 1024.0)
        except Exception as exc:
            logger.debug("nvmlDeviceGetPcieThroughput(tx) failed: %s", exc)
        return rx, tx

    # -- GPM per-engine metrics ------------------------------------------

    def gpm_engine_util(self) -> tuple[float | None, float | None]:
        """Return ``(sm_util_pct, mem_ctrl_util_pct)`` via the GPM interface.

        Falls back to ``(None, None)`` if the driver/GPU does not advertise
        GPM support. We use ``nvmlDeviceGetFieldValues`` (the documented
        per-field accessor) when the constants are present on the binding;
        otherwise we return ``None`` rather than guessing field IDs.
        """
        if self._handle is None:
            return None, None
        nvml = self._pynvml
        get_field_values = getattr(nvml, "nvmlDeviceGetFieldValues", None)
        sm_id = getattr(nvml, "NVML_FIELD_GPM_UTIL_SM", None) or getattr(
            nvml, "NVML_GPM_METRIC_UTILIZATION_SM", None
        )
        mem_id = getattr(nvml, "NVML_FIELD_GPM_UTIL_MEMCTRL", None) or getattr(
            nvml, "NVML_GPM_METRIC_UTILIZATION_MEMCTRL", None
        )
        if get_field_values is None or sm_id is None or mem_id is None:
            return None, None
        try:
            sample = get_field_values(self._handle, [sm_id, mem_id])
            sm = self._extractFieldValue(sample, 0)
            mem = self._extractFieldValue(sample, 1)
            return sm, mem
        except Exception as exc:
            logger.debug("nvmlDeviceGetFieldValues(GPM) failed: %s", exc)
            return None, None

    @staticmethod
    def _extractFieldValue(sample: Any, idx: int) -> float | None:
        # The shape returned by nvmlDeviceGetFieldValues varies across
        # nvidia-ml-py releases. Be defensive: support both a list of
        # records with a ``.value`` attr and a list of scalars.
        try:
            entry = sample[idx]
        except (IndexError, TypeError):
            return None
        for attr in ("utilization", "value", "uiVal", "ulVal", "dblVal"):
            v = getattr(entry, attr, None)
            if v is None and isinstance(entry, dict):
                v = entry.get(attr)
            if v is None:
                continue
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
        return None


# --- Cgroup v2 parser -----------------------------------------------------

def _read_cgroup_cpu_stat(cgroup_path: Path) -> _CgroupCpu | None:
    """Parse ``cpu.stat`` for ``usage_usec``."""
    target = cgroup_path / "cpu.stat"
    try:
        text = target.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return None
    for line in text.splitlines():
        if line.startswith("usage_usec "):
            try:
                return _CgroupCpu(usage_usec=int(line.split()[1]))
            except (IndexError, ValueError):
                return None
    return None


def _read_cgroup_memory_current(cgroup_path: Path) -> int | None:
    target = cgroup_path / "memory.current"
    try:
        return int(target.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, PermissionError, ValueError):
        return None


# --- Procfs helpers (mirrors v1) -----------------------------------------

def _read_cpu_times() -> _CpuTimes | None:
    try:
        line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
        parts = [int(v) for v in line.split()[1:9]]
        while len(parts) < 8:
            parts.append(0)
        return _CpuTimes(*parts[:8])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def _cpu_percentages(prev: _CpuTimes | None, current: _CpuTimes | None) -> dict[str, float | None]:
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


# --- Monitor class -------------------------------------------------------

class ResourceMonitorV2:
    """NVML + cgroups v2 sampler with self-regulating cadence.

    Parameters mirror v1's ``ResourceMonitor`` for easy swap-in via the
    ``RESOURCE_MONITOR_BACKEND`` env flag. v2 extras:

    * ``cgroup_path`` — path to a cgroups v2 directory whose ``cpu.stat`` and
      ``memory.current`` we will attribute per-component metrics to.
    * ``nvml_factory`` — injection seam used by tests to substitute a mock
      backend without a real GPU.
    """

    def __init__(
        self,
        trace_path: Path,
        markers_path: Path,
        *,
        interval_seconds: float = 1.0,
        gpu_index: int = 0,
        cgroup_path: str | Path | None = None,
        nvml_factory: Callable[[int], _NvmlBackend] | None = None,
        overhead_threshold: float = DEFAULT_OVERHEAD_THRESHOLD,
        min_interval_seconds: float = DEFAULT_MIN_INTERVAL_S,
        max_interval_seconds: float = DEFAULT_MAX_INTERVAL_S,
    ) -> None:
        self.trace_path = Path(trace_path)
        self.markers_path = Path(markers_path)
        self.interval_seconds = max(interval_seconds, min_interval_seconds)
        self.gpu_index = gpu_index
        self.cgroup_path = Path(cgroup_path) if cgroup_path else None
        self._nvml_factory = nvml_factory or _NvmlBackend
        self._overhead_threshold = overhead_threshold
        self._min_interval = min_interval_seconds
        self._max_interval = max_interval_seconds

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_wall = 0.0
        self._started_perf = 0.0
        self._marker_lock = threading.Lock()
        self._stage_starts: dict[str, float] = {}

        # State for delta-based counters.
        self._prev_cpu: _CpuTimes | None = None
        self._prev_disk: tuple[float, _DiskCounters] | None = None
        self._prev_cgroup_cpu: tuple[float, _CgroupCpu] | None = None

        # Effective interval — auto-extended when self-time is too high.
        self._interval_eff = self.interval_seconds
        # Last sample's loop-measured self-time in ms.
        self._last_self_ms: float = 0.0

        self._nvml: _NvmlBackend | None = None
        self._previous_signal_handler: Any = None

    # -- Context manager --------------------------------------------------

    def __enter__(self) -> "ResourceMonitorV2":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    @property
    def elapsed(self) -> float:
        if self._started_perf <= 0:
            return 0.0
        return time.perf_counter() - self._started_perf

    @property
    def effective_interval(self) -> float:
        """The currently-active interval (after any auto-back-off)."""
        return self._interval_eff

    # -- Lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        # Initialise NVML on the constructing thread (not in the worker) so
        # import errors surface synchronously and main.py can fall back to v1.
        try:
            self._nvml = self._nvml_factory(self.gpu_index)
        except NvmlNotAvailableError:
            raise
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.markers_path.parent.mkdir(parents=True, exist_ok=True)
        self._started_wall = time.time()
        self._started_perf = time.perf_counter()
        self._prev_cpu = _read_cpu_times()
        self._prev_disk = (self.elapsed, _read_disk_counters())
        if self.cgroup_path is not None:
            initial = _read_cgroup_cpu_stat(self.cgroup_path)
            if initial is not None:
                self._prev_cgroup_cpu = (self.elapsed, initial)
        self._write_trace_header()
        self._write_marker_header()
        self._thread = threading.Thread(
            target=self._run,
            name="resource-monitor-v2",
            daemon=True,
        )
        # Low priority: the monitor must never starve the workload it
        # measures. We cannot set nice() from inside a thread on POSIX
        # (setpriority is per-process), but we *can* keep the thread
        # cooperative via small sleep windows + Event.wait().
        self._thread.start()
        self._install_signal_handler()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, self._interval_eff * 2))
        self._thread = None
        self._restore_signal_handler()
        if self._nvml is not None:
            self._nvml.close()
            self._nvml = None

    def _install_signal_handler(self) -> None:
        """Flush the latest sample on SIGTERM/SIGINT so traces are not lost."""

        def _handler(signum: int, frame: Any) -> None:
            try:
                self._sample_once()
            finally:
                # Restore and re-raise so the default exit behaviour wins.
                self._restore_signal_handler()
                # Re-raise: SIGINT default behaviour
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        try:
            self._previous_signal_handler = (
                signal.getsignal(signal.SIGTERM),
                signal.getsignal(signal.SIGINT),
            )
            signal.signal(signal.SIGTERM, _handler)
            signal.signal(signal.SIGINT, _handler)
        except (ValueError, OSError):
            # signal handlers can only be installed from the main thread.
            self._previous_signal_handler = None

    def _restore_signal_handler(self) -> None:
        if self._previous_signal_handler is None:
            return
        try:
            signal.signal(signal.SIGTERM, self._previous_signal_handler[0])
            signal.signal(signal.SIGINT, self._previous_signal_handler[1])
        except (ValueError, OSError):
            pass
        self._previous_signal_handler = None

    # -- Stage markers (matches v1 API) ----------------------------------

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

    # -- Sampling loop ---------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once_timed()
            self._stop.wait(self._interval_eff)
        # Final sample so the trace always contains the tail.
        self._sample_once_timed()

    def _sample_once_timed(self) -> None:
        """Take one sample, time it from the loop's perspective, and
        back off the interval if collection overhead grew.

        The timing wraps the entire ``_sample_once`` call so that any
        injected slowness (real driver stalls, slow mock backends, etc.) is
        observable by the cadence logic — not just the work the sampler
        performs *inside* its own bookkeeping."""
        sample_start = time.perf_counter()
        try:
            self._sample_once()
        finally:
            self_ms = (time.perf_counter() - sample_start) * 1000.0
            # Update the trace row's self-time column retroactively so the
            # user can see what the monitor observed.
            self._last_self_ms = self_ms
            self._maybe_adjust_interval(self_ms)

    def _write_trace_header(self) -> None:
        with self.trace_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=SAMPLE_FIELDS).writeheader()

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

    def _sample_once(self) -> dict[str, Any]:
        """Take one sample, return the row (does NOT write the CSV).

        Returning instead of writing lets the outer ``_sample_once_timed``
        loop attach an authoritative self-time measurement that includes any
        injected slowness. Tests and callers that want a one-shot read
        without writing can still call this method directly.
        """
        elapsed = self.elapsed
        row: dict[str, Any] = {
            "elapsed_s": elapsed,
            "wall_time_s": self._started_wall + elapsed,
        }

        # Host CPU/mem (system-wide /proc) -----------------------------------
        current_cpu = _read_cpu_times()
        row.update(_cpu_percentages(self._prev_cpu, current_cpu))
        self._prev_cpu = current_cpu
        row.update(_host_memory())

        # GPU via NVML -------------------------------------------------------
        row.update(self._gpu_metrics())

        # Disk rates (system-wide /proc) ------------------------------------
        row.update(self._disk_rates(elapsed))

        # Per-component via cgroups v2 --------------------------------------
        row.update(self._cgroup_metrics(elapsed))

        # Placeholders — overwritten by the timed wrapper if used.
        row["sample_self_ms"] = 0.0
        row["interval_eff_s"] = self._interval_eff
        return row

    def _sample_once_timed(self) -> None:
        """Take one sample, time it from the loop's perspective, write the
        row to disk, and back off the interval if collection overhead grew.

        The timing wraps the entire ``_sample_once`` call so that any
        injected slowness (real driver stalls, slow mock backends, etc.) is
        observable by the cadence logic — not just the work the sampler
        performs *inside* its own bookkeeping."""
        sample_start = time.perf_counter()
        row = self._sample_once()
        self_ms = (time.perf_counter() - sample_start) * 1000.0
        row["sample_self_ms"] = self_ms
        self._last_self_ms = self_ms
        self._maybe_adjust_interval(self_ms)
        self._write_row(row)

    def _write_row(self, row: dict[str, Any]) -> None:
        with self.trace_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS)
            writer.writerow({
                key: "" if row.get(key) is None else f"{float(row[key]):.6f}"
                for key in SAMPLE_FIELDS
            })

    def _maybe_adjust_interval(self, self_ms: float) -> None:
        interval_ms = self._interval_eff * 1000.0
        if interval_ms <= 0:
            return
        ratio = self_ms / interval_ms
        if ratio > self._overhead_threshold:
            new_interval = min(
                self._interval_eff * DEFAULT_OVERHEAD_FACTOR,
                self._max_interval,
            )
            if new_interval != self._interval_eff:
                logger.info(
                    "Resource monitor self-overhead %.1f ms / %.1f ms (%.0f%%) "
                    "exceeded threshold; extending interval %.3fs -> %.3fs",
                    self_ms,
                    interval_ms,
                    ratio * 100,
                    self._interval_eff,
                    new_interval,
                )
                self._interval_eff = new_interval

    # -- GPU collection --------------------------------------------------

    def _gpu_metrics(self) -> dict[str, float | None]:
        if self._nvml is None or not self._nvml.available:
            return self._empty_gpu_row()
        gpu_util, gpu_mem_util = self._nvml.utilization_rates()
        mem_used_gb, mem_total_gb = self._nvml.memory_info()
        power_w = self._nvml.power_watts()
        pcie_rx, pcie_tx = self._nvml.pcie_throughput_mb_s()
        sm_util, mem_ctrl_util = self._nvml.gpm_engine_util()
        return {
            "gpu_util_pct": gpu_util,
            "gpu_mem_util_pct": gpu_mem_util,
            "gpu_mem_used_gb": mem_used_gb,
            "gpu_mem_total_gb": mem_total_gb,
            "gpu_power_w": power_w,
            "pcie_rx_mb_s": pcie_rx,
            "pcie_tx_mb_s": pcie_tx,
            "gpu_sm_util_pct": sm_util,
            "gpu_mem_ctrl_util_pct": mem_ctrl_util,
        }

    @staticmethod
    def _empty_gpu_row() -> dict[str, float | None]:
        return {
            "gpu_util_pct": None,
            "gpu_mem_util_pct": None,
            "gpu_mem_used_gb": None,
            "gpu_mem_total_gb": None,
            "gpu_power_w": None,
            "pcie_rx_mb_s": None,
            "pcie_tx_mb_s": None,
            "gpu_sm_util_pct": None,
            "gpu_mem_ctrl_util_pct": None,
        }

    # -- Disk / cgroup helpers ------------------------------------------

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

    def _cgroup_metrics(self, elapsed: float) -> dict[str, float | None]:
        out: dict[str, float | None] = {
            "proc_cpu_pct": None,
            "proc_mem_mb": None,
        }
        if self.cgroup_path is None:
            return out

        # Memory is absolute at any instant — no delta needed.
        mem_bytes = _read_cgroup_memory_current(self.cgroup_path)
        if mem_bytes is not None:
            out["proc_mem_mb"] = mem_bytes / (1024.0 * 1024.0)

        # CPU needs a delta over two samples to be meaningful.
        current = _read_cgroup_cpu_stat(self.cgroup_path)
        if current is None:
            self._prev_cgroup_cpu = None
            return out
        if self._prev_cgroup_cpu is None:
            self._prev_cgroup_cpu = (elapsed, current)
            return out
        prev_elapsed, prev = self._prev_cgroup_cpu
        delta_s = max(elapsed - prev_elapsed, 1e-9)
        usage_delta_usec = max(current.usage_usec - prev.usage_usec, 0)
        # cgroup CPU % is normalized to one logical core: a fully busy cgroup
        # across N cores will report ~100*N. We expose the raw value so
        # callers can multiply by the number of cores they expect to use.
        out["proc_cpu_pct"] = (usage_delta_usec / 1e6) * 100.0 / delta_s
        self._prev_cgroup_cpu = (elapsed, current)
        return out


# --- Env-var helpers ------------------------------------------------------

def enabled_from_env() -> bool:
    raw = os.getenv("BENCHMARK_RESOURCE_MONITOR", "false").strip().lower()
    return raw in ("1", "true", "yes", "on")


def backend_from_env() -> str:
    return os.getenv("RESOURCE_MONITOR_BACKEND", "v1").strip().lower()


def is_v2_requested() -> bool:
    return backend_from_env() == "v2"


def interval_from_env() -> float:
    return float(os.getenv("BENCHMARK_RESOURCE_MONITOR_INTERVAL_SECONDS", "1.0"))


def gpu_index_from_env() -> int:
    return int(os.getenv("BENCHMARK_RESOURCE_MONITOR_GPU_INDEX", "0"))


def cgroup_path_from_env() -> str | None:
    return os.getenv("BENCHMARK_RESOURCE_MONITOR_CGROUP") or None
