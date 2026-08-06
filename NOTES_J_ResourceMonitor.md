# J. Resource Monitor v2 (NVML + cgroups v2)

This note documents the optional v2 resource-monitor backend
(`benchmark/resource_monitor_v2.py`) inspired by **RAGPerf §3.4
"Performance Metrics"** (arXiv:2603.10765v1). The v1 backend
(`benchmark/resource_monitor.py`, procfs + `nvidia-smi` shell-out) remains
the default and is untouched.

---

## TL;DR

Set one environment variable to switch backends:

```bash
export BENCHMARK_RESOURCE_MONITOR=true
export RESOURCE_MONITOR_BACKEND=v2
export BENCHMARK_RESOURCE_MONITOR_INTERVAL_SECONDS=0.2
export BENCHMARK_RESOURCE_MONITOR_CGROUP=/sys/fs/cgroup/rag.slice
```

If `nvidia-ml-py` is missing or NVML cannot initialise, `main.py`
automatically prints a warning and falls back to v1.

---

## Why NVML beats `nvidia-smi`

The v1 backend shells out to `nvidia-smi --query-gpu=...` for every sample.
Each invocation forks, execs, parses CSV, and exits — a sub-process that on a
busy GPU node routinely takes **50–150 ms**. That overhead is a multiple of
the desired sample period when you want high-frequency tracing (e.g. 10 Hz)
and it adds jitter that pollutes the trace you are trying to collect.

NVML is the in-process C library that `nvidia-smi` itself wraps. The official
Python binding (`pynvml` / `pip install nvidia-ml-py`) calls NVML directly:

| Property                         | `nvidia-smi` (v1)     | `pynvml` (v2)         |
|----------------------------------|-----------------------|-----------------------|
| Per-sample latency               | ~50–150 ms            | **<1 ms**             |
| Subprocess fork/exec             | Yes                   | No                    |
| `utilization.gpu` update rate    | ~1/6 s                | Same (NVML limitation)|
| **GPM per-engine SM util**       | Not exposed           | **Yes**               |
| **GPM mem-controller util**      | Not exposed           | **Yes**               |
| PCIe rx/tx throughput            | Yes (via query)       | Yes (via NVML query)  |
| Power draw                       | Yes                   | Yes                   |

The GPM (GPU Performance Metrics) interface is the critical addition
RAGPerf calls out: per-engine utilization that `nvidia-smi` does **not**
surface by default. Without it, RAG pipeline authors cannot tell whether a
GPU bottleneck lives in the SM compute path, the memory controller, or PCIe.

---

## Why cgroups v2

`/proc/stat`, `/proc/meminfo`, and `/proc/diskstats` are
**system-wide**. On a host that runs the RAG pipeline alongside other
services, the trace attributes the OS, the database, and background load to
the benchmark — making runs noisy and hard to compare.

cgroups v2 exposes per-cgroup CPU, memory, and IO accounting:

```
/sys/fs/cgroup/<cg>/cpu.stat        # usage_usec, user_usec, system_usec, ...
/sys/fs/cgroup/<cg>/memory.current  # current RSS in bytes
/sys/fs/cgroup/<cg>/io.stat         # per-device IO counters
```

Run the RAG pipeline inside a dedicated cgroup (e.g. a `systemd` slice) and
the v2 backend attributes CPU/memory to *that component only*, removing
host-noise from the trace.

---

## Install

`nvidia-ml-py` is the official NVIDIA-published PyPI wheel that ships
`pynvml`. It is a free, pure-Python binding (no separate native build).

```bash
pip install nvidia-ml-py
```

Add to `requirements.txt` as **optional** — it must not become a hard
dependency, since CI and CPU-only dev boxes do not have a GPU.

---

## YAML / env schema

The backend is configured purely via environment variables for backwards
compatibility with v1's mechanism:

| Env var                                       | Default     | Meaning                                              |
|-----------------------------------------------|-------------|------------------------------------------------------|
| `BENCHMARK_RESOURCE_MONITOR`                  | `false`     | Master switch — both backends respect it.            |
| `RESOURCE_MONITOR_BACKEND`                    | `v1`        | `v1` (procfs/nvidia-smi) or `v2` (NVML/cgroups).     |
| `BENCHMARK_RESOURCE_MONITOR_INTERVAL_SECONDS` | `1.0`       | Requested sample period (seconds). Floored at 0.1 s. |
| `BENCHMARK_RESOURCE_MONITOR_GPU_INDEX`        | `0`         | NVML device index.                                   |
| `BENCHMARK_RESOURCE_MONITOR_CGROUP`           | unset       | Optional cgroups v2 path for per-component metrics.  |

A `config.py`-driven YAML equivalent can be added later; for now env vars
match v1's pattern exactly.

---

## Field reference

The v2 CSV keeps every v1 column **in the same order** so existing analysis
notebooks keep working (DictReader ignores trailing columns). New columns
are appended:

| Column                   | Source                      | Notes                                              |
|--------------------------|-----------------------------|----------------------------------------------------|
| `gpu_sm_util_pct`        | NVML GPM                    | Per-engine SM utilization %. More accurate than `utilization.gpu`. |
| `gpu_mem_ctrl_util_pct`  | NVML GPM                    | Memory-controller utilization %.                   |
| `proc_cpu_pct`           | cgroups v2 `cpu.stat`       | Per-cgroup CPU %, normalised to one logical core.  |
| `proc_mem_mb`            | cgroups v2 `memory.current` | Per-cgroup RSS in MiB.                             |
| `sample_self_ms`         | internal                    | Loop-measured self-time of the sampler. Visible so you can watch the auto-back-off. |
| `interval_eff_s`         | internal                    | The effective (post-adjustment) interval used for the next sleep. |

The full v2 header (in order):

```
elapsed_s, wall_time_s, cpu_user_pct, cpu_system_pct,
host_mem_used_gb, host_mem_pct,
gpu_util_pct, gpu_mem_util_pct, gpu_mem_used_gb, gpu_mem_total_gb,
gpu_power_w, pcie_rx_mb_s, pcie_tx_mb_s,
disk_read_gb_s, disk_write_gb_s,
gpu_sm_util_pct, gpu_mem_ctrl_util_pct, proc_cpu_pct, proc_mem_mb,
sample_self_ms, interval_eff_s
```

---

## Self-regulating cadence

Sub-ms sampling is the design target. If collection overhead grows beyond
**20 % of the requested interval** (e.g. due to driver stalls, a saturated
PCIe bus, or a slow mock backend in tests), the v2 loop **doubles** the
effective interval, capped at `[0.1 s, 10 s]`. This guarantees the monitor
cannot starve the workload it measures. Each row records both
`sample_self_ms` and `interval_eff_s` so post-hoc analysis can see when and
how much back-off occurred.

---

## Migration path from v1

1. `pip install nvidia-ml-py` on the GPU node.
2. Run the RAG pipeline in a cgroup, e.g.:
   ```bash
   sudo systemctl create slice rag.slice   # or systemd-run --slice=rag.slice
   ```
3. Set the env vars from the TL;DR above.
4. Re-run. The CSV schema is a superset of v1, so existing analysis scripts
   that read by column name keep working; scripts that index by column
   number need to be aware of the new trailing columns.
5. To roll back: `unset RESOURCE_MONITOR_BACKEND` (or set to `v1`).

---

## Files

- `benchmark/resource_monitor_v2.py` — the backend.
- `tests/test_resource_monitor_v2.py` — 25 tests covering cgroup parsing,
  metric extraction (mocked NVML), overhead tracking, and v1/v2 backwards
  compatibility.
- `main.py` — picks v2 when `RESOURCE_MONITOR_BACKEND=v2`, falls back to v1
  with a warning if `nvidia-ml-py` is missing.
- `benchmark/resource_monitor.py` — v1, unchanged.
