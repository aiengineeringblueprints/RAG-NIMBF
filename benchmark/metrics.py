import subprocess
import time
from pathlib import Path


_GPU_CACHE: tuple[float, dict | None] | None = None


def get_gpu_usage(cache_seconds: float = 0.0) -> dict | None:
    """Return GPU usage, optionally reusing a recent nvidia-smi sample.

    Includes instantaneous power draw (``power_draw_w``) so energy can be
    estimated for locally hosted models.
    """
    global _GPU_CACHE
    now = time.monotonic()
    if cache_seconds > 0 and _GPU_CACHE is not None:
        ts, cached = _GPU_CACHE
        if now - ts < cache_seconds:
            return cached

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            _GPU_CACHE = (now, None)
            return None

        line = result.stdout.strip().split("\n")[0]
        parts = [float(v.strip()) for v in line.split(",")]
        gpu_util, mem_used, mem_total = parts[:3]
        power_draw = parts[3] if len(parts) > 3 else None

        usage = {
            "gpu_utilization_pct": gpu_util,
            "memory_used_mb": mem_used,
            "memory_total_mb": mem_total,
            "power_draw_w": power_draw,
        }
        _GPU_CACHE = (now, usage)
        return usage
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        _GPU_CACHE = (now, None)
        return None


def read_host_energy_joules() -> float | None:
    """Return cumulative host (CPU + RAM + package) energy in joules via RAPL.

    Sums ``energy_uj`` across all Intel RAPL powercap domains under
    ``/sys/class/powercap/intel-rapl/``. Returns ``None`` when RAPL is
    unavailable (e.g. AMD/non-Linux). Counters may wrap around
    ``max_energy_range_uj``; callers should use short, same-run deltas.
    """
    base = "/sys/class/powercap/intel-rapl"
    if not Path(base).exists():
        return None
    total_uj: float = 0.0
    try:
        for path in Path(base).rglob("energy_uj"):
            try:
                total_uj += float(path.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                continue
    except (FileNotFoundError, PermissionError):
        return None
    return total_uj / 1_000_000.0


def estimate_energy_cost_usd(
    energy_kwh: float | None, price_per_kwh: float | None
) -> float | None:
    """Estimate energy cost in USD from consumed kWh and a price per kWh."""
    if energy_kwh is None or price_per_kwh is None:
        return None
    return energy_kwh * price_per_kwh
