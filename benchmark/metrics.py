import subprocess
import time


_GPU_CACHE: tuple[float, dict | None] | None = None


def get_gpu_usage(cache_seconds: float = 0.0) -> dict | None:
    """Return GPU usage, optionally reusing a recent nvidia-smi sample."""
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
                "--query-gpu=utilization.gpu,memory.used,memory.total",
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
        gpu_util, mem_used, mem_total = [float(v.strip()) for v in line.split(",")]

        usage = {
            "gpu_utilization_pct": gpu_util,
            "memory_used_mb": mem_used,
            "memory_total_mb": mem_total,
        }
        _GPU_CACHE = (now, usage)
        return usage
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        _GPU_CACHE = (now, None)
        return None
