import subprocess


def get_gpu_usage() -> dict | None:
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
            return None

        line = result.stdout.strip().split("\n")[0]
        gpu_util, mem_used, mem_total = [float(v.strip()) for v in line.split(",")]

        return {
            "gpu_utilization_pct": gpu_util,
            "memory_used_mb": mem_used,
            "memory_total_mb": mem_total,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None
