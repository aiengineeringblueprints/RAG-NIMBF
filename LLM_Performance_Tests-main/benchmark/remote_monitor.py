"""Remote system monitoring via SSH for DGX Spark / remote GPU servers.

Polls CPU, Memory and GPU stats on a remote host via SSH and:
1. Gibt die Werte LIVE im Terminal aus
2. Trackt Peak-Werte pro Modell-Call (reset zwischen Modellen)
3. Sammelt die Zeitreihe intern — wird am Ende von main.py als
   eigener MLflow-Run geloggt (kein nested Run während des Benchmarks).

Usage:
    monitor = RemoteMonitor(host="141.39.193.218", user="nico")
    monitor.start()

    monitor.reset_peaks()
    ... benchmark für Modell A ...
    peaks = monitor.get_peaks()

    monitor.stop()
    monitor.log_to_mlflow()   # Am Ende, nach allen Experimenten
"""

import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RemoteMonitor:
    """Lightweight remote system monitor via SSH."""

    host: str
    user: str = "nico"
    interval_s: float = 5.0
    ssh_timeout_s: int = 10
    live_output: bool = True

    # Internal state
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _step: int = field(default=0, init=False, repr=False)
    _metrics_log: list[dict] = field(default_factory=list, init=False, repr=False)
    _peaks: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def start(self) -> bool:
        """Start background monitoring thread. Returns True if SSH works."""
        if not self._can_ssh():
            logger.warning(
                "SSH to %s@%s failed — remote monitoring disabled.",
                self.user, self.host,
            )
            print(f"  [Monitor] SSH zu {self.user}@{self.host} "
                  f"fehlgeschlagen — deaktiviert.")
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name=f"monitor-{self.host}"
        )
        self._thread.start()
        print(f"  [Monitor] Gestartet — {self.user}@{self.host} "
              f"alle {self.interval_s}s")
        return True

    def stop(self) -> list[dict]:
        """Stop monitoring and return collected metrics."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.ssh_timeout_s + 2)
        return self._metrics_log

    def reset_peaks(self) -> None:
        """Reset peak tracking — call this before each model test."""
        with self._lock:
            self._peaks.clear()

    def get_peaks(self) -> dict[str, float]:
        """Return peak values since last reset."""
        with self._lock:
            return dict(self._peaks)

    def log_to_mlflow(self) -> None:
        """Log the collected time series as a separate top-level MLflow run.

        Call this AFTER the benchmark is done. Creates a clean,
        non-nested run with all metrics as step-based time series.
        """
        if not self._metrics_log:
            return
        try:
            import mlflow

            with mlflow.start_run(run_name=f"Server Monitor · {self.host}"):
                mlflow.set_tags({
                    "type": "server_monitor",
                    "host": self.host,
                    "user": self.user,
                    "interval_s": str(self.interval_s),
                    "samples": str(len(self._metrics_log)),
                })

                # Metrik-Namen mit Einheiten für die MLflow-UI
                unit_map = {
                    "gpu_util_pct":     "GPU Util pct",
                    "gpu_mem_util_pct": "GPU Mem Util pct",
                    "gpu_temp_c":       "GPU Temp C",
                    "gpu_power_w":      "GPU Power W",
                    "mem_used_gb":      "RAM Used GB",
                    "mem_available_gb": "RAM Available GB",
                    "mem_used_pct":     "RAM Used pct",
                    "cpu_load_1m":      "CPU Load 1m",
                }

                for step, sample in enumerate(self._metrics_log):
                    named: dict[str, float] = {}
                    for k, v in sample.items():
                        if k != "timestamp" and isinstance(v, (int, float)):
                            named[unit_map.get(k, k)] = float(v)
                    if named:
                        mlflow.log_metrics(named, step=step)

                # Gesamt-Peaks als Summary-Metriken (ohne step)
                all_peaks: dict[str, float] = {}
                for sample in self._metrics_log:
                    for key in ("gpu_util_pct", "gpu_mem_util_pct", "gpu_temp_c",
                                "gpu_power_w", "mem_used_gb", "mem_used_pct",
                                "cpu_load_1m"):
                        val = sample.get(key)
                        if val is not None:
                            pk: str = f"Peak {unit_map.get(key, key)}"
                            if pk not in all_peaks or val > all_peaks[pk]:
                                all_peaks[pk] = float(val)
                if all_peaks:
                    mlflow.log_metrics(all_peaks)

            print(f"  [Monitor] {self.host}: {len(self._metrics_log)} "
                  f"Messpunkte in MLflow geloggt.")
        except Exception as exc:
            print(f"  [Monitor] MLflow-Logging fehlgeschlagen: {exc}")

    # ── Private ───────────────────────────────────────────────────────────

    def _can_ssh(self) -> bool:
        try:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5",
                 "-o", "BatchMode=yes",
                 f"{self.user}@{self.host}", "echo ok"],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0 and "ok" in result.stdout
        except Exception:
            return False

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                metrics = self._collect()
                if metrics:
                    self._metrics_log.append(metrics)
                    self._update_peaks(metrics)
                    if self.live_output:
                        self._print_live(metrics)
                    self._step += 1
            except Exception as exc:
                logger.debug("Remote monitor poll failed: %s", exc)

            self._stop_event.wait(self.interval_s)

    def _collect(self) -> dict | None:
        # Single SSH call: free + loadavg + nvidia-smi
        # Für Unified Memory (Spark): auch /proc/meminfo für genauere Werte
        cmd = (
            "free -b 2>/dev/null | grep Mem; "
            "cat /proc/loadavg 2>/dev/null; "
            "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,"
            "temperature.gpu,power.draw "
            "--format=csv,noheader,nounits 2>/dev/null || echo 'N/A'"
        )
        try:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5",
                 "-o", "BatchMode=yes",
                 f"{self.user}@{self.host}", cmd],
                capture_output=True, text=True,
                timeout=self.ssh_timeout_s,
            )
            if result.returncode != 0:
                return None
            return self._parse(result.stdout)
        except Exception:
            return None

    def _parse(self, output: str) -> dict:
        metrics: dict[str, float] = {"timestamp": time.time()}
        lines = output.strip().split("\n")

        for line in lines:
            # free -b: "Mem:  total  used  free  shared  buff/cache  available"
            if line.startswith("Mem:"):
                parts = line.split()
                if len(parts) >= 7:
                    total = int(parts[1])
                    used = int(parts[2])
                    available = int(parts[6])
                    metrics["mem_used_gb"] = round(used / (1024**3), 1)
                    metrics["mem_available_gb"] = round(available / (1024**3), 1)
                    if total > 0:
                        metrics["mem_used_pct"] = round(used / total * 100, 1)

            # /proc/loadavg: "1.23 0.98 0.76 2/345 12345"
            elif "/" in line and not line.startswith("Mem"):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        metrics["cpu_load_1m"] = float(parts[0])
                    except ValueError:
                        pass

            # nvidia-smi: "gpu_util%, mem_util%, temp, power"
            # mem_util% returns N/A on Spark (unified memory) — skipped gracefully
            elif "," in line:
                # Ersetze [N/A] durch leere Strings, damit der Rest noch parsed wird
                cleaned = line.replace("[N/A]", "").replace("N/A", "")
                gpu_parts = [p.strip() for p in cleaned.split(",")]
                if len(gpu_parts) >= 4:
                    for idx, key in enumerate(("gpu_util_pct", "gpu_mem_util_pct",
                                                "gpu_temp_c", "gpu_power_w")):
                        try:
                            if gpu_parts[idx]:
                                metrics[key] = float(gpu_parts[idx])
                        except (ValueError, IndexError):
                            pass

        return metrics

    def _update_peaks(self, metrics: dict) -> None:
        with self._lock:
            for key in ("gpu_util_pct", "gpu_mem_util_pct", "gpu_temp_c",
                         "gpu_power_w", "mem_used_gb", "mem_used_pct",
                         "cpu_load_1m"):
                val = metrics.get(key)
                if val is not None:
                    if key not in self._peaks or val > self._peaks[key]:
                        self._peaks[key] = val

    def _print_live(self, metrics: dict) -> None:
        gpu = metrics.get("gpu_util_pct", -1)
        mem = metrics.get("mem_used_gb", -1)
        mem_pct = metrics.get("mem_used_pct", -1)
        temp = metrics.get("gpu_temp_c", -1)
        power = metrics.get("gpu_power_w", -1)
        load = metrics.get("cpu_load_1m", -1)

        short = self.host.split(".")[-1] if "." in self.host else self.host

        line = (
            f"\r  [{short}] GPU:{gpu:5.1f}%  "
            f"Mem:{mem:5.1f}GB ({mem_pct:.0f}%)  "
            f"Temp:{temp:4.0f}°C  "
            f"Power:{power:5.1f}W  "
            f"CPU:{load:4.1f}  "
        )
        sys.stderr.write(line)
        sys.stderr.flush()
