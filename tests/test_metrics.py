"""Tests for benchmark.metrics — GPU usage monitoring."""

from unittest.mock import patch, MagicMock
import subprocess

from benchmark.metrics import get_gpu_usage


class TestGetGpuUsage:
    @patch("benchmark.metrics.subprocess.run")
    def test_successful_gpu_read(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="45, 2048, 8192\n",
        )
        result = get_gpu_usage()

        assert result is not None
        assert result["gpu_utilization_pct"] == 45.0
        assert result["memory_used_mb"] == 2048.0
        assert result["memory_total_mb"] == 8192.0

    @patch("benchmark.metrics.subprocess.run")
    def test_nvidia_smi_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        assert get_gpu_usage() is None

    @patch("benchmark.metrics.subprocess.run")
    def test_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)
        assert get_gpu_usage() is None

    @patch("benchmark.metrics.subprocess.run")
    def test_nonzero_return_code(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert get_gpu_usage() is None

    @patch("benchmark.metrics.subprocess.run")
    def test_malformed_output(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="not,valid,data,extra\n")
        # Too many values: should raise ValueError which is caught
        assert get_gpu_usage() is None
