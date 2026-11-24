"""The test suite must not depend on the machine-local `.env` or shell env (#60)."""

import os
import subprocess
import sys
from pathlib import Path

import config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_config_dotenv_loading_does_not_leak_into_tests(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("RETRIEVAL_STRATEGY=mmr\n")

    config.load_dotenv(env_file)

    assert "RETRIEVAL_STRATEGY" not in os.environ


def test_env_matrix_uses_code_defaults():
    (cfg,) = config.get_all_combinations()

    assert cfg.retrieval_strategy == "similarity"


def test_benchmark_vars_are_absent_inside_tests():
    # Run standalone this is trivially true; the subprocess test below runs it
    # with the variables exported to prove the suite clears them.
    assert "RETRIEVAL_STRATEGY" not in os.environ
    assert "LLM_MODELS" not in os.environ


def test_shell_exported_benchmark_vars_are_cleared_per_test():
    env = {**os.environ, "RETRIEVAL_STRATEGY": "mmr", "LLM_MODELS": "leaky:1b"}

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_env_isolation.py::test_benchmark_vars_are_absent_inside_tests",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stdout + result.stderr
