"""Reproducibility snapshots for benchmark runs."""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


_SENSITIVE_MARKERS = (
    "api_key",
    "auth_value",
    "password",
    "secret",
    "token",
    "key",
    # Arbitrary provider option objects may contain credentials under names
    # this framework does not know. Treat the complete payload as sensitive.
    "options_json",
    "tool_arguments_json",
)

_ENV_PREFIXES = (
    "BENCHMARK_",
    "CHUNK_",
    "CUSTOM_",
    "DATASET_",
    "EMBEDDING_",
    "EVAL_",
    "GPU_",
    "LANCEDB_",
    "LLM_",
    "MLFLOW_",
    "MCP_",
    "OLLAMA_",
    "OPENAI_COMPAT_",
    "OTEL_",
    "PROMPT_",
    "RAG_",
    "RERANKER_",
    "RETRIEVAL_",
    "SEMANTIC_",
    "VECTOR_",
)


def run_git(args: list[str], cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _redact(name: str, value: Any) -> Any:
    lowered = name.lower()
    if any(marker in lowered for marker in _SENSITIVE_MARKERS):
        return "<redacted>" if value else value
    return value


def _config_to_dict(config: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(config):
        data = dataclasses.asdict(config)
    else:
        data = dict(getattr(config, "__dict__", {}))
    return {key: _redact(key, value) for key, value in sorted(data.items())}


def _tracked_environment() -> dict[str, str]:
    env: dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        if key.startswith(_ENV_PREFIXES):
            env[key] = str(_redact(key, value))
    return env


def _packages() -> list[str]:
    packages = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            packages.append(f"{name}=={dist.version}")
    return sorted(set(packages), key=str.lower)


def build_reproducibility_manifest(configs: list[Any]) -> dict[str, Any]:
    """Build a JSON-serializable run snapshot."""
    git_status = run_git(["status", "--short"]) or ""
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "process": {
            "cwd": str(Path.cwd()),
            "argv": sys.argv,
        },
        "git": {
            "commit": run_git(["rev-parse", "HEAD"]),
            "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "remote": run_git(["remote", "get-url", "origin"]),
            "dirty": bool(git_status),
            "status_short": git_status.splitlines(),
            "diff_stat": run_git(["diff", "--stat"]),
        },
        "environment": _tracked_environment(),
        "experiments": _experiment_metadata(configs),
        "configs": [_config_to_dict(config) for config in configs],
    }


def _experiment_metadata(configs: list[Any]) -> list[dict[str, str]]:
    """Distinct experiment metadata blocks of the planned configs (#65)."""
    from benchmark.lineage.experiment import parse_experiment_metadata

    seen: list[dict[str, str]] = []
    for config in configs:
        metadata = parse_experiment_metadata(getattr(config, "experiment_metadata_json", None))
        if metadata and metadata not in seen:
            seen.append(metadata)
    return seen


def write_reproducibility_bundle(results_dir: Path, configs: list[Any]) -> Path:
    """Write manifest and package freeze files under a results directory."""
    out_dir = results_dir / "reproducibility"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_reproducibility_manifest(configs)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "packages.txt").write_text(
        "\n".join(_packages()) + "\n",
        encoding="utf-8",
    )
    return out_dir
