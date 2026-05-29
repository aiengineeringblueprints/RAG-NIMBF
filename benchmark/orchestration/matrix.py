"""Build benchmark configuration matrices from experiment manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from itertools import product
from pathlib import Path
from typing import Any

from config import BenchmarkConfig, get_env_combinations
from benchmark.providers import parse_model_id


@dataclass(frozen=True)
class ExperimentSpec:
    """User-facing experiment plan loaded from JSON/YAML."""

    name: str
    dataset: dict[str, Any]
    matrix: dict[str, list[Any]]
    settings: dict[str, Any]


def load_experiment_spec(path: Path) -> ExperimentSpec:
    """Load an experiment manifest from JSON or YAML."""
    raw_text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(raw_text)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML experiment files require PyYAML. Install project "
                "requirements or use a JSON manifest."
            ) from exc
        raw = yaml.safe_load(raw_text)
    else:
        raise ValueError("Experiment manifest must end with .json, .yaml, or .yml")

    if not isinstance(raw, dict):
        raise ValueError("Experiment manifest must contain an object at the top level")

    name = str(raw.get("experiment_name") or raw.get("name") or path.stem)
    dataset = _dict_or_empty(raw.get("dataset"))
    matrix = {
        key: _as_list(value)
        for key, value in _dict_or_empty(raw.get("matrix")).items()
    }
    settings = _dict_or_empty(raw.get("settings"))
    return ExperimentSpec(name=name, dataset=dataset, matrix=matrix, settings=settings)


def build_configs_from_spec(
    spec: ExperimentSpec | None = None,
    base_configs: list[BenchmarkConfig] | None = None,
) -> list[BenchmarkConfig]:
    """Return concrete BenchmarkConfig objects for an optional manifest.

    With no spec, this preserves the existing .env-driven grid behavior.
    With a spec, .env still provides defaults for omitted fields, while
    dataset/settings/matrix entries override the base config.
    """
    env_configs = base_configs if base_configs is not None else get_env_combinations()
    if spec is None:
        return env_configs

    base = env_configs[0]
    dataset_scalars, dataset_matrix = _split_dataset_config(spec.dataset)
    base = _apply_dataset(base, dataset_scalars)
    base = _apply_settings(base, spec.settings)

    matrix = _normalize_matrix({**dataset_matrix, **spec.matrix})
    if not matrix:
        return [base]

    keys = list(matrix)
    configs: list[BenchmarkConfig] = []
    seen_configs: set[BenchmarkConfig] = set()
    for values in product(*(matrix[key] for key in keys)):
        updates = _updates_for_combo(dict(zip(keys, values)))
        cfg = replace(base, **updates)
        if cfg.chunking_strategy == "semantic":
            cfg = replace(cfg, chunk_size=None, chunk_overlap=None)
        elif cfg.chunk_size is None or cfg.chunk_overlap is None:
            raise ValueError(
                "Non-semantic chunking requires chunk_size and chunk_overlap"
            )
        if cfg.chunk_overlap is not None and cfg.chunk_size is not None:
            if cfg.chunk_overlap >= cfg.chunk_size:
                raise ValueError(
                    f"chunk_overlap ({cfg.chunk_overlap}) must be less than "
                    f"chunk_size ({cfg.chunk_size})"
                )
        if cfg in seen_configs:
            continue
        seen_configs.add(cfg)
        configs.append(cfg)
    return configs


def summarize_matrix(configs: list[BenchmarkConfig]) -> dict[str, Any]:
    """Return a compact dry-run summary."""
    models = sorted({c.llm_model for c in configs})
    embeddings = sorted({c.embedding_model for c in configs})
    datasets = sorted({f"{c.dataset_name}/{c.dataset_subset}" for c in configs})
    sample_sizes = sorted({c.dataset_sample_size for c in configs})
    return {
        "num_configs": len(configs),
        "models": models,
        "embedding_models": embeddings,
        "datasets": datasets,
        "sample_sizes": sample_sizes,
        "total_questions": sum(c.dataset_sample_size for c in configs),
    }


def _dict_or_empty(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Expected an object in experiment manifest")
    return value


def _split_dataset_config(
    dataset: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    scalars: dict[str, Any] = {}
    matrix: dict[str, list[Any]] = {}
    for key, value in dataset.items():
        if isinstance(value, list):
            matrix[f"dataset_{key}"] = value
        else:
            scalars[key] = value
    return scalars, matrix


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _normalize_matrix(matrix: dict[str, list[Any]]) -> dict[str, list[Any]]:
    normalized: dict[str, list[Any]] = {}
    for key, values in matrix.items():
        if not values:
            continue
        if key == "chunking_strategies":
            normalized["chunking_strategy"] = values
        elif key == "chunk_sizes":
            normalized["chunk_size"] = values
        elif key == "chunk_overlaps":
            normalized["chunk_overlap"] = values
        elif key == "llm_models":
            normalized["llm_model"] = values
        elif key == "embedding_models":
            normalized["embedding_model"] = values
        elif key == "prompt_templates":
            normalized["prompt_template"] = values
        elif key == "reranker_models":
            normalized["reranker_model"] = [
                None if str(v).lower() in ("", "none", "null") else v
                for v in values
            ]
        elif key in ("dataset_name", "dataset_names"):
            normalized["dataset_name"] = values
        elif key in ("dataset_subset", "dataset_subsets"):
            normalized["dataset_subset"] = values
        elif key in ("dataset_sample_size", "dataset_sample_sizes"):
            normalized["dataset_sample_size"] = values
        else:
            normalized[key] = values
    return normalized


def _apply_dataset(config: BenchmarkConfig, dataset: dict[str, Any]) -> BenchmarkConfig:
    updates: dict[str, Any] = {}
    if "name" in dataset:
        updates["dataset_name"] = str(dataset["name"])
    if "subset" in dataset:
        updates["dataset_subset"] = "" if dataset["subset"] is None else str(dataset["subset"])
    if "sample_size" in dataset:
        updates["dataset_sample_size"] = int(dataset["sample_size"])
    return replace(config, **updates) if updates else config


def _apply_settings(config: BenchmarkConfig, settings: dict[str, Any]) -> BenchmarkConfig:
    allowed = {field.name for field in fields(BenchmarkConfig)}
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise ValueError(f"Unknown settings field(s): {', '.join(unknown)}")
    updates = {key: _coerce_field_value(key, value) for key, value in settings.items()}
    return replace(config, **updates) if updates else config


def _updates_for_combo(combo: dict[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(BenchmarkConfig)}
    updates: dict[str, Any] = {}
    for key, value in combo.items():
        if key not in allowed:
            raise ValueError(f"Unknown matrix field: {key}")
        if key == "llm_model":
            provider, model_name = parse_model_id(str(value))
            updates["llm_provider"] = provider
            updates["llm_model"] = model_name
        else:
            updates[key] = _coerce_field_value(key, value)
    return updates


def _coerce_field_value(key: str, value: Any) -> Any:
    if key in {
        "chunk_size",
        "chunk_overlap",
        "retrieval_top_k",
        "retrieval_fetch_k",
        "reranker_top_k",
        "max_new_tokens",
        "dataset_sample_size",
        "eval_critic_max_tokens",
        "semantic_breakpoint_amount",
    }:
        return None if value is None else int(value)
    if key in {"retrieval_use_hyde", "llm_answer_value_fallback"}:
        return _to_bool(value)
    if key in {"retrieval_mmr_lambda", "rag_http_timeout_seconds"}:
        return float(value)
    if key in {
        "reranker_model",
        "rag_http_endpoint_url",
        "rag_http_headers",
        "rag_http_auth_header",
        "rag_http_auth_value",
    }:
        return None if value is None else str(value)
    if key in {
        "benchmark_stage",
        "retrieval_mode",
        "retrieval_strategy",
        "custom_retrieval_metrics_mode",
        "semantic_breakpoint_type",
        "vector_db_backend",
        "rag_system_adapter",
        "llm_answer_strip_mode",
    }:
        return str(value).lower()
    if key == "dataset_subset":
        return "" if value is None else str(value)
    return value


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)
