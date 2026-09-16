"""Build benchmark configuration matrices from experiment manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from itertools import product
from pathlib import Path
from typing import Any

from config import BenchmarkConfig, get_env_combinations, validate_benchmark_config
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
        key: _as_list(value) for key, value in _dict_or_empty(raw.get("matrix")).items()
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
    base = validate_benchmark_config(base)

    matrix = _normalize_matrix({**dataset_matrix, **spec.matrix})
    if not matrix:
        configs = [base]
        _validate_mcp_comparison_fairness(configs)
        return configs

    keys = list(matrix)
    configs: list[BenchmarkConfig] = []
    seen_configs: set[BenchmarkConfig] = set()
    for values in product(*(matrix[key] for key in keys)):
        updates = _updates_for_combo(dict(zip(keys, values)))
        cfg = replace(base, **updates)
        if cfg.chunking_strategy in {"semantic", "provider"}:
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
        cfg = validate_benchmark_config(cfg)
        if cfg in seen_configs:
            continue
        seen_configs.add(cfg)
        configs.append(cfg)
    _validate_mcp_comparison_fairness(configs)
    return configs


def _validate_mcp_comparison_fairness(configs: list[BenchmarkConfig]) -> None:
    """Reject RAG-vs-MCP matrices whose shared experimental controls differ."""
    internal = [config for config in configs if config.rag_system_adapter == "internal"]
    mcp = [
        config
        for config in configs
        if config.rag_system_adapter == "mcp" and config.mcp_enforce_fairness
    ]
    if not internal or not mcp:
        return

    for config in mcp:
        if not config.dataset_corpus_path or not config.mcp_corpus_path:
            raise ValueError(
                "Fair RAG-vs-MCP comparison requires both dataset_corpus_path "
                "and mcp_corpus_path"
            )
        dataset_corpus = Path(config.dataset_corpus_path).resolve()
        mcp_corpus = Path(config.mcp_corpus_path).resolve()
        if dataset_corpus != mcp_corpus:
            raise ValueError(
                "RAG-vs-MCP corpus mismatch: dataset_corpus_path and "
                "mcp_corpus_path must resolve to the same directory"
            )

    def shared_controls(config: BenchmarkConfig) -> tuple[Any, ...]:
        return (
            config.dataset_name,
            config.dataset_subset,
            config.dataset_path,
            config.dataset_corpus_path,
            config.dataset_sample_size,
            config.retrieval_top_k,
        )

    def generated_controls(config: BenchmarkConfig) -> tuple[Any, ...]:
        return shared_controls(config) + (
            config.llm_provider,
            config.llm_model,
            config.prompt_template,
            config.max_new_tokens,
        )

    internal_shared = {shared_controls(config) for config in internal}
    internal_generated = {generated_controls(config) for config in internal}
    missing = [
        config
        for config in mcp
        if (
            shared_controls(config) not in internal_shared
            if config.mcp_result_mode == "answer"
            else generated_controls(config) not in internal_generated
        )
    ]
    if missing:
        raise ValueError(
            "RAG-vs-MCP fairness check failed: every MCP configuration must have "
            "an internal RAG configuration with the same dataset, corpus, top-k, "
            "and (for generated answers) model/prompt controls"
        )


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
                None if str(v).lower() in ("", "none", "null") else v for v in values
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
        new_name = str(dataset["name"])
        updates["dataset_name"] = new_name
        # Keep license metadata in sync with the dataset switch so the
        # reproducibility manifest reflects the actually-selected dataset.
        from benchmark.dataset_adapters import get_adapter

        adapter = get_adapter(new_name)
        updates["dataset_license"] = adapter.license
        updates["dataset_research_only"] = adapter.license_research_only
    if "subset" in dataset:
        updates["dataset_subset"] = (
            "" if dataset["subset"] is None else str(dataset["subset"])
        )
    if "sample_size" in dataset:
        updates["dataset_sample_size"] = int(dataset["sample_size"])
    if "path" in dataset:
        updates["dataset_path"] = (
            None if dataset["path"] is None else str(dataset["path"])
        )
    if "corpus_path" in dataset:
        updates["dataset_corpus_path"] = (
            None if dataset["corpus_path"] is None else str(dataset["corpus_path"])
        )
    if "license" in dataset:
        updates["dataset_license"] = (
            None if dataset["license"] is None else str(dataset["license"])
        )
    return replace(config, **updates) if updates else config


def _apply_settings(
    config: BenchmarkConfig, settings: dict[str, Any]
) -> BenchmarkConfig:
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
        "rag_managed_options_json",
        "ingestion_options_json",
        "mcp_args_json",
        "mcp_http_headers_json",
        "mcp_tool_arguments_json",
        "mcp_allowed_tools_json",
    }:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True, separators=(",", ":"))
        return str(value)
    if key in {
        "chunk_size",
        "chunk_overlap",
        "retrieval_top_k",
        "retrieval_candidate_k",
        "generation_context_k",
        "retrieval_fetch_k",
        "retrieval_multihop_rounds",
        "reranker_top_k",
        "max_new_tokens",
        "dataset_sample_size",
        "eval_critic_max_tokens",
        "semantic_breakpoint_amount",
        "llm_performance_call_counts",
        "mcp_max_agent_rounds",
        "mcp_max_retries",
    }:
        if key == "llm_performance_call_counts":
            values = (
                value if isinstance(value, (list, tuple)) else str(value).split(",")
            )
            return tuple(int(item) for item in values)
        return None if value is None else int(value)
    if key in {
        "retrieval_use_hyde",
        "retrieval_multihop",
        "retrieval_keyword_enabled",
        "rag_managed_reuse_resources",
        "rag_managed_cleanup",
        "llm_answer_value_fallback",
        "ragas_enabled",
        "custom_metrics_enabled",
        "llm_performance_enabled",
        "llm_performance_warmup",
        "mcp_continue_on_error",
        "mcp_enforce_fairness",
    }:
        return _to_bool(value)
    if key in {
        "retrieval_mmr_lambda",
        "rag_http_timeout_seconds",
        "rag_managed_request_timeout_seconds",
        "rag_managed_poll_interval_seconds",
        "rag_managed_poll_timeout_seconds",
        "retrieval_similarity_threshold",
        "retrieval_vector_similarity_weight",
        "generation_temperature",
        "generation_top_p",
        "generation_presence_penalty",
        "generation_frequency_penalty",
        "llm_performance_timeout_seconds",
        "mcp_timeout_seconds",
        "mcp_retry_backoff_seconds",
    }:
        return float(value)
    if key in {
        "reranker_model",
        "rag_http_endpoint_url",
        "rag_http_headers",
        "rag_http_auth_header",
        "rag_http_auth_value",
        "rag_managed_base_url",
        "rag_managed_api_key_env",
        "rag_managed_dataset_prefix",
        "rag_managed_options_json",
        "ingestion_method",
        "ingestion_options_json",
        "dataset_path",
        "dataset_corpus_path",
        "dataset_question_field",
        "dataset_ground_truth_field",
        "dataset_context_field",
        "dataset_metadata_field",
        "mcp_server_url",
        "mcp_command",
        "mcp_args_json",
        "mcp_env_vars",
        "mcp_http_headers_json",
        "mcp_tool_name",
        "mcp_question_argument",
        "mcp_top_k_argument",
        "mcp_tool_arguments_json",
        "mcp_result_field",
        "mcp_allowed_tools_json",
        "mcp_corpus_path",
        "corpus_parser",
        "dataset_license",
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
        "llm_performance_source",
        "mcp_transport",
        "mcp_result_mode",
        "mcp_execution_mode",
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
