"""Tests for resumable experiment orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.orchestration.matrix import (
    ExperimentSpec,
    build_configs_from_spec,
    load_experiment_spec,
    summarize_matrix,
)
from benchmark.orchestration.worker import ProgressStore, config_result_path
from benchmark.clearml_task import (
    config_from_clearml_parameters,
    clearml_parameters_from_config,
)


def _base_env(monkeypatch):
    monkeypatch.setenv("LLM_MODELS", "ollama:gemma3:4b")
    monkeypatch.setenv("EMBEDDING_MODELS", "nomic-embed-text:latest")
    monkeypatch.setenv("CHUNK_SIZES", "1000")
    monkeypatch.setenv("CHUNK_OVERLAPS", "200")
    monkeypatch.setenv("CHUNKING_STRATEGIES", "recursive")
    monkeypatch.setenv("PROMPT_TEMPLATES", "concise")
    monkeypatch.setenv("DATASET_NAME", "t2-ragbench")
    monkeypatch.setenv("DATASET_SUBSET", "FinQA")
    monkeypatch.setenv("DATASET_SAMPLE_SIZE", "10")


def test_manifest_matrix_expands_and_deduplicates_semantic(monkeypatch):
    _base_env(monkeypatch)
    spec = ExperimentSpec(
        name="matrix-test",
        dataset={"name": "t2-ragbench", "subset": "FinQA", "sample_size": 7},
        settings={},
        matrix={
            "llm_models": ["ollama:gemma3:4b", "ollama:qwen3:8b"],
            "chunking_strategies": ["recursive", "semantic"],
            "chunk_sizes": [300, 500],
            "chunk_overlaps": [50],
            "retrieval_top_k": [3, 5],
            "prompt_templates": ["concise"],
        },
    )

    configs = build_configs_from_spec(spec)

    # recursive: 2 models * 2 sizes * 1 overlap * 2 top-k = 8
    # semantic ignores size/overlap: 2 models * 2 top-k = 4
    assert len(configs) == 12
    assert {c.dataset_sample_size for c in configs} == {7}
    assert any(c.chunking_strategy == "semantic" and c.chunk_size is None for c in configs)
    assert any("_k3" in c.name for c in configs)


def test_json_manifest_loads(tmp_path: Path):
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps({
            "experiment_name": "json-test",
            "dataset": {"sample_size": 2},
            "matrix": {"retrieval_top_k": [3]},
        }),
        encoding="utf-8",
    )

    spec = load_experiment_spec(path)

    assert spec.name == "json-test"
    assert spec.dataset == {"sample_size": 2}
    assert spec.matrix == {"retrieval_top_k": [3]}



def test_manifest_dataset_lists_expand_without_dropping_duplicates(monkeypatch):
    _base_env(monkeypatch)
    spec = ExperimentSpec(
        name="dataset-grid",
        dataset={
            "name": ["squad", "ragas-wikiqa"],
            "subset": None,
            "sample_size": [2, 3],
        },
        settings={},
        matrix={"retrieval_top_k": [3]},
    )

    configs = build_configs_from_spec(spec)

    assert len(configs) == 4
    assert {c.dataset_name for c in configs} == {"squad", "ragas-wikiqa"}
    assert {c.dataset_subset for c in configs} == {""}
    assert {c.dataset_sample_size for c in configs} == {2, 3}


def test_get_all_combinations_uses_benchmark_config_file(monkeypatch, tmp_path: Path):
    _base_env(monkeypatch)
    manifest = tmp_path / "experiment.yaml"
    manifest.write_text(
        """
experiment_name: yaml-entrypoint
dataset:
  sample_size: 2
settings:
  retrieval_strategy: mmr
  retrieval_mmr_lambda: 0.7
matrix:
  retrieval_top_k: [3, 5]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("BENCHMARK_CONFIG_FILE", str(manifest))

    from config import get_all_combinations

    configs = get_all_combinations()

    assert len(configs) == 2
    assert {c.retrieval_top_k for c in configs} == {3, 5}
    assert {c.dataset_sample_size for c in configs} == {2}
    assert {c.retrieval_strategy for c in configs} == {"mmr"}
    assert {c.retrieval_mmr_lambda for c in configs} == {0.7}


def test_progress_store_requires_result_file(monkeypatch, tmp_path: Path):
    _base_env(monkeypatch)
    config = build_configs_from_spec()[0]
    progress = ProgressStore(tmp_path / "progress.json")
    result_path = config_result_path(tmp_path, config)

    progress.mark_completed(config, result_path)

    assert not progress.is_completed(config, result_path)
    result_path.parent.mkdir(parents=True)
    result_path.write_text("{}", encoding="utf-8")
    assert progress.is_completed(config, result_path)


def test_summarize_matrix_counts_questions(monkeypatch):
    _base_env(monkeypatch)
    spec = ExperimentSpec(
        name="summary-test",
        dataset={"sample_size": 4},
        settings={},
        matrix={"retrieval_top_k": [3, 5]},
    )
    configs = build_configs_from_spec(spec)

    summary = summarize_matrix(configs)

    assert summary["num_configs"] == 2
    assert summary["total_questions"] == 8


def test_clearml_parameters_exclude_secret_fields(monkeypatch):
    _base_env(monkeypatch)
    config = build_configs_from_spec()[0]

    params = clearml_parameters_from_config(config)

    assert "retrieval_top_k" in params
    assert "llm_openai_compat_api_key" not in params
    assert "rag_http_auth_value" not in params


def test_clearml_parameter_overrides_preserve_provider_without_prefix(monkeypatch):
    _base_env(monkeypatch)
    config = build_configs_from_spec()[0]
    openai_config = config.__class__(
        **{**config.__dict__, "llm_provider": "openai", "llm_model": "baseline-model"}
    )

    updated = config_from_clearml_parameters(
        openai_config,
        {
            "llm_model": "replacement-model",
            "retrieval_top_k": "11",
            "retrieval_fetch_k": None,
            "retrieval_use_hyde": "true",
        },
    )

    assert updated.llm_provider == "openai"
    assert updated.llm_model == "replacement-model"
    assert updated.retrieval_top_k == 11
    assert updated.retrieval_fetch_k is None
    assert updated.retrieval_use_hyde is True


def test_clearml_parameter_overrides_parse_prefixed_llm_model(monkeypatch):
    _base_env(monkeypatch)
    config = build_configs_from_spec()[0]

    updated = config_from_clearml_parameters(
        config,
        {"llm_model": "openai:Qwen/Qwen3-32B-AWQ"},
    )

    assert updated.llm_provider == "openai"
    assert updated.llm_model == "Qwen/Qwen3-32B-AWQ"
