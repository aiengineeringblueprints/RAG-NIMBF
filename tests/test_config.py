"""Tests for config.py — BenchmarkConfig and get_all_combinations."""

import os
import pytest
from unittest.mock import patch

from config import (
    BenchmarkConfig,
    get_all_combinations,
    _parse_list,
    _parse_int_list,
    _validate_positive_int,
    _chunk_parameter_pairs_for_strategy,
)


# ---------------------------------------------------------------------------
# _parse_list
# ---------------------------------------------------------------------------

class TestParseList:
    def test_single_value(self):
        assert _parse_list("gemma3:4b") == ["gemma3:4b"]

    def test_comma_separated(self):
        assert _parse_list("a, b, c") == ["a", "b", "c"]

    def test_trailing_commas_ignored(self):
        assert _parse_list("a,,b,") == ["a", "b"]

    def test_empty_string(self):
        assert _parse_list("") == []


# ---------------------------------------------------------------------------
# _parse_int_list
# ---------------------------------------------------------------------------

class TestParseIntList:
    def test_single_value(self):
        assert _parse_int_list("1000", "TEST") == [1000]

    def test_multiple_values(self):
        assert _parse_int_list("100, 200, 300", "TEST") == [100, 200, 300]

    def test_invalid_int_raises(self):
        with pytest.raises(ValueError, match="Invalid integer value 'abc'"):
            _parse_int_list("100, abc", "TEST_VAR")


# ---------------------------------------------------------------------------
# _validate_positive_int
# ---------------------------------------------------------------------------

class TestValidatePositiveInt:
    def test_positive_passes(self):
        _validate_positive_int(1, "test")  # no error

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="must be a positive non-zero integer"):
            _validate_positive_int(0, "test")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="must be a positive non-zero integer"):
            _validate_positive_int(-5, "test")


# ---------------------------------------------------------------------------
# BenchmarkConfig
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> BenchmarkConfig:
    defaults = dict(
        llm_model="gemma3:4b",
        llm_provider="ollama",
        embedding_model="nomic-embed-text:latest",
        chunk_size=1000,
        chunk_overlap=200,
        chunking_strategy="recursive",
        retrieval_top_k=5,
        retrieval_strategy="similarity",
        retrieval_fetch_k=None,
        retrieval_mmr_lambda=0.5,
        retrieval_use_hyde=False,
        max_new_tokens=256,
        ollama_base_url="http://localhost:11434",
        ollama_api_key=None,
        openai_compat_base_url=None,
        openai_compat_api_key=None,
        llm_ollama_base_url=None,
        llm_ollama_api_key=None,
        llm_openai_compat_base_url=None,
        llm_openai_compat_api_key=None,
        eval_critic_ollama_base_url=None,
        eval_critic_ollama_api_key=None,
        eval_critic_openai_compat_base_url=None,
        eval_critic_openai_compat_api_key=None,
        embedding_ollama_base_url=None,
        embedding_ollama_api_key=None,
        eval_critic_max_tokens=4096,
        dataset_name="t2-ragbench",
        dataset_subset="FinQA",
        dataset_sample_size=50,
        eval_critic_llm="gemma3:12b",
        eval_critic_embedding="nomic-embed-text:latest",
        reranker_model=None,
        reranker_top_k=3,
        prompt_template="concise",
        llm_answer_strip_mode="tags_only",
        llm_answer_value_fallback=True,
        semantic_breakpoint_type="percentile",
        semantic_breakpoint_amount=95,
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


class TestBenchmarkConfig:
    def test_name_property(self):
        cfg = _make_config()
        assert cfg.name == "recursive_cs1000_co200_nomic-embed-text:latest_gemma3:4b_concise"

    def test_frozen(self):
        cfg = _make_config()
        with pytest.raises(AttributeError):
            cfg.llm_model = "other"

    def test_llm_base_url_ollama(self):
        cfg = _make_config(ollama_base_url="http://server:11434")
        assert cfg.llm_base_url() == "http://server:11434"

    def test_llm_base_url_openai(self):
        cfg = _make_config(
            llm_provider="openai",
            openai_compat_base_url="https://api.example.com/v1",
        )
        assert cfg.llm_base_url() == "https://api.example.com/v1"

    def test_llm_base_url_openai_fallback_empty(self):
        cfg = _make_config(llm_provider="openai", openai_compat_base_url=None)
        assert cfg.llm_base_url() == ""

    def test_llm_api_key_ollama(self):
        cfg = _make_config(ollama_api_key="super")
        assert cfg.llm_api_key() == "super"

    def test_llm_api_key_ollama_none(self):
        cfg = _make_config(ollama_api_key=None)
        assert cfg.llm_api_key() is None

    def test_llm_api_key_openai(self):
        cfg = _make_config(
            llm_provider="openai",
            openai_compat_api_key="sk-test",
        )
        assert cfg.llm_api_key() == "sk-test"

    def test_reranker_model_none_by_default(self):
        cfg = _make_config()
        assert cfg.reranker_model is None
        assert cfg.reranker_top_k == 3

    def test_name_includes_reranker(self):
        cfg = _make_config(reranker_model="huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert "rerank-" in cfg.name
        assert "cross-encoder" in cfg.name

    def test_name_excludes_reranker_when_none(self):
        cfg = _make_config()
        assert "rerank" not in cfg.name

    def test_embedding_provider_ollama_default(self):
        cfg = _make_config(embedding_model="nomic-embed-text:latest")
        assert cfg.embedding_provider == "ollama"

    def test_embedding_provider_huggingface(self):
        cfg = _make_config(embedding_model="huggingface:BAAI/bge-small-en-v1.5")
        assert cfg.embedding_provider == "huggingface"

    def test_retrieval_defaults(self):
        cfg = _make_config()
        assert cfg.retrieval_strategy == "similarity"
        assert cfg.retrieval_fetch_k is None
        assert cfg.retrieval_mmr_lambda == 0.5
        assert cfg.retrieval_use_hyde is False

    def test_name_includes_mmr_when_set(self):
        cfg = _make_config(retrieval_strategy="mmr", retrieval_mmr_lambda=0.7)
        assert "_mmr-l0.7" in cfg.name

    def test_name_excludes_mmr_when_similarity(self):
        cfg = _make_config(retrieval_strategy="similarity")
        assert "mmr" not in cfg.name

    def test_name_includes_hyde_when_enabled(self):
        cfg = _make_config(retrieval_use_hyde=True)
        assert "_hyde" in cfg.name

    def test_name_excludes_hyde_when_disabled(self):
        cfg = _make_config(retrieval_use_hyde=False)
        assert "hyde" not in cfg.name

    def test_semantic_defaults(self):
        cfg = _make_config()
        assert cfg.semantic_breakpoint_type == "percentile"
        assert cfg.semantic_breakpoint_amount == 95

    def test_semantic_name_uses_breakpoint_not_size_overlap(self):
        cfg = _make_config(
            chunking_strategy="semantic", chunk_size=None, chunk_overlap=None
        )
        assert (
            cfg.name
            == "semantic_bppercentile95_nomic-embed-text:latest_gemma3:4b_concise"
        )
        assert "_cs" not in cfg.name
        assert "_coNone" not in cfg.name

    def test_name_includes_external_adapter(self):
        cfg = _make_config(rag_system_adapter="http")
        assert cfg.name.endswith("_http")


class TestChunkParameterPairs:
    def test_semantic_ignores_size_and_overlap_grid(self):
        assert _chunk_parameter_pairs_for_strategy(
            "semantic", [500, 1000], [100, 200]
        ) == [(None, None)]

    def test_non_semantic_keeps_size_overlap_grid(self):
        assert _chunk_parameter_pairs_for_strategy(
            "recursive", [500, 1000], [100, 200]
        ) == [
            (500, 100),
            (500, 200),
            (1000, 100),
            (1000, 200),
        ]

# ---------------------------------------------------------------------------
# get_all_combinations
# ---------------------------------------------------------------------------

class TestGetAllCombinations:
    @patch.dict(os.environ, {
        "LLM_MODELS": "ollama:gemma3:4b",
        "EMBEDDING_MODELS": "nomic-embed-text:latest",
        "CHUNK_SIZES": "1000",
        "CHUNK_OVERLAPS": "200",
        "CHUNKING_STRATEGIES": "recursive",
    }, clear=False)
    def test_single_config(self):
        configs = get_all_combinations()
        assert len(configs) == 1
        assert configs[0].llm_provider == "ollama"
        assert configs[0].llm_model == "gemma3:4b"
        assert configs[0].vector_db_backend == "chroma"
        assert configs[0].benchmark_stage == "all"
        assert configs[0].rag_system_adapter == "internal"

    @patch.dict(os.environ, {
        "RAG_SYSTEM_ADAPTER": "http",
        "RAG_HTTP_ENDPOINT_URL": "http://rag.local/query",
        "RAG_HTTP_ANSWER_FIELD": "result.answer",
        "RAG_HTTP_CONTEXTS_FIELD": "result.contexts",
    }, clear=False)
    def test_http_adapter_config(self):
        configs = get_all_combinations()
        assert configs[0].rag_system_adapter == "http"
        assert configs[0].rag_http_endpoint_url == "http://rag.local/query"
        assert configs[0].rag_http_answer_field == "result.answer"
        assert configs[0].rag_http_contexts_field == "result.contexts"

    @patch.dict(os.environ, {
        "RAG_SYSTEM_ADAPTER": "http",
        "RAG_HTTP_ENDPOINT_URL": "",
    }, clear=False)
    def test_http_adapter_requires_endpoint(self):
        with pytest.raises(ValueError, match="RAG_HTTP_ENDPOINT_URL is required"):
            get_all_combinations()

    @patch.dict(os.environ, {
        "LLM_MODELS": "ollama:gpt-oss:20b,openai:Qwen/Qwen3-32B-AWQ",
        "EMBEDDING_MODELS": "nomic-embed-text:latest",
        "CHUNK_SIZES": "1000",
        "CHUNK_OVERLAPS": "200",
        "CHUNKING_STRATEGIES": "recursive",
        "OPENAI_COMPAT_BASE_URL": "https://api.example.com/v1",
        "OPENAI_COMPAT_API_KEY": "test-key",
    }, clear=False)
    def test_mixed_providers(self):
        configs = get_all_combinations()
        assert len(configs) == 2
        providers = {c.llm_provider for c in configs}
        assert providers == {"ollama", "openai"}

    @patch.dict(os.environ, {
        "LLM_MODELS": "gemma3:4b",
        "EMBEDDING_MODELS": "emb1, emb2",
        "CHUNK_SIZES": "500, 1000",
        "CHUNK_OVERLAPS": "50",
        "CHUNKING_STRATEGIES": "recursive, character",
    }, clear=False)
    def test_cartesian_product(self):
        configs = get_all_combinations()
        # 1 llm x 2 emb x 2 sizes x 1 overlap x 2 strategies = 8
        assert len(configs) == 8

    @patch.dict(os.environ, {
        "LLM_MODELS": "ollama:gpt-oss:20b",
        "OLLAMA_BASE_URL": "http://example.local/ollama",
        "OLLAMA_API_KEY": "super",
        "OPENAI_COMPAT_BASE_URL": "https://optimaise.ddnss.de/v1",
        "OPENAI_COMPAT_API_KEY": "",
        "EMBEDDING_MODELS": "nomic-embed-text:latest",
        "CHUNK_SIZES": "1000",
        "CHUNK_OVERLAPS": "200",
        "CHUNKING_STRATEGIES": "recursive",
    }, clear=False)
    def test_api_keys_loaded(self):
        configs = get_all_combinations()
        assert configs[0].ollama_api_key == "super"
        # empty string becomes None
        assert configs[0].openai_compat_api_key is None

    @patch.dict(os.environ, {
        "LLM_MODELS": "test",
        "EMBEDDING_MODELS": "emb",
        "CHUNK_SIZES": "200",
        "CHUNK_OVERLAPS": "200",
        "CHUNKING_STRATEGIES": "recursive",
    }, clear=False)
    def test_overlap_ge_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            get_all_combinations()

    @patch.dict(os.environ, {
        "LLM_MODELS": "gemma3:4b",
        "EMBEDDING_MODELS": "nomic-embed-text:latest",
        "CHUNK_SIZES": "1000",
        "CHUNK_OVERLAPS": "200",
        "CHUNKING_STRATEGIES": "recursive",
        "RERANKER_MODELS": "huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2",
    }, clear=False)
    def test_reranker_in_combinations(self):
        configs = get_all_combinations()
        assert len(configs) == 1
        assert configs[0].reranker_model == "huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2"

    @patch.dict(os.environ, {
        "LLM_MODELS": "gemma3:4b",
        "EMBEDDING_MODELS": "nomic-embed-text:latest",
        "CHUNK_SIZES": "1000",
        "CHUNK_OVERLAPS": "200",
        "CHUNKING_STRATEGIES": "recursive",
        "RERANKER_MODELS": "",
    }, clear=False)
    def test_no_reranker_same_count(self):
        configs = get_all_combinations()
        assert len(configs) == 1
        assert configs[0].reranker_model is None

    @patch.dict(os.environ, {
        "LLM_MODELS": "gemma3:4b",
        "EMBEDDING_MODELS": "nomic-embed-text:latest",
        "CHUNK_SIZES": "1000",
        "CHUNK_OVERLAPS": "200",
        "CHUNKING_STRATEGIES": "recursive",
        "LLM_ANSWER_STRIP_MODE": "bogus",
    }, clear=False)
    def test_invalid_strip_mode_raises(self):
        with pytest.raises(ValueError, match="LLM_ANSWER_STRIP_MODE"):
            get_all_combinations()

    @patch.dict(os.environ, {
        "LLM_MODELS": "gemma3:4b",
        "EMBEDDING_MODELS": "nomic-embed-text:latest",
        "CHUNK_SIZES": "1000",
        "CHUNK_OVERLAPS": "200",
        "CHUNKING_STRATEGIES": "recursive",
        "RETRIEVAL_STRATEGY": "invalid",
    }, clear=False)
    def test_invalid_retrieval_strategy_raises(self):
        with pytest.raises(ValueError, match="RETRIEVAL_STRATEGY"):
            get_all_combinations()

    @patch.dict(os.environ, {
        "LLM_MODELS": "gemma3:4b",
        "EMBEDDING_MODELS": "nomic-embed-text:latest",
        "CHUNK_SIZES": "1000",
        "CHUNK_OVERLAPS": "200",
        "CHUNKING_STRATEGIES": "recursive",
        "RETRIEVAL_MMR_LAMBDA": "1.5",
    }, clear=False)
    def test_invalid_mmr_lambda_raises(self):
        with pytest.raises(ValueError, match="RETRIEVAL_MMR_LAMBDA"):
            get_all_combinations()

    @patch.dict(os.environ, {
        "LLM_MODELS": "gemma3:4b",
        "EMBEDDING_MODELS": "nomic-embed-text:latest",
        "CHUNK_SIZES": "1000",
        "CHUNK_OVERLAPS": "200",
        "CHUNKING_STRATEGIES": "recursive",
        "SEMANTIC_BREAKPOINT_TYPE": "invalid",
    }, clear=False)
    def test_invalid_breakpoint_type_raises(self):
        with pytest.raises(ValueError, match="SEMANTIC_BREAKPOINT_TYPE"):
            get_all_combinations()

    @patch.dict(os.environ, {
        "LLM_MODELS": "gemma3:4b",
        "EMBEDDING_MODELS": "nomic-embed-text:latest",
        "CHUNK_SIZES": "1000",
        "CHUNK_OVERLAPS": "200",
        "CHUNKING_STRATEGIES": "recursive",
        "RETRIEVAL_STRATEGY": "mmr",
        "RETRIEVAL_FETCH_K": "25",
        "RETRIEVAL_MMR_LAMBDA": "0.7",
    }, clear=False)
    def test_mmr_config_loaded(self):
        configs = get_all_combinations()
        assert len(configs) == 1
        assert configs[0].retrieval_strategy == "mmr"
        assert configs[0].retrieval_fetch_k == 25
        assert configs[0].retrieval_mmr_lambda == 0.7

    @patch.dict(os.environ, {
        "LLM_MODELS": "gemma3:4b",
        "EMBEDDING_MODELS": "nomic-embed-text:latest",
        "CHUNK_SIZES": "1000",
        "CHUNK_OVERLAPS": "200",
        "CHUNKING_STRATEGIES": "recursive",
        "RETRIEVAL_USE_HYDE": "true",
    }, clear=False)
    def test_hyde_config_loaded(self):
        configs = get_all_combinations()
        assert configs[0].retrieval_use_hyde is True

    @patch.dict(os.environ, {
        "LLM_MODELS": "gemma3:4b",
        "EMBEDDING_MODELS": "nomic-embed-text:latest",
        "CHUNK_SIZES": "1000",
        "CHUNK_OVERLAPS": "200",
        "CHUNKING_STRATEGIES": "recursive",
        "VECTOR_DB_BACKEND": "lancedb",
        "LANCEDB_PATH": ".test-lancedb",
        "BENCHMARK_STAGE": "query",
    }, clear=False)
    def test_backend_and_stage_loaded(self):
        configs = get_all_combinations()
        assert configs[0].vector_db_backend == "lancedb"
        assert configs[0].lancedb_path == ".test-lancedb"
        assert configs[0].benchmark_stage == "query"

    @patch.dict(os.environ, {
        "LLM_MODELS": "gemma3:4b",
        "EMBEDDING_MODELS": "nomic-embed-text:latest",
        "CHUNK_SIZES": "1000",
        "CHUNK_OVERLAPS": "200",
        "CHUNKING_STRATEGIES": "recursive",
        "VECTOR_DB_BACKEND": "bogus",
    }, clear=False)
    def test_invalid_vector_db_backend_raises(self):
        with pytest.raises(ValueError, match="VECTOR_DB_BACKEND"):
            get_all_combinations()

    @patch.dict(os.environ, {
        "LLM_MODELS": "gemma3:4b",
        "EMBEDDING_MODELS": "nomic-embed-text:latest",
        "CHUNK_SIZES": "1000",
        "CHUNK_OVERLAPS": "200",
        "CHUNKING_STRATEGIES": "recursive",
        "RETRIEVAL_MODE": "direct",
        "BENCHMARK_STAGE": "index",
    }, clear=False)
    def test_index_stage_requires_retrieval_mode(self):
        with pytest.raises(ValueError, match="BENCHMARK_STAGE=index"):
            get_all_combinations()


    @patch.dict(os.environ, {
        "LLM_MODELS": "gemma3:4b",
        "EMBEDDING_MODELS": "nomic-embed-text:latest",
        "CHUNK_SIZES": "1000",
        "CHUNK_OVERLAPS": "200",
        "CHUNKING_STRATEGIES": "recursive",
        "CUSTOM_RETRIEVAL_METRICS_MODE": "gold_doc",
    }, clear=False)
    def test_custom_retrieval_metrics_mode_loaded(self):
        configs = get_all_combinations()
        assert configs[0].custom_retrieval_metrics_mode == "gold_doc"

    @patch.dict(os.environ, {
        "LLM_MODELS": "gemma3:4b",
        "EMBEDDING_MODELS": "nomic-embed-text:latest",
        "CHUNK_SIZES": "1000",
        "CHUNK_OVERLAPS": "200",
        "CHUNKING_STRATEGIES": "recursive",
        "CUSTOM_RETRIEVAL_METRICS_MODE": "bogus",
    }, clear=False)
    def test_invalid_custom_retrieval_metrics_mode_raises(self):
        with pytest.raises(ValueError, match="CUSTOM_RETRIEVAL_METRICS_MODE"):
            get_all_combinations()

    @patch.dict(os.environ, {
        "LLM_MODELS": "gemma3:4b",
        "EMBEDDING_MODELS": "nomic-embed-text:latest",
        "CHUNK_SIZES": "500,1000",
        "CHUNK_OVERLAPS": "100,200",
        "CHUNKING_STRATEGIES": "semantic",
        "PROMPT_TEMPLATES": "detailed,concise",
        "RERANKER_MODELS": "",
    }, clear=False)
    def test_semantic_combinations_ignore_size_overlap_grid(self):
        configs = get_all_combinations()
        assert len(configs) == 2
        assert {c.prompt_template for c in configs} == {"detailed", "concise"}
        assert {c.chunk_size for c in configs} == {None}
        assert {c.chunk_overlap for c in configs} == {None}
        assert all("_cs" not in c.name and "_coNone" not in c.name for c in configs)

    @patch.dict(os.environ, {
        "LLM_MODELS": "gemma3:4b",
        "EMBEDDING_MODELS": "nomic-embed-text:latest",
        "CHUNK_SIZES": "500,1000",
        "CHUNK_OVERLAPS": "100,200",
        "CHUNKING_STRATEGIES": "recursive,semantic",
        "RERANKER_MODELS": "",
    }, clear=False)
    def test_mixed_chunking_only_collapses_semantic_grid(self):
        configs = get_all_combinations()
        recursive = [c for c in configs if c.chunking_strategy == "recursive"]
        semantic = [c for c in configs if c.chunking_strategy == "semantic"]
        assert len(recursive) == 4
        assert len(semantic) == 1
        assert semantic[0].chunk_size is None
        assert semantic[0].chunk_overlap is None

    @patch.dict(os.environ, {
        "LLM_MODELS": "gemma3:4b",
        "EMBEDDING_MODELS": "nomic-embed-text:latest",
        "CHUNK_SIZES": "200",
        "CHUNK_OVERLAPS": "200",
        "CHUNKING_STRATEGIES": "semantic",
        "RERANKER_MODELS": "",
    }, clear=False)
    def test_semantic_only_allows_ignored_overlap_equal_size(self):
        configs = get_all_combinations()
        assert len(configs) == 1
        assert configs[0].chunk_size is None
        assert configs[0].chunk_overlap is None
