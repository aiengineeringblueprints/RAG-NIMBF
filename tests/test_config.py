"""Tests for config.py — BenchmarkConfig and get_all_combinations."""

import os
import pytest
from unittest.mock import patch

from config import BenchmarkConfig, get_all_combinations, _parse_list, _parse_int_list, _validate_positive_int


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
        dataset_subset="FinQA",
        dataset_sample_size=50,
        eval_critic_llm="gemma3:12b",
        eval_critic_embedding="nomic-embed-text:latest",
        reranker_model=None,
        reranker_top_k=3,
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


class TestBenchmarkConfig:
    def test_name_property(self):
        cfg = _make_config()
        assert cfg.name == "recursive_cs1000_co200_nomic-embed-text:latest_gemma3:4b"

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
        "OLLAMA_BASE_URL": "http://141.39.193.218/ollama",
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
