"""Tests for benchmark.providers — provider factory and model ID parsing."""

import pytest
from unittest.mock import patch, MagicMock

from benchmark.providers import parse_model_id, get_chat_model


# ---------------------------------------------------------------------------
# parse_model_id
# ---------------------------------------------------------------------------

class TestParseModelId:
    def test_ollama_prefix(self):
        assert parse_model_id("ollama:gemma3:4b") == ("ollama", "gemma3:4b")

    def test_ollama_prefix_multi_colon(self):
        """Model names like 'gpt-oss:20b' contain colons themselves."""
        assert parse_model_id("ollama:gpt-oss:20b") == ("ollama", "gpt-oss:20b")

    def test_openai_prefix(self):
        assert parse_model_id("openai:Qwen/Qwen3-32B-AWQ") == ("openai", "Qwen/Qwen3-32B-AWQ")

    def test_unprefixed_defaults_to_ollama(self):
        assert parse_model_id("nomic-embed-text:latest") == ("ollama", "nomic-embed-text:latest")

    def test_plain_name_no_colon(self):
        assert parse_model_id("mistral") == ("ollama", "mistral")

    def test_non_provider_prefix_treated_as_ollama(self):
        """Colons from model names like 'gemma3:4b' should not be misread."""
        assert parse_model_id("gemma3:4b") == ("ollama", "gemma3:4b")

    def test_huggingface_prefix(self):
        assert parse_model_id("huggingface:BAAI/bge-small-en-v1.5") == (
            "huggingface", "BAAI/bge-small-en-v1.5",
        )

    def test_huggingface_cross_encoder(self):
        assert parse_model_id("huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2") == (
            "huggingface", "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )


# ---------------------------------------------------------------------------
# get_chat_model
# ---------------------------------------------------------------------------

class TestGetChatModel:
    def test_ollama_returns_base_chat_model(self):
        from langchain_core.language_models.chat_models import BaseChatModel

        model = get_chat_model(
            provider="ollama",
            model_name="gemma3:4b",
            base_url="http://localhost:11434",
        )
        assert isinstance(model, BaseChatModel)

    def test_openai_returns_base_chat_model(self):
        from langchain_core.language_models.chat_models import BaseChatModel

        model = get_chat_model(
            provider="openai",
            model_name="Qwen/Qwen3-32B-AWQ",
            base_url="https://example.com/v1",
            api_key="test-key",
        )
        assert isinstance(model, BaseChatModel)

    def test_openai_without_api_key_uses_placeholder(self):
        """Should not crash when api_key is None."""
        model = get_chat_model(
            provider="openai",
            model_name="test-model",
            base_url="https://example.com/v1",
        )
        assert model is not None

    def test_ollama_model_attributes(self):
        model = get_chat_model(
            provider="ollama",
            model_name="gemma3:4b",
            base_url="http://localhost:11434",
            max_tokens=512,
            temperature=0.5,
        )
        # ChatOllama stores these as attributes
        assert model.model == "gemma3:4b"
        assert model.base_url == "http://localhost:11434"

    def test_openai_model_attributes(self):
        model = get_chat_model(
            provider="openai",
            model_name="test-model",
            base_url="https://example.com/v1",
            api_key="mykey",
            max_tokens=100,
        )
        assert model.model_name == "test-model"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider 'huggingface'"):
            get_chat_model(
                provider="huggingface",
                model_name="test",
                base_url="http://localhost",
            )
