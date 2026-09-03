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
        from langchain_core.runnables import RunnableBinding

        model = get_chat_model(
            provider="ollama",
            model_name="gemma3:4b",
            base_url="http://localhost:11434",
        )
        # bind(think=False) wraps the model in RunnableBinding
        assert isinstance(model, (BaseChatModel, RunnableBinding))

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


# ---------------------------------------------------------------------------
# _TokenCountingChatModel
# ---------------------------------------------------------------------------

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from benchmark.providers import _TokenCountingChatModel


class _FakeChatModel:
    """Stands in for a BaseChatModel, returning canned AIMessages."""

    def __init__(self, responses):
        self._responses = list(responses)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = AIMessage(content="ok", usage_metadata=self._responses.pop(0))
        return ChatResult(generations=[ChatGeneration(message=msg)])


class TestTokenCountingChatModel:
    def test_aggregates_usage_across_calls(self):
        inner = _FakeChatModel([
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            {"input_tokens": 20, "output_tokens": 7, "total_tokens": 27},
        ])
        counter = _TokenCountingChatModel(inner)

        counter._generate([])
        counter._generate([])

        assert counter.input_tokens == 30
        assert counter.output_tokens == 12
        assert counter.total_tokens == 42

    def test_handles_missing_usage_metadata(self):
        inner_calls = {"n": 0}

        class _NoUsage:
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                inner_calls["n"] += 1
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

        counter = _TokenCountingChatModel(_NoUsage())
        counter._generate([])

        assert counter.input_tokens == 0
        assert counter.output_tokens == 0
        assert counter.total_tokens == 0

    def test_accepts_openai_style_token_keys(self):
        class _OpenAIStyle:
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                msg = AIMessage(
                    content="ok",
                    response_metadata={"token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}},
                )
                return ChatResult(generations=[ChatGeneration(message=msg)])

        counter = _TokenCountingChatModel(_OpenAIStyle())
        counter._generate([])

        assert counter.input_tokens == 100
        assert counter.output_tokens == 50
        assert counter.total_tokens == 150
