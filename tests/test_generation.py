"""Tests for benchmark.generation — LLM factory and answer generation."""

from unittest.mock import patch, MagicMock

from langchain_core.language_models.chat_models import BaseChatModel

from benchmark.generation import get_llm, generate_answer, GenerationResult


class TestGetLlm:
    def test_ollama_provider(self):
        llm = get_llm(
            provider="ollama",
            model_name="gemma3:4b",
            base_url="http://localhost:11434",
        )
        assert isinstance(llm, BaseChatModel)

    def test_openai_provider(self):
        llm = get_llm(
            provider="openai",
            model_name="Qwen/Qwen3-32B-AWQ",
            base_url="https://api.example.com/v1",
            api_key="test",
        )
        assert isinstance(llm, BaseChatModel)

    def test_passes_max_tokens(self):
        llm = get_llm(
            provider="ollama",
            model_name="test",
            base_url="http://localhost:11434",
            max_new_tokens=512,
        )
        assert isinstance(llm, BaseChatModel)


class TestGenerateAnswer:
    @patch("benchmark.generation.get_gpu_usage", return_value={"gpu_utilization_pct": 50.0, "memory_used_mb": 1000.0})
    def test_basic_generation(self, mock_gpu):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Paris is the capital of France."
        mock_response.usage_metadata = {"output_tokens": 6}
        mock_llm.invoke.return_value = mock_response

        result = generate_answer(
            mock_llm,
            "What is the capital of France?",
            ["France is a country in Europe. Paris is its capital."],
        )

        assert isinstance(result, GenerationResult)
        assert result.answer == "Paris is the capital of France."
        assert result.token_count == 6
        assert result.total_seconds > 0
        assert result.tokens_per_second > 0
        assert result.gpu_usage == {"gpu_utilization_pct": 50.0, "memory_used_mb": 1000.0}

    @patch("benchmark.generation.get_gpu_usage", return_value=None)
    def test_no_usage_metadata(self, mock_gpu):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test answer"
        mock_response.usage_metadata = None
        mock_llm.invoke.return_value = mock_response

        result = generate_answer(mock_llm, "question", ["context"])

        assert result.token_count == 0
        assert result.tokens_per_second == 0
        assert result.gpu_usage is None

    @patch("benchmark.generation.get_gpu_usage", return_value=None)
    def test_messages_structure(self, mock_gpu):
        """Verify the LLM receives proper system + human messages."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "answer"
        mock_response.usage_metadata = {"output_tokens": 1}
        mock_llm.invoke.return_value = mock_response

        generate_answer(mock_llm, "What is X?", ["ctx1", "ctx2"])

        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0].type == "system"
        assert call_args[1].type == "human"
        # Human message should contain both contexts and the question
        assert "ctx1" in call_args[1].content
        assert "ctx2" in call_args[1].content
        assert "What is X?" in call_args[1].content

    @patch("benchmark.generation.get_gpu_usage", return_value=None)
    def test_custom_template(self, mock_gpu):
        """Verify custom system_prompt and human_template are used."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "42"
        mock_response.usage_metadata = {"output_tokens": 1}
        mock_llm.invoke.return_value = mock_response

        generate_answer(
            mock_llm, "What is X?", ["ctx1"],
            system_prompt="Be brief.",
            human_template="Q: {question}\nC: {context}\nA:",
        )

        call_args = mock_llm.invoke.call_args[0][0]
        assert call_args[0].content == "Be brief."
        assert "Q: What is X?" in call_args[1].content
        assert "C: ctx1" in call_args[1].content
        assert "A:" in call_args[1].content
