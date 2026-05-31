"""Tests for benchmark.generation — LLM factory and answer generation."""

from unittest.mock import patch, MagicMock

from langchain_core.language_models.chat_models import BaseChatModel

from benchmark.generation import (
    get_llm,
    generate_answer,
    GenerationResult,
    strip_thinking,
    extract_concise_fallback,
    normalize_percentage_answer,
    extract_final_value,
    _safe_eval_expr,
    _validate_answer,
    _postprocess_answer,
)


def _mock_stream_chunks(content: str, usage: dict | None = None, additional_kwargs: dict | None = None):
    """Create a mock stream that yields chunks simulating streaming."""
    chunks = []
    # Split content into small chunks to simulate streaming
    chunk_size = max(1, len(content) // 3)
    for i in range(0, len(content), chunk_size):
        chunk = MagicMock()
        chunk.content = content[i:i + chunk_size]
        chunk.additional_kwargs = {}
        chunk.usage_metadata = None
        chunks.append(chunk)
    # Final chunk with usage metadata
    if usage or additional_kwargs:
        final = MagicMock()
        final.content = ""
        final.usage_metadata = usage
        final.additional_kwargs = additional_kwargs or {}
        chunks.append(final)
    return chunks


class TestStripThinking:
    def test_tags_only_strips_think_tags(self):
        raw = "<think\n>internal reasoning</think >\n494"
        assert strip_thinking(raw, "tags_only") == "494"

    def test_full_drops_thinking_heuristic(self):
        raw = "Okay, let's see.\n494"
        assert strip_thinking(raw, "full") == ""
        assert strip_thinking(raw, "tags_only") == raw.strip()

    def test_off_untouched_except_whitespace(self):
        raw = "   <think\n>x\n</think >  "
        result = strip_thinking(raw, "off")
        assert result == raw.strip()


class TestSafeEvalExpr:
    def test_simple_addition(self):
        assert _safe_eval_expr("110 + 74") == 184.0

    def test_simple_division(self):
        assert _safe_eval_expr("473 / 7170") == 473.0 / 7170.0

    def test_parentheses(self):
        assert _safe_eval_expr("74360 / (74360 + 260)") == 74360.0 / (74360 + 260)

    def test_negative(self):
        assert _safe_eval_expr("-384") == -384.0

    def test_complex_expression(self):
        result = _safe_eval_expr("(100 - 50) * 2 / 100")
        assert abs(result - 1.0) < 0.001

    def test_invalid_returns_none(self):
        assert _safe_eval_expr("hello world") is None

    def test_division_by_zero_returns_none(self):
        assert _safe_eval_expr("1 / 0") is None

    def test_unsupported_ops_return_none(self):
        assert _safe_eval_expr("2 ** 3") is None


class TestExtractConciseFallback:
    def test_last_line_number(self):
        assert extract_concise_fallback("a\nb\n12.5%") == "12.5%"

    def test_yes_no(self):
        assert extract_concise_fallback("well\nyes") == "yes"

    def test_empty(self):
        assert extract_concise_fallback("") == ""
        assert extract_concise_fallback("no digits here") == ""

    def test_arithmetic_expression(self):
        assert extract_concise_fallback("74360 / (74360 + 260)") == str(74360.0 / (74360 + 260))

    def test_pipe_separated_takes_first(self):
        assert extract_concise_fallback("11.1 | 10.0") == "11.1"

    def test_trailing_zeros_collapsed(self):
        assert extract_concise_fallback("0.0000000000000000") == "0"

    def test_multi_line_addition(self):
        assert extract_concise_fallback("110\n74") == "184"

    def test_multi_line_float_addition(self):
        result = extract_concise_fallback("10.5\n3.2")
        assert float(result) == 13.7


class TestExtractFinalValue:
    def test_extracts_final_value(self):
        assert extract_final_value("Liability 2014: 494 minus 2013: 0\nFINAL: 494.0") == "494.0"

    def test_no_final_line_returns_original(self):
        assert extract_final_value("just a plain answer") == "just a plain answer"

    def test_final_with_decimal(self):
        assert extract_final_value("Calculation here\nFINAL: 0.269") == "0.269"

    def test_final_zero_collapsed(self):
        assert extract_final_value("Result\nFINAL: 0.000000") == "0"

    def test_final_negative(self):
        assert extract_final_value("Decrease\nFINAL: -0.468") == "-0.468"


class TestNormalizePercentageAnswer:
    def test_no_change_when_already_decimal(self):
        assert normalize_percentage_answer("0.12", "What percentage?") == "0.12"

    def test_no_change_when_zero(self):
        assert normalize_percentage_answer("0", "What percentage?") == "0"

    def test_divides_with_ground_truth_cross_reference(self):
        assert normalize_percentage_answer("12.0", "What percentage?", ground_truth="0.12") == "0.12"

    def test_no_divide_when_ground_truth_also_large(self):
        assert normalize_percentage_answer("12.0", "What is the value?", ground_truth="1200") == "12.0"

    def test_divides_with_keyword_heuristic_no_ground_truth(self):
        result = normalize_percentage_answer("57.2", "What percentage of revenue?")
        assert abs(float(result) - 0.572) < 0.0001

    def test_no_divide_without_keywords(self):
        assert normalize_percentage_answer("12.0", "What was the total revenue?") == "12.0"

    def test_negative_percentage_with_ground_truth(self):
        result = normalize_percentage_answer("-46.8", "percentage change?", ground_truth="-0.468")
        assert abs(float(result) - (-0.468)) < 0.0001

    def test_empty_answer_unchanged(self):
        assert normalize_percentage_answer("", "What percentage?") == ""

    def test_non_numeric_answer_unchanged(self):
        assert normalize_percentage_answer("not a number", "What percentage?") == "not a number"

    def test_percent_sign_in_question(self):
        result = normalize_percentage_answer("72.7", "What % of total?")
        assert abs(float(result) - 0.727) < 0.0001

    def test_large_negative_not_normalized_without_gt(self):
        result = normalize_percentage_answer("-384", "What was the percentage change?")
        assert float(result) == -3.84


class TestValidateAnswer:
    def test_valid_answer(self):
        assert _validate_answer("494.0") is True

    def test_empty_answer(self):
        assert _validate_answer("") is False

    def test_whitespace_only(self):
        assert _validate_answer("   ") is False


class TestPostprocessAnswer:
    def test_finqa_template(self):
        result = _postprocess_answer(
            "Liability 2014: 494\nFINAL: 494.0", None, "question",
            strip_mode="tags_only", value_fallback=True,
            ground_truth=None, prompt_template_name="finqa",
        )
        assert result == "494.0"

    def test_empty_with_fallback(self):
        result = _postprocess_answer(
            "Okay, let's see.\n42", None, "question",
            strip_mode="full", value_fallback=True,
            ground_truth=None, prompt_template_name=None,
        )
        assert result == "42"


class TestGetLlm:
    def test_ollama_provider(self):
        from langchain_core.runnables import RunnableBinding

        llm = get_llm(
            provider="ollama",
            model_name="gemma3:4b",
            base_url="http://localhost:11434",
        )
        assert isinstance(llm, (BaseChatModel, RunnableBinding))

    def test_openai_provider(self):
        llm = get_llm(
            provider="openai",
            model_name="Qwen/Qwen3-32B-AWQ",
            base_url="https://api.example.com/v1",
            api_key="test",
        )
        assert isinstance(llm, BaseChatModel)

    def test_passes_max_tokens(self):
        from langchain_core.runnables import RunnableBinding

        llm = get_llm(
            provider="ollama",
            model_name="test",
            base_url="http://localhost:11434",
            max_new_tokens=512,
        )
        assert isinstance(llm, (BaseChatModel, RunnableBinding))


class TestGenerateAnswer:
    @patch("benchmark.generation.get_gpu_usage", return_value={"gpu_utilization_pct": 50.0, "memory_used_mb": 1000.0})
    def test_basic_generation_with_streaming(self, mock_gpu):
        mock_llm = MagicMock()
        chunks = _mock_stream_chunks("Paris is the capital of France.", usage={"output_tokens": 6})
        mock_llm.stream.return_value = iter(chunks)

        result = generate_answer(
            mock_llm,
            "What is the capital of France?",
            ["France is a country in Europe. Paris is its capital."],
        )

        assert isinstance(result, GenerationResult)
        assert result.answer == "Paris is the capital of France."
        assert result.token_count == 6
        assert result.output_tokens == 6
        assert result.total_seconds > 0
        assert result.ttft_seconds > 0
        assert result.ttft_seconds <= result.total_seconds
        assert result.answer_valid is True
        assert result.gpu_usage == {"gpu_utilization_pct": 50.0, "memory_used_mb": 1000.0}


    @patch("benchmark.generation.get_gpu_usage", return_value=None)
    @patch.dict("os.environ", {
        "LLM_MODEL_PRICING_USD_PER_1M": '{"deepseek-v4-pro": {"input": 1.0, "output": 2.0}}',
    }, clear=False)
    def test_token_split_and_cost_from_usage_metadata(self, mock_gpu):
        mock_llm = MagicMock()
        chunks = _mock_stream_chunks(
            "answer",
            usage={"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
        )
        mock_llm.stream.return_value = iter(chunks)

        result = generate_answer(
            mock_llm, "q", ["ctx"], cost_model_name="deepseek-v4-pro"
        )

        assert result.input_tokens == 1000
        assert result.output_tokens == 500
        assert result.total_tokens == 1500
        assert result.token_count == 500
        assert result.estimated_cost_usd == 0.002

    @patch("benchmark.generation.get_gpu_usage", return_value=None)
    def test_invoke_usage_falls_back_to_response_metadata(self, mock_gpu):
        mock_llm = MagicMock()
        mock_llm.stream.side_effect = NotImplementedError("no streaming")
        mock_response = MagicMock()
        mock_response.content = "Test answer"
        mock_response.usage_metadata = None
        mock_response.additional_kwargs = {}
        mock_response.response_metadata = {
            "token_usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            }
        }
        mock_llm.invoke.return_value = mock_response

        result = generate_answer(mock_llm, "question", ["context"])

        assert result.input_tokens == 7
        assert result.output_tokens == 3
        assert result.total_tokens == 10

    @patch("benchmark.generation.get_gpu_usage", return_value=None)
    def test_fallback_to_invoke(self, mock_gpu):
        """When streaming fails, falls back to invoke."""
        mock_llm = MagicMock()
        mock_llm.stream.side_effect = NotImplementedError("no streaming")
        mock_response = MagicMock()
        mock_response.content = "Test answer"
        mock_response.usage_metadata = None
        mock_response.additional_kwargs = {}
        mock_llm.invoke.return_value = mock_response

        result = generate_answer(mock_llm, "question", ["context"])

        assert result.answer == "Test answer"
        assert result.ttft_seconds == result.total_seconds  # invoke fallback: ttft == total
        mock_llm.invoke.assert_called_once()

    @patch("benchmark.generation.get_gpu_usage", return_value=None)
    def test_messages_structure(self, mock_gpu):
        """Verify the LLM receives proper system + human messages."""
        mock_llm = MagicMock()
        chunks = _mock_stream_chunks("answer", usage={"output_tokens": 1})
        mock_llm.stream.return_value = iter(chunks)

        generate_answer(mock_llm, "What is X?", ["ctx1", "ctx2"])

        call_args = mock_llm.stream.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0].type == "system"
        assert call_args[1].type == "human"
        assert "ctx1" in call_args[1].content
        assert "ctx2" in call_args[1].content
        assert "What is X?" in call_args[1].content

    @patch("benchmark.generation.get_gpu_usage", return_value=None)
    def test_custom_template(self, mock_gpu):
        """Verify custom system_prompt and human_template are used."""
        mock_llm = MagicMock()
        chunks = _mock_stream_chunks("42", usage={"output_tokens": 1})
        mock_llm.stream.return_value = iter(chunks)

        generate_answer(
            mock_llm, "What is X?", ["ctx1"],
            system_prompt="Be brief.",
            human_template="Q: {question}\nC: {context}\nA:",
        )

        call_args = mock_llm.stream.call_args[0][0]
        assert call_args[0].content == "Be brief."
        assert "Q: What is X?" in call_args[1].content
        assert "C: ctx1" in call_args[1].content
        assert "A:" in call_args[1].content

    @patch("benchmark.generation.get_gpu_usage", return_value=None)
    def test_full_strip_with_value_fallback(self, mock_gpu):
        mock_llm = MagicMock()
        chunks = _mock_stream_chunks(
            "Okay, let's see.\nThe result is 42",
            usage={"output_tokens": 3},
            additional_kwargs={},
        )
        mock_llm.stream.return_value = iter(chunks)

        result = generate_answer(
            mock_llm, "q", ["ctx"], strip_mode="full", value_fallback=True,
        )

        assert result.answer == "42"

    @patch("benchmark.generation.get_gpu_usage", return_value=None)
    def test_empty_answer_marks_invalid(self, mock_gpu):
        mock_llm = MagicMock()
        chunks = _mock_stream_chunks("Okay, let's see.", usage={"output_tokens": 3})
        mock_llm.stream.return_value = iter(chunks)

        result = generate_answer(
            mock_llm, "q", ["ctx"], strip_mode="full", value_fallback=False,
        )

        assert result.answer == ""
        assert result.answer_valid is False

    @patch("benchmark.generation.get_gpu_usage", return_value=None)
    def test_finqa_template_extracts_final(self, mock_gpu):
        """Verify finqa template extracts FINAL: value."""
        mock_llm = MagicMock()
        chunks = _mock_stream_chunks(
            "Liability 2014: 494 minus 2013: 0 = 494\nFINAL: 494.0",
            usage={"output_tokens": 10},
        )
        mock_llm.stream.return_value = iter(chunks)

        result = generate_answer(
            mock_llm, "What was the change?", ["ctx"],
            prompt_template_name="finqa",
        )

        assert result.answer == "494.0"

    @patch("benchmark.generation.get_gpu_usage", return_value=None)
    def test_percentage_normalization_with_ground_truth(self, mock_gpu):
        """Verify percentage normalization when ground_truth is provided."""
        mock_llm = MagicMock()
        chunks = _mock_stream_chunks("57.2", usage={"output_tokens": 1})
        mock_llm.stream.return_value = iter(chunks)

        result = generate_answer(
            mock_llm,
            "What percentage of revenue?",
            ["ctx"],
            ground_truth="0.572",
        )

        assert abs(float(result.answer) - 0.572) < 0.0001

    @patch("benchmark.generation.get_gpu_usage", return_value=None)
    def test_ttft_less_than_total(self, mock_gpu):
        """Streaming should give TTFT < total."""
        mock_llm = MagicMock()
        chunks = _mock_stream_chunks("Some answer text here", usage={"output_tokens": 4})
        mock_llm.stream.return_value = iter(chunks)

        result = generate_answer(mock_llm, "q", ["ctx"])

        assert result.ttft_seconds <= result.total_seconds
