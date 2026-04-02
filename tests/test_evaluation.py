"""Tests for benchmark.evaluation — RAGAS evaluation pipeline."""

import math

import pytest
from unittest.mock import patch, MagicMock

from benchmark.evaluation import evaluate_results, EvaluationResult


class TestEvaluateResultsEmpty:
    def test_empty_questions_returns_error(self):
        result = evaluate_results(
            questions=[],
            ground_truths=[],
            answers=[],
            contexts=[],
        )
        assert isinstance(result, EvaluationResult)
        assert result.error == "No questions provided for evaluation."
        assert result.metric_means == {}
        assert result.per_sample_scores == []


class TestEvaluateResultsModelInit:
    """Test that provider routing works during critic model initialization."""

    @patch("benchmark.evaluation.LangchainEmbeddingsWrapper")
    @patch("benchmark.evaluation.get_embedding_model")
    @patch("benchmark.evaluation.LangchainLLMWrapper")
    @patch("benchmark.evaluation.get_chat_model")
    @patch("benchmark.evaluation.evaluate")
    def test_ollama_critic_routes_correctly(self, mock_evaluate, mock_get_chat, mock_llm_wrap, mock_get_emb, mock_emb_wrap):
        mock_get_chat.return_value = MagicMock()
        mock_llm_wrap.return_value = MagicMock()
        mock_get_emb.return_value = MagicMock()
        mock_emb_wrap.return_value = MagicMock()

        mock_result = MagicMock()
        mock_result.scores = [{"faithfulness": 0.9, "answer_relevancy": 0.8}]
        mock_evaluate.return_value = mock_result

        result = evaluate_results(
            questions=["q1"],
            ground_truths=["gt1"],
            answers=["a1"],
            contexts=[["ctx1"]],
            critic_llm_model="ollama:gemma3:12b",
            ollama_base_url="http://server:11434",
            ollama_api_key="mykey",
        )

        call_kwargs = mock_get_chat.call_args[1]
        assert call_kwargs["provider"] == "ollama"
        assert call_kwargs["base_url"] == "http://server:11434"
        assert call_kwargs["api_key"] == "mykey"

    @patch("benchmark.evaluation.LangchainEmbeddingsWrapper")
    @patch("benchmark.evaluation.get_embedding_model")
    @patch("benchmark.evaluation.LangchainLLMWrapper")
    @patch("benchmark.evaluation.get_chat_model")
    @patch("benchmark.evaluation.evaluate")
    def test_openai_critic_routes_correctly(self, mock_evaluate, mock_get_chat, mock_llm_wrap, mock_get_emb, mock_emb_wrap):
        mock_get_chat.return_value = MagicMock()
        mock_llm_wrap.return_value = MagicMock()
        mock_get_emb.return_value = MagicMock()
        mock_emb_wrap.return_value = MagicMock()

        mock_result = MagicMock()
        mock_result.scores = [{"faithfulness": 0.9}]
        mock_evaluate.return_value = mock_result

        result = evaluate_results(
            questions=["q1"],
            ground_truths=["gt1"],
            answers=["a1"],
            contexts=[["ctx1"]],
            critic_llm_model="openai:Qwen/Qwen3-32B-AWQ",
            ollama_base_url="http://server:11434",
            openai_compat_base_url="https://api.example.com/v1",
            openai_compat_api_key="sk-test",
        )

        call_kwargs = mock_get_chat.call_args[1]
        assert call_kwargs["provider"] == "openai"
        assert call_kwargs["base_url"] == "https://api.example.com/v1"
        assert call_kwargs["api_key"] == "sk-test"

    @patch("benchmark.evaluation.LangchainEmbeddingsWrapper")
    @patch("benchmark.evaluation.get_embedding_model")
    @patch("benchmark.evaluation.LangchainLLMWrapper")
    @patch("benchmark.evaluation.get_chat_model")
    @patch("benchmark.evaluation.evaluate")
    def test_embedding_factory_called_with_provider(self, mock_evaluate, mock_get_chat, mock_llm_wrap, mock_get_emb, mock_emb_wrap):
        mock_get_chat.return_value = MagicMock()
        mock_llm_wrap.return_value = MagicMock()
        mock_get_emb.return_value = MagicMock()
        mock_emb_wrap.return_value = MagicMock()

        mock_result = MagicMock()
        mock_result.scores = [{"faithfulness": 0.9}]
        mock_evaluate.return_value = mock_result

        evaluate_results(
            questions=["q1"],
            ground_truths=["gt1"],
            answers=["a1"],
            contexts=[["ctx1"]],
            embedding_model="huggingface:BAAI/bge-small-en-v1.5",
            ollama_base_url="http://server:11434",
            ollama_api_key="bearer-token",
        )

        # Verify get_embedding_model was called with huggingface provider
        emb_call_kwargs = mock_get_emb.call_args[1]
        assert emb_call_kwargs["provider"] == "huggingface"
        assert mock_get_emb.call_args[0][0] == "BAAI/bge-small-en-v1.5"

    @patch("benchmark.evaluation.LangchainEmbeddingsWrapper")
    @patch("benchmark.evaluation.get_embedding_model")
    @patch("benchmark.evaluation.LangchainLLMWrapper")
    @patch("benchmark.evaluation.get_chat_model")
    @patch("benchmark.evaluation.evaluate")
    def test_fallback_to_generator_model(self, mock_evaluate, mock_get_chat, mock_llm_wrap, mock_get_emb, mock_emb_wrap):
        """When critic_llm_model is None, it should fall back to llm_model."""
        mock_get_chat.return_value = MagicMock()
        mock_llm_wrap.return_value = MagicMock()
        mock_get_emb.return_value = MagicMock()
        mock_emb_wrap.return_value = MagicMock()

        mock_result = MagicMock()
        mock_result.scores = [{"faithfulness": 0.9}]
        mock_evaluate.return_value = mock_result

        evaluate_results(
            questions=["q1"],
            ground_truths=["gt1"],
            answers=["a1"],
            contexts=[["ctx1"]],
            llm_model="openai:base-model",
            critic_llm_model=None,
            openai_compat_base_url="https://api.example.com/v1",
        )

        call_kwargs = mock_get_chat.call_args[1]
        assert call_kwargs["provider"] == "openai"
        assert call_kwargs["model_name"] == "base-model"


class TestEvaluateResultsScoreParsing:
    @patch("benchmark.evaluation.LangchainEmbeddingsWrapper")
    @patch("benchmark.evaluation.get_embedding_model")
    @patch("benchmark.evaluation.LangchainLLMWrapper")
    @patch("benchmark.evaluation.get_chat_model")
    @patch("benchmark.evaluation.evaluate")
    def test_nan_scores_filtered(self, mock_evaluate, mock_get_chat, mock_llm_wrap, mock_get_emb, mock_emb_wrap):
        mock_get_chat.return_value = MagicMock()
        mock_llm_wrap.return_value = MagicMock()
        mock_get_emb.return_value = MagicMock()
        mock_emb_wrap.return_value = MagicMock()

        mock_result = MagicMock()
        mock_result.scores = [
            {"faithfulness": 0.9, "answer_relevancy": math.nan},
            {"faithfulness": 0.7, "answer_relevancy": 0.8},
        ]
        mock_evaluate.return_value = mock_result

        result = evaluate_results(
            questions=["q1", "q2"],
            ground_truths=["gt1", "gt2"],
            answers=["a1", "a2"],
            contexts=[["c1"], ["c2"]],
        )

        assert result.error is None
        assert result.metric_means["faithfulness"] == pytest.approx(0.8)
        assert result.metric_means["answer_relevancy"] == pytest.approx(0.8)
        assert result.per_sample_scores[0]["answer_relevancy"] is None
        assert result.per_sample_scores[1]["answer_relevancy"] == 0.8

    @patch("benchmark.evaluation.LangchainEmbeddingsWrapper")
    @patch("benchmark.evaluation.get_embedding_model")
    @patch("benchmark.evaluation.LangchainLLMWrapper")
    @patch("benchmark.evaluation.get_chat_model")
    @patch("benchmark.evaluation.evaluate")
    def test_valid_sample_counts(self, mock_evaluate, mock_get_chat, mock_llm_wrap, mock_get_emb, mock_emb_wrap):
        mock_get_chat.return_value = MagicMock()
        mock_llm_wrap.return_value = MagicMock()
        mock_get_emb.return_value = MagicMock()
        mock_emb_wrap.return_value = MagicMock()

        mock_result = MagicMock()
        mock_result.scores = [
            {"faithfulness": 0.9, "answer_relevancy": 0.8},
            {"faithfulness": 0.7, "answer_relevancy": None},
        ]
        mock_evaluate.return_value = mock_result

        result = evaluate_results(
            questions=["q1", "q2"],
            ground_truths=["gt1", "gt2"],
            answers=["a1", "a2"],
            contexts=[["c1"], ["c2"]],
        )

        assert result.samples_with_valid_scores["faithfulness"] == 2
        assert result.samples_with_valid_scores["answer_relevancy"] == 1
