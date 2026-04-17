from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from benchmark.tracing import observe

from ragas import evaluate, EvaluationDataset, SingleTurnSample, RunConfig
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._answer_correctness import answer_correctness
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from benchmark.providers import parse_model_id, get_chat_model, wrap_for_ragas
from benchmark.embedding import get_embedding_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    metric_means: dict[str, float]
    per_sample_scores: list[dict[str, float | None]]
    error: str | None = None
    samples_with_valid_scores: dict[str, int] = field(default_factory=dict)


@observe(name="ragas_evaluation")
def evaluate_results(
    questions: list[str],
    ground_truths: list[str],
    answers: list[str],
    contexts: list[list[str]],
    llm_model: str = "gemma3:4b",
    embedding_model: str = "nomic-embed-text:latest",
    critic_llm_model: str | None = None,
    critic_embedding_model: str | None = None,
    ollama_base_url: str = "http://localhost:11434",
    ollama_api_key: str | None = None,
    openai_compat_base_url: str | None = None,
    openai_compat_api_key: str | None = None,
    critic_ollama_base_url: str | None = None,
    critic_ollama_api_key: str | None = None,
    critic_openai_compat_base_url: str | None = None,
    critic_openai_compat_api_key: str | None = None,
    critic_max_tokens: int = 10000,
) -> EvaluationResult:
    if not questions:
        return EvaluationResult(
            metric_means={},
            per_sample_scores=[],
            error="No questions provided for evaluation.",
            samples_with_valid_scores={},
        )

    # Fall back to the generator model when critic models are not specified.
    effective_critic_llm = critic_llm_model or llm_model
    effective_critic_embedding = critic_embedding_model or embedding_model

    samples = []
    for q, gt, a, ctx in zip(questions, ground_truths, answers, contexts):
        samples.append(
            SingleTurnSample(
                user_input=q,
                response=a,
                reference=gt,
                retrieved_contexts=ctx,
            )
        )

    eval_dataset = EvaluationDataset(samples=samples)

    try:
        # Resolve critic LLM via provider factory (use critic-specific URLs)
        critic_provider, critic_model_name = parse_model_id(effective_critic_llm)
        if critic_provider == "openai":
            critic_base = critic_openai_compat_base_url or openai_compat_base_url or ""
            critic_key = critic_openai_compat_api_key or openai_compat_api_key
        else:
            critic_base = critic_ollama_base_url or ollama_base_url
            critic_key = critic_ollama_api_key or ollama_api_key

        critic_chat = get_chat_model(
            provider=critic_provider,
            model_name=critic_model_name,
            base_url=critic_base,
            api_key=critic_key,
            max_tokens=critic_max_tokens,
            temperature=0.0,
        )
        critic_llm = LangchainLLMWrapper(wrap_for_ragas(critic_chat))

        # Resolve critic embeddings via factory
        emb_provider, emb_model_name = parse_model_id(effective_critic_embedding)
        critic_emb = get_embedding_model(
            emb_model_name,
            ollama_base_url,
            ollama_api_key,
            provider=emb_provider,
        )
        critic_embeddings = LangchainEmbeddingsWrapper(critic_emb)
    except (ConnectionError, TimeoutError) as exc:
        error_msg = f"Failed to connect to provider: {exc}"
        logger.error(error_msg)
        return EvaluationResult(
            metric_means={},
            per_sample_scores=[],
            error=error_msg,
            samples_with_valid_scores={},
        )
    except (RuntimeError, ValueError) as exc:
        error_msg = f"Failed to initialise critic models: {exc}"
        logger.error(error_msg)
        return EvaluationResult(
            metric_means={},
            per_sample_scores=[],
            error=error_msg,
            samples_with_valid_scores={},
        )

    # metrics = [faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall]
    metrics = [faithfulness]

    # max_workers=1: local models process requests serially anyway.
    # More workers just cause queuing and timeouts. Raise timeout instead.
    run_config = RunConfig(timeout=600, max_retries=2, max_wait=60, max_workers=1)

    try:
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=critic_llm,
            embeddings=critic_embeddings,
            run_config=run_config,
        )
    except (ConnectionError, TimeoutError) as exc:
        error_msg = f"Network error during RAGAS evaluation: {exc}"
        logger.error(error_msg)
        return EvaluationResult(
            metric_means={},
            per_sample_scores=[],
            error=error_msg,
            samples_with_valid_scores={},
        )
    except (RuntimeError, ValueError) as exc:
        error_msg = f"Evaluation failed: {exc}"
        logger.error(error_msg)
        return EvaluationResult(
            metric_means={},
            per_sample_scores=[],
            error=error_msg,
            samples_with_valid_scores={},
        )

    per_sample_scores: list[dict[str, float | None]] = []
    scores_accum: dict[str, list[float]] = {}

    for score_dict in result.scores:
        sample_scores: dict[str, float | None] = {}
        for k, v in score_dict.items():
            if isinstance(v, (int, float)) and not math.isnan(v):
                clean_v = float(v)
                sample_scores[k] = clean_v
                if k not in scores_accum:
                    scores_accum[k] = []
                scores_accum[k].append(clean_v)
            else:
                sample_scores[k] = None
        per_sample_scores.append(sample_scores)

    means = {k: sum(v) / len(v) for k, v in scores_accum.items() if v}
    valid_counts = {k: len(v) for k, v in scores_accum.items() if v}

    return EvaluationResult(
        metric_means=means,
        per_sample_scores=per_sample_scores,
        error=None,
        samples_with_valid_scores=valid_counts,
    )
