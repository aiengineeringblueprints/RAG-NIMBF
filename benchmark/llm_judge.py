"""LLM-as-judge answer quality scoring with an order-bias control.

Scores each (question, answer, ground truth) on a 0-1 rubric using the
configured critic LLM — independent of RAGAS. Each sample is judged twice
with the rubric criteria in opposite orders; the mean score is reported as
``llm_judge_score`` and the absolute difference as ``llm_judge_order_bias``
(a judge-robustness diagnostic: high values mean the verdict depends on
prompt wording rather than answer quality).

Pure post-processing over stored answers; follows the same provider plumbing
as ``benchmark/evaluation.py`` and returns a ``CustomMetricsResult`` so it
merges into the existing reporting path unchanged.
"""

from __future__ import annotations

import json
import logging
import re

from benchmark.custom_metrics import CustomMetricsResult
from benchmark.providers import get_chat_model, parse_model_id

logger = logging.getLogger(__name__)

_RUBRIC = [
    "accuracy: the answer is factually consistent with the ground truth",
    "completeness: the answer addresses every part of the question",
    "relevance: the answer does not drift from what was asked",
    "honesty: if uncertain, the answer does not invent unsupported facts",
]

_PROMPT_TEMPLATE = """You are judging the quality of a question-answering system.

Question: {question}
Ground truth: {ground_truth}
System answer: {answer}

Rate the system answer on these criteria:
{criteria}

First give a total score between 0 and 10 (0 = unusable, 10 = perfect).
Respond with ONLY a JSON object: {{"score": <number>}}
"""


def _parse_score(text: str) -> float | None:
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "score" in payload:
            return float(payload["score"])
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    match = re.search(r'"?score"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', text)
    if match:
        return float(match.group(1))
    return None


def _normalize(score: float) -> float:
    return max(0.0, min(1.0, score / 10.0))


def _format_criteria(rubric: list[str]) -> str:
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(rubric))


def judge_answers(
    questions: list[str],
    ground_truths: list[str],
    answers: list[str],
    *,
    critic_llm_model: str,
    ollama_base_url: str = "http://localhost:11434",
    ollama_api_key: str | None = None,
    openai_compat_base_url: str | None = None,
    openai_compat_api_key: str | None = None,
    critic_max_tokens: int = 2000,
) -> CustomMetricsResult:
    """Judge every answer; returns ``llm_judge_score`` + ``llm_judge_order_bias``."""
    provider, model_name = parse_model_id(critic_llm_model)
    if provider == "openai":
        base = openai_compat_base_url or ""
        key = openai_compat_api_key
    else:
        base = ollama_base_url
        key = ollama_api_key
    try:
        judge = get_chat_model(
            provider=provider,
            model_name=model_name,
            base_url=base,
            api_key=key,
            max_tokens=critic_max_tokens,
            temperature=0.0,
        )
    except (RuntimeError, ValueError) as exc:
        return CustomMetricsResult(
            metric_means={},
            per_sample=[],
            samples_with_valid_scores={},
            error=f"Failed to initialise judge model: {exc}",
        )

    per_sample: list[dict[str, float | None]] = []
    scores: list[float] = []
    biases: list[float] = []
    failures = 0
    reversed_rubric = list(reversed(_RUBRIC))

    for question, ground_truth, answer in zip(questions, ground_truths, answers):
        sample_scores: list[float] = []
        for rubric in (_RUBRIC, reversed_rubric):
            prompt = _PROMPT_TEMPLATE.format(
                question=question,
                ground_truth=ground_truth,
                answer=answer,
                criteria=_format_criteria(rubric),
            )
            try:
                response = judge.invoke(prompt)
                text = getattr(response, "content", "") or str(response)
                raw = _parse_score(str(text))
            except Exception as exc:  # judge failure must not kill the run
                logger.warning("LLM judge failed on sample: %s", exc)
                raw = None
            if raw is not None:
                sample_scores.append(_normalize(raw))
        if not sample_scores:
            per_sample.append({"llm_judge_score": None, "llm_judge_order_bias": None})
            failures += 1
            continue
        score = sum(sample_scores) / len(sample_scores)
        bias = (
            abs(sample_scores[0] - sample_scores[1])
            if len(sample_scores) == 2
            else None
        )
        scores.append(score)
        if bias is not None:
            biases.append(bias)
        per_sample.append(
            {
                "llm_judge_score": score,
                "llm_judge_order_bias": bias,
            }
        )

    if failures == len(questions):
        return CustomMetricsResult(
            metric_means={},
            per_sample=per_sample,
            samples_with_valid_scores={},
            error="LLM judge failed on every sample",
        )

    metric_means: dict[str, float] = {}
    valid_counts: dict[str, int] = {}
    if scores:
        metric_means["llm_judge_score"] = sum(scores) / len(scores)
        valid_counts["llm_judge_score"] = len(scores)
    if biases:
        metric_means["llm_judge_order_bias"] = sum(biases) / len(biases)
        valid_counts["llm_judge_order_bias"] = len(biases)
    error = (
        f"LLM judge failed on {failures} sample(s)" if failures else None
    )
    return CustomMetricsResult(
        metric_means=metric_means,
        per_sample=per_sample,
        samples_with_valid_scores=valid_counts,
        error=error,
    )
