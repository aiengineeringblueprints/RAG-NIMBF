"""LLM-as-judge hallucination detector.

Uses the existing ``benchmark.providers`` factory to create an LLM client.
Prompts the LLM to rate hallucination on a 1-5 scale, normalized to 0-1.
"""
from __future__ import annotations

import logging
import re

from hallucination_eval.detectors import (
    HallucinationDetector,
    HallucinationSample,
    DetectorOutput,
    register_detector,
)

logger = logging.getLogger(__name__)

HALLUCINATION_JUDGE_PROMPT = """\
You are an expert evaluator for detecting hallucinations in RAG systems.

Given:
- CONTEXT: Retrieved documents
- RESPONSE: Generated answer

Rate the faithfulness of the RESPONSE to the CONTEXT on a scale of 1-5:
1 = Completely hallucinated (major claims not supported by context)
2 = Mostly hallucinated (significant unsupported claims)
3 = Partially faithful (mix of supported and unsupported claims)
4 = Mostly faithful (minor unsupported details)
5 = Completely faithful (all claims supported by context)

CONTEXT:
{context}

RESPONSE:
{response}

Rate (1-5):"""


@register_detector
class LLMJudgeDetector(HallucinationDetector):
    name = "llm_judge"
    description = "LLM-as-judge hallucination evaluation"
    requires_gpu = False
    requires_llm = True

    def __init__(
        self,
        provider: str,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        max_tokens: int = 64,
    ):
        from benchmark.providers import get_chat_model

        self.llm = get_chat_model(
            provider=provider,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=0.0,
        )

    def score_single(self, context: str, response: str, **kw) -> float:
        prompt = HALLUCINATION_JUDGE_PROMPT.format(context=context, response=response)
        result = self.llm.invoke(prompt)
        rating = self._parse_rating(result.content)
        return (rating - 1) / 4.0  # normalize 1-5 to 0-1

    @staticmethod
    def _parse_rating(text: str) -> int:
        """Extract 1-5 rating from LLM response."""
        match = re.search(r"\b([1-5])\b", text.strip())
        if match:
            return int(match.group(1))
        return 3  # default to middle

    def score_samples(self, samples: list[HallucinationSample]) -> list[DetectorOutput]:
        results: list[DetectorOutput] = []
        for i, sample in enumerate(samples):
            context = "\n\n".join(sample.contexts)
            try:
                score = self.score_single(context, sample.answer)
                results.append(DetectorOutput(sample_index=i, score=score))
            except Exception as exc:
                logger.warning("LLM judge failed for sample %d: %s", i, exc)
                results.append(
                    DetectorOutput(sample_index=i, score=float("nan"), error=str(exc))
                )
        return results
