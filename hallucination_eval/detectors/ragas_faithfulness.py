"""RAGAS Faithfulness detector — reads pre-computed scores from benchmark JSONs.

Zero compute, zero API calls. Serves as the baseline comparator.
"""
from __future__ import annotations

import math

from hallucination_eval.detectors import (
    HallucinationDetector,
    HallucinationSample,
    DetectorOutput,
    register_detector,
)


@register_detector
class RAGASFaithfulnessDetector(HallucinationDetector):
    name = "ragas_faithfulness"
    description = "RAGAS Faithfulness (from existing benchmark results)"
    requires_gpu = False
    requires_llm = False

    def score_samples(self, samples: list[HallucinationSample]) -> list[DetectorOutput]:
        return [
            DetectorOutput(
                sample_index=i,
                score=(
                    s.ragas_faithfulness
                    if s.ragas_faithfulness is not None
                    and not math.isnan(s.ragas_faithfulness)
                    else float("nan")
                ),
            )
            for i, s in enumerate(samples)
        ]

    def score_single(self, context: str, response: str, **kw) -> float:
        raise NotImplementedError(
            "RAGAS Faithfulness requires pre-computed scores from benchmark data"
        )
