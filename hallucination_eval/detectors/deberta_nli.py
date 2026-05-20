"""DeBERTa-v3 NLI hallucination detector — cross-encoder, no LLM needed.

Uses ``cross-encoder/nli-deberta-v3-base`` from HuggingFace.
Classifies (premise, hypothesis) as entailment/neutral/contradiction.
Returns entailment probability as faithfulness score.
"""
from __future__ import annotations

import logging
import math

from hallucination_eval.detectors import (
    HallucinationDetector,
    HallucinationSample,
    DetectorOutput,
    register_detector,
)

logger = logging.getLogger(__name__)

# Label indices: 0=contradiction, 1=neutral, 2=entailment
ENTAILMENT_IDX = 2


@register_detector
class DeBERTaNLIDetector(HallucinationDetector):
    name = "deberta_nli"
    description = "DeBERTa-v3 NLI cross-encoder (no LLM needed)"
    requires_gpu = False
    requires_llm = False

    def __init__(self, device: str = "cpu", batch_size: int = 16):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.batch_size = batch_size
        logger.info("Loading DeBERTa NLI model on %s...", device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "cross-encoder/nli-deberta-v3-base"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "cross-encoder/nli-deberta-v3-base"
        )
        self.model.to(device)
        self.model.eval()
        logger.info("DeBERTa NLI model loaded.")

    def score_single(self, context: str, response: str, **kw) -> float:
        import torch

        inputs = self.tokenizer(
            context,
            response,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[0][ENTAILMENT_IDX].item()

    def score_samples(self, samples: list[HallucinationSample]) -> list[DetectorOutput]:
        results: list[DetectorOutput] = []
        for i, sample in enumerate(samples):
            context = "\n\n".join(sample.contexts)
            try:
                score = self.score_single(context, sample.answer)
                results.append(DetectorOutput(sample_index=i, score=score))
            except Exception as exc:
                logger.warning("DeBERTa NLI failed for sample %d: %s", i, exc)
                results.append(
                    DetectorOutput(sample_index=i, score=float("nan"), error=str(exc))
                )

        return results
