"""Hallucination detector base class and registry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class HallucinationSample:
    """One evaluated Q&A pair with all available annotations."""

    run_name: str
    dataset_name: str
    config_name: str
    llm_model: str
    question: str
    ground_truth: str
    answer: str
    contexts: list[str]
    ragas_faithfulness: float | None = None
    ragas_context_recall: float | None = None
    ragas_semantic_similarity: float | None = None
    custom_scores: dict[str, float | None] = field(default_factory=dict)


@dataclass
class DetectorOutput:
    """Per-sample hallucination detection result."""

    sample_index: int
    score: float  # 0 = fully hallucinated, 1 = fully faithful
    raw_output: dict | None = None
    error: str | None = None


class HallucinationDetector(ABC):
    """Base class for hallucination detection methods."""

    name: str = ""
    description: str = ""
    requires_gpu: bool = False
    requires_llm: bool = False

    @abstractmethod
    def score_samples(
        self,
        samples: list[HallucinationSample],
    ) -> list[DetectorOutput]:
        """Score all samples. Returns one DetectorOutput per sample."""
        ...

    @abstractmethod
    def score_single(
        self,
        context: str,
        response: str,
        question: str = "",
        ground_truth: str = "",
    ) -> float:
        """Score a single (context, response) pair. Returns 0-1."""
        ...


DETECTOR_REGISTRY: dict[str, type[HallucinationDetector]] = {}


def register_detector(cls: type[HallucinationDetector]) -> type[HallucinationDetector]:
    """Register a detector class by its ``name`` attribute."""
    DETECTOR_REGISTRY[cls.name] = cls
    return cls
