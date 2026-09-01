"""Robustness / capability probes for RAG generators.

These tests come from the RGB / RECALL / NoMIRACL line of work (see the
RAGChecker survey) and probe how the generator behaves on *adversarial*
or *degraded* inputs rather than natural ones:

    * Negative rejection  — when the retrieved context cannot answer the
      question, does the model refuse (or hallucinate)?
    * Noise robustness    — when distractor chunks are injected into the
      context, how much does generation quality degrade?

The module is functional and dependency-light.  ``measure_negative_rejection``
is a pure computation over already-generated answers; the distractor builder
and degradation helper just transform data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from benchmark.custom_metrics import is_refusal_answer


# ── Result container ─────────────────────────────────────────────────

@dataclass(frozen=True)
class NegativeRejectionResult:
    """Aggregated negative-rejection behaviour on unanswerable questions.

    ``unanswerable_count``  – number of samples flagged unanswerable.
    ``rejection_rate``      – fraction of unanswerable samples where the
                              model refused rather than answering.
    ``hallucination_rate``  – fraction of unanswerable samples where the
                              model produced a non-refusal answer.
    ``per_sample``          – per-sample flags (``refused`` / ``non_refusal``).
    """

    unanswerable_count: int
    rejection_rate: float | None
    hallucination_rate: float | None
    per_sample: list[dict[str, float | None]] = field(default_factory=list)
    error: str | None = None


def measure_negative_rejection(
    questions: list[str],
    answers: list[str],
    unanswerable: list[bool],
) -> NegativeRejectionResult:
    """Measure how often the model rejects unanswerable questions.

    ``unanswerable`` is a per-sample flag (from dataset annotation or the
    dynamic ground-truth synthesizer) indicating the context genuinely
    cannot support an answer.
    """
    if len(questions) != len(answers) or len(questions) != len(unanswerable):
        return NegativeRejectionResult(
            unanswerable_count=0,
            rejection_rate=None,
            hallucination_rate=None,
            error="questions, answers and unanswerable must be equal length.",
        )

    per_sample: list[dict[str, float | None]] = []
    unans_idx = [i for i, u in enumerate(unanswerable) if u]
    rejected = 0

    for i, (q, a) in enumerate(zip(questions, answers)):
        refused = is_refusal_answer(a)
        if unanswerable[i]:
            if refused:
                rejected += 1
        per_sample.append({"refused": 1.0 if refused else 0.0})

    n = len(unans_idx)
    rejection_rate = rejected / n if n else None
    hallucination_rate = (n - rejected) / n if n else None

    return NegativeRejectionResult(
        unanswerable_count=n,
        rejection_rate=rejection_rate,
        hallucination_rate=hallucination_rate,
        per_sample=per_sample,
    )


# ── Distractor injection (noise robustness) ──────────────────────────

def add_distractor_contexts(
    contexts: list[list[str]],
    distractors: Sequence[str],
    n_distractors: int = 1,
    seed: int | None = None,
) -> list[list[str]]:
    """Return context lists with *n_distractors* extra chunks appended.

    Distractors are inserted at the end (i.e. lowest retrieval priority)
    so the original retrieval order is preserved.  Used to build the
    "noisy" arm of a noise-robustness A/B comparison.
    """
    import random

    rng = random.Random(seed)
    out: list[list[str]] = []
    for ctx in contexts:
        extra = list(distractors)
        if len(extra) > n_distractors:
            rng.shuffle(extra)
            extra = extra[:n_distractors]
        out.append(list(ctx) + extra)
    return out


def measure_noise_robustness(
    scores_clean: list[float],
    scores_noisy: list[float],
) -> dict[str, float | None]:
    """Summarise quality degradation between clean and noisy contexts.

    ``scores_clean`` / ``scores_noisy`` are per-sample scores for the same
    questions (e.g. faithfulness or claim-F1) computed with and without
    injected distractors.
    """
    if len(scores_clean) != len(scores_noisy):
        return {"robustness_delta": None, "robustness_ratio": None}
    if not scores_clean:
        return {"robustness_delta": None, "robustness_ratio": None}

    clean = sum(scores_clean) / len(scores_clean)
    noisy = sum(scores_noisy) / len(scores_noisy)
    return {
        "robustness_delta": clean - noisy,
        "robustness_ratio": (noisy / clean) if clean > 0 else None,
    }


# ── Compositional noise robustness over a score function ─────────────

def measure_noise_robustness_fn(
    questions: list[str],
    contexts: list[list[str]],
    answers: list[str],
    distractors: Sequence[str],
    score_fn: Callable[[list[str], list[list[str]], list[str]], list[float]],
    n_distractors: int = 1,
    seed: int | None = None,
) -> dict[str, float | None]:
    """Inject distractors, re-score, and return the degradation summary.

    ``score_fn(questions, contexts, answers) -> list[float]`` is any
    per-sample scoring function (e.g. a wrapper around RAGChecker
    faithfulness or RAGAS semantic similarity).
    """
    clean = score_fn(questions, contexts, answers)
    noisy_ctx = add_distractor_contexts(contexts, distractors, n_distractors, seed)
    noisy = score_fn(questions, noisy_ctx, answers)
    return measure_noise_robustness(clean, noisy)
