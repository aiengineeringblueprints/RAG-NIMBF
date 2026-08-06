"""Inference-time evaluator that wraps a trained :class:`RobertaTraceClassifier`.

The public class, :class:`RobertaTraceEvaluator`, is a drop-in
replacement for the Ragas LLM-judge call used elsewhere in the framework:
it takes parallel lists of ``questions``, ``answers`` and ``contexts``
and returns a :class:`RobertaTraceEvalResult` whose schema mirrors
:class:`benchmark.evaluation.EvaluationResult` (per-sample scores +
aggregate means + error string).

Key properties
--------------
* CPU-friendly — no GPU required at inference time.
* Lazy model loading — the HF stack is only imported when the evaluator
  is actually called.
* Clear error if no trained checkpoint exists, pointing at the training
  instructions (see ``NOTES_I_RoBERTa.md``).
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .model import (
    ADHERENCE_NUM_CLASSES,
    DEFAULT_NUM_BINS,
    TRACE_METRICS,
    RobertaTraceClassifier,
    dequantize_bin,
)

logger = logging.getLogger(__name__)

# Hard cap on RoBERTa input length.  ``roberta-base`` supports up to 512
# position embeddings; longer inputs are truncated.
MAX_INPUT_LENGTH = 512

# Default checkpoint lookup locations.  If the user passes neither an
# explicit ``model_path`` nor ``model_hub_id``, we check these dirs in
# order before failing.
DEFAULT_LOCAL_PATHS: tuple[str, ...] = (
    "models/roberta_trace",
    os.path.expanduser("~/.cache/roberta_trace"),
)


@dataclass(frozen=True)
class RobertaTraceEvalResult:
    """Schema-compatible mirror of :class:`benchmark.evaluation.EvaluationResult`.

    ``metric_means`` keys are the four TRACe metrics suffixed with the
    same naming convention used by Ragas (no overlap with Ragas metric
    names, so ``evaluator: both`` can merge dicts without collision).
    """

    metric_means: dict[str, float]
    per_sample_scores: list[dict[str, float | None]]
    error: str | None = None
    samples_with_valid_scores: dict[str, int] = field(default_factory=dict)


def _format_input(question: str, context: str, response: str) -> str:
    """Build the Roberta input string.

    Format: ``<s> question </s> context </s> response </s>``.  Special
    tokens are added by the tokenizer (we don't paste them in raw), so
    here we just join the three fields with the Roberta separator.
    """
    q = (question or "").strip()
    c = (context or "").strip()
    r = (response or "").strip()
    # RoBERTa uses ``</s>`` as the separator between segments.
    return f"{q}</s></s>{c}</s></s>{r}"


def _flatten_contexts(contexts: Sequence[str] | Sequence[Sequence[str]] | None) -> str:
    """Accept either a single list of strings or a list of lists."""
    if contexts is None:
        return ""
    out: list[str] = []
    for item in contexts:
        if isinstance(item, (list, tuple)):
            out.extend(str(s) for s in item if s)
        else:
            if item:
                out.append(str(item))
    return "\n".join(out).strip()


class RobertaTraceEvaluator:
    """Drop-in evaluator that scores samples via a trained RoBERTa model.

    Parameters
    ----------
    model_path
        Local directory containing a trained checkpoint (output of
        ``train.py``).
    model_hub_id
        HF Hub id (e.g. ``"your-org/roberta-trace-ragbench"``) to download
        the checkpoint from.  Mutually exclusive with ``model_path``; if
        both are set, ``model_path`` wins.
    num_bins
        Override the bin count read from the checkpoint.  Almost never
        needed; included for debugging.
    batch_size
        Inference batch size (default 8).  Lower for memory-constrained
        CPUs.
    device
        ``"cpu"`` by default.  Pass ``"cuda"`` / ``"cuda:0"`` etc. for GPU.
    max_input_length
        Tokenizer max length (default 512 = RoBERTa cap).
    """

    def __init__(
        self,
        model_path: str | None = None,
        model_hub_id: str | None = None,
        num_bins: int | None = None,
        batch_size: int = 8,
        device: str = "cpu",
        max_input_length: int = MAX_INPUT_LENGTH,
    ) -> None:
        self.model_path = model_path
        self.model_hub_id = model_hub_id
        self.num_bins = num_bins
        self.batch_size = batch_size
        self.device = device
        self.max_input_length = max_input_length
        self._classifier: RobertaTraceClassifier | None = None
        self._tokenizer = None  # populated on first call

    # ── model resolution ────────────────────────────────────────────
    def _resolve_checkpoint(self) -> str:
        """Find the trained checkpoint or raise an actionable error."""
        if self.model_path and os.path.exists(self.model_path):
            return self.model_path
        if self.model_hub_id:
            return self.model_hub_id
        for candidate in DEFAULT_LOCAL_PATHS:
            if os.path.exists(candidate) and os.path.exists(
                os.path.join(candidate, "trace_heads.pt")
            ):
                return candidate
        raise FileNotFoundError(
            "No trained RoBERTa-TRACe checkpoint found. Either set "
            "ROBERTA_TRACE_MODEL_PATH / ROBERTA_TRACE_MODEL_HUB_ID, "
            "pass model_path=, or train one with:\n"
            "    python -m benchmark.roberta_evaluator.train "
            "--output_dir models/roberta_trace\n"
            "See NOTES_I_RoBERTa.md for the full recipe."
        )

    def _ensure_loaded(self) -> None:
        if self._classifier is not None:
            return
        checkpoint = self._resolve_checkpoint()
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "transformers is required for RoBERTa evaluation. "
                "Install with `pip install transformers>=4.30`."
            ) from exc

        if self.num_bins is not None:
            # Build classifier directly (path is a raw encoder dir, no heads).
            self._classifier = RobertaTraceClassifier(
                model_name=checkpoint, num_bins=self.num_bins
            )
            self._classifier.build()
        else:
            # Load full multi-task checkpoint.
            self._classifier = RobertaTraceClassifier.from_pretrained(checkpoint)

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        except Exception:  # pragma: no cover - HF error paths vary
            # Fall back to the encoder backbone's tokenizer.
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._classifier.model_name
            )

        try:
            import torch  # type: ignore[import-not-found]

            self._classifier.build().to(self.device)
            self._torch = torch
        except ImportError:  # pragma: no cover - torch checked at build()
            raise ImportError(
                "torch is required for RoBERTa evaluation. "
                "Install with `pip install torch>=2.0`."
            )

    # ── scoring ─────────────────────────────────────────────────────
    def score_batch(
        self,
        questions: Sequence[str],
        contexts: Sequence[Sequence[str] | Sequence[Sequence[str]] | str],
        responses: Sequence[str],
    ) -> list[dict[str, float | None]]:
        """Score a batch of (question, context, response) triples.

        Returns a list of dicts (one per input row) with the four TRACe
        scores.  Rows that fail tokenisation or inference get all-None
        scores so the caller can still align them with the input.
        """
        if not (len(questions) == len(contexts) == len(responses)):
            raise ValueError(
                "questions, contexts and responses must have equal length"
            )
        if not questions:
            return []

        self._ensure_loaded()
        assert self._classifier is not None and self._tokenizer is not None
        torch = self._torch

        per_sample: list[dict[str, float | None]] = []
        for start in range(0, len(questions), self.batch_size):
            chunk_q = list(questions[start : start + self.batch_size])
            chunk_c = list(contexts[start : start + self.batch_size])
            chunk_r = list(responses[start : start + self.batch_size])

            formatted = [
                _format_input(q, _flatten_contexts([c]), r)
                for q, c, r in zip(chunk_q, chunk_c, chunk_r)
            ]
            tokens = self._tokenizer(
                formatted,
                padding=True,
                truncation=True,
                max_length=self.max_input_length,
                return_tensors="pt",
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            try:
                with torch.no_grad():
                    logits = self._classifier.predict_logits(
                        input_ids=tokens["input_ids"],
                        attention_mask=tokens.get("attention_mask"),
                    )
            except RuntimeError as exc:  # pragma: no cover - OOM etc.
                logger.warning("RoBERTa inference failed for batch @%d: %s", start, exc)
                for _ in chunk_q:
                    per_sample.append({m: None for m in TRACE_METRICS})
                continue

            # Argmax per head.  Score heads are dequantised back to [0,1];
            # adherence is reported as P(true) ∈ [0, 1] (softmax of the
            # positive class).
            for i in range(len(chunk_q)):
                sample: dict[str, float | None] = {}
                for metric in TRACE_METRICS:
                    head_logits = logits[metric][i]
                    if metric == "adherence":
                        probs = torch.softmax(head_logits, dim=-1)
                        # Positive class = index 1 (label True).
                        pos = min(ADHERENCE_NUM_CLASSES - 1, 1)
                        sample[metric] = float(probs[pos].item())
                    else:
                        bin_idx = int(torch.argmax(head_logits).item())
                        sample[metric] = dequantize_bin(bin_idx, DEFAULT_NUM_BINS)
                per_sample.append(sample)
        return per_sample

    # ── framework-friendly API ──────────────────────────────────────
    def evaluate(
        self,
        questions: Sequence[str],
        ground_truths: Sequence[str],
        answers: Sequence[str],
        contexts: Sequence[Sequence[str]],
        **_unused,
    ) -> RobertaTraceEvalResult:
        """Mirror of :func:`benchmark.evaluation.evaluate_results`.

        ``ground_truths`` is accepted for API symmetry with Ragas but is
        not used: TRACe is reference-free at inference time (the response
        is scored only against the retrieved context).
        """
        if not questions:
            return RobertaTraceEvalResult(
                metric_means={},
                per_sample_scores=[],
                error="No questions provided for RoBERTa evaluation.",
                samples_with_valid_scores={},
            )
        try:
            self._ensure_loaded()
        except (ImportError, FileNotFoundError) as exc:
            return RobertaTraceEvalResult(
                metric_means={},
                per_sample_scores=[{} for _ in questions],
                error=str(exc),
                samples_with_valid_scores={},
            )

        per_sample = self.score_batch(questions, contexts, answers)

        accum: dict[str, list[float]] = {m: [] for m in TRACE_METRICS}
        for sample in per_sample:
            for metric in TRACE_METRICS:
                v = sample.get(metric)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    accum[metric].append(float(v))

        means = {k: sum(v) / len(v) for k, v in accum.items() if v}
        valid = {k: len(v) for k, v in accum.items() if v}
        return RobertaTraceEvalResult(
            metric_means=means,
            per_sample_scores=per_sample,
            error=None,
            samples_with_valid_scores=valid,
        )


def evaluate_with_roberta(
    questions: Sequence[str],
    ground_truths: Sequence[str],
    answers: Sequence[str],
    contexts: Sequence[Sequence[str]],
    *,
    model_path: str | None = None,
    model_hub_id: str | None = None,
    device: str = "cpu",
    **kwargs,
) -> RobertaTraceEvalResult:
    """Functional entry point matching ``benchmark.evaluation.evaluate_results``.

    Resolves the checkpoint from ``ROBERTA_TRACE_MODEL_PATH`` /
    ``ROBERTA_TRACE_MODEL_HUB_ID`` env vars when neither ``model_path``
    nor ``model_hub_id`` is provided.
    """
    resolved_path = model_path or os.environ.get("ROBERTA_TRACE_MODEL_PATH")
    resolved_hub = model_hub_id or os.environ.get("ROBERTA_TRACE_MODEL_HUB_ID")
    evaluator = RobertaTraceEvaluator(
        model_path=resolved_path,
        model_hub_id=resolved_hub,
        device=device,
    )
    return evaluator.evaluate(questions, ground_truths, answers, contexts, **kwargs)
