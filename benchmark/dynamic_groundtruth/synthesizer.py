"""Top-level orchestrator for dynamic ground-truth synthesis.

Pipeline (RAGPerf §3.2):

    chunk
      └─ masker.select_mask                       → MaskResult
      └─ distilbert_generator.fill_mask            → Candidate
      └─ t5_question_generator.generate_question   → QuestionResult
      └─ quality gates + dedup                     → UpdateWithGroundTruth

The synthesizer is intentionally side-effect-free apart from model
loading.  ``synthesize_batch`` is the typical entry point — it accepts a
list of chunk dicts and returns a list of :class:`UpdateWithGroundTruth`.

Integration with the workload generator (separate worktree)::

    from benchmark.dynamic_groundtruth import DynamicGroundTruthSynthesizer

    synth = DynamicGroundTruthSynthesizer()
    updates = synth.synthesize_batch(chunks)
    WorkloadGenerator.update_question_pool(updates)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

from .masker import MASK_TOKEN, MaskError, MaskResult, select_mask

logger = logging.getLogger(__name__)


# ── Quality thresholds (overridable per call) ─────────────────────────

MIN_CHUNK_CHARS = 50          # spec safeguard: skip too-short chunks
DEFAULT_DEVICE = -1           # CPU by default; set 0 for first GPU


# ── Dataclasses ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class UpdateWithGroundTruth:
    """Output record for one synthesised update.

    Maps 1:1 to the JSONL schema in the spec::

        {"chunk_id": "...", "original": "...", "updated": "...",
         "masked_token_original": "1995", "masked_token_new": "2003",
         "question": "When was X founded?", "answer": "2003"}
    """

    chunk_id: str
    original: str
    updated: str
    masked_token_original: str
    masked_token_new: str
    question: str
    answer: str
    mask_kind: str = ""
    spacy_used: bool = False
    confidence: float = 0.0


@dataclass(frozen=True)
class SynthesisSkipped:
    """Diagnostic record for chunks that did not yield an update."""

    chunk_id: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class SynthesisReport:
    """Aggregate outcome of a batch run."""

    updates: list[UpdateWithGroundTruth]
    skipped: list[SynthesisSkipped] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = len(self.updates) + len(self.skipped)
        return (len(self.updates) / total) if total else 0.0


# ── Synthesizer ───────────────────────────────────────────────────────

class DynamicGroundTruthSynthesizer:
    """Orchestrates masker + DistilBERT + T5 into a single-call API.

    Parameters
    ----------
    device
        ``-1`` for CPU, ``0`` for first GPU.  Default ``-1``.
    distilbert_model
        HuggingFace id of the masked-LM model.
    t5_qg_model
        HuggingFace id of the question-generation model.
    use_spacy
        Whether to attempt spaCy-based masker first.
    min_chunk_chars
        Skip chunks shorter than this.
    min_confidence
        Discard DistilBERT candidates below this score.
    max_question_words
        Discard T5 questions longer than this.
    dedup_questions
        If True (default), drop updates whose question was already emitted
        earlier in the same batch.

    All model-loading happens lazily on the first call to
    :meth:`synthesize_one` / :meth:`synthesize_batch`.
    """

    def __init__(
        self,
        *,
        device: int = DEFAULT_DEVICE,
        distilbert_model: str = "distilbert-base-uncased",
        t5_qg_model: str = "valhalla/t5-base-qg-hl",
        use_spacy: bool = True,
        spacy_model: str = "en_core_web_sm",
        min_chunk_chars: int = MIN_CHUNK_CHARS,
        min_confidence: float = 0.05,
        max_question_words: int = 30,
        dedup_questions: bool = True,
    ) -> None:
        self.device = device
        self.distilbert_model = distilbert_model
        self.t5_qg_model = t5_qg_model
        self.use_spacy = use_spacy
        self.spacy_model = spacy_model
        self.min_chunk_chars = min_chunk_chars
        self.min_confidence = min_confidence
        self.max_question_words = max_question_words
        self.dedup_questions = dedup_questions

    # ── Public API ──────────────────────────────────────────────────

    def synthesize_batch(
        self,
        chunks: Sequence[dict | tuple[str, str] | str],
    ) -> SynthesisReport:
        """Synthesise updates for a batch of chunks.

        Parameters
        ----------
        chunks
            Each item is one of:

            * ``dict`` with keys ``chunk_id`` (str) and ``text``/``chunk``/``content`` (str)
            * ``tuple`` ``(chunk_id, text)``
            * ``str`` — chunk id auto-assigned as ``f"chunk-{i:04d}"``

        Returns
        -------
        SynthesisReport
        """
        updates: list[UpdateWithGroundTruth] = []
        skipped: list[SynthesisSkipped] = []
        seen_questions: set[str] = set()

        for i, raw in enumerate(chunks):
            chunk_id, text = _coerce_chunk(raw, i)

            try:
                update = self._synthesize_one_with_id(chunk_id, text)
            except _Skip as exc:
                skipped.append(SynthesisSkipped(
                    chunk_id=chunk_id, reason=exc.reason, detail=exc.detail,
                ))
                continue

            if self.dedup_questions:
                key = update.question.lower()
                if key in seen_questions:
                    skipped.append(SynthesisSkipped(
                        chunk_id=chunk_id,
                        reason="duplicate_question",
                        detail=update.question,
                    ))
                    continue
                seen_questions.add(key)

            updates.append(update)

        return SynthesisReport(updates=updates, skipped=skipped)

    def synthesize_one(self, chunk_id: str, text: str) -> UpdateWithGroundTruth:
        """Synthesise a single update.  Raises :class:`_Skip` if rejected."""
        return self._synthesize_one_with_id(chunk_id, text)

    # ── Internal ────────────────────────────────────────────────────

    def _synthesize_one_with_id(
        self, chunk_id: str, text: str,
    ) -> UpdateWithGroundTruth:
        if not text or not text.strip():
            raise _Skip("empty_chunk", "chunk text is empty")
        if len(text) < self.min_chunk_chars:
            raise _Skip(
                "chunk_too_short",
                f"chunk has {len(text)} chars < {self.min_chunk_chars}",
            )

        # 1) Mask
        try:
            mask = select_mask(
                text,
                prefer_spacy=self.use_spacy,
                spacy_model=self.spacy_model,
            )
        except MaskError as exc:
            raise _Skip("no_mask_target", str(exc))

        # 2) DistilBERT replacement
        from .distilbert_generator import (
            DistilBERTError, DependencyError as DGBDep, fill_mask,
        )
        try:
            fill = fill_mask(
                mask.masked_chunk,
                original_token=mask.original_token,
                model_name=self.distilbert_model,
                device=self.device,
                min_confidence=self.min_confidence,
            )
        except DGBDep:
            raise
        except DistilBERTError as exc:
            raise _Skip("distilbert_failed", str(exc))

        new_token = fill.chosen.token
        updated = _substitute_mask(text, mask, new_token)

        # 3) T5 question generation, focused on the new token
        from .t5_question_generator import (
            T5QuestionError, DependencyError as T5Dep, generate_question,
        )
        answer_start, answer_end = _span_of_new_token(text, mask, new_token, updated)
        try:
            q = generate_question(
                updated,
                answer_start=answer_start,
                answer_end=answer_end,
                model_name=self.t5_qg_model,
                device=self.device,
                max_question_words=self.max_question_words,
            )
        except T5Dep:
            raise
        except T5QuestionError as exc:
            raise _Skip("t5_failed", str(exc))

        return UpdateWithGroundTruth(
            chunk_id=chunk_id,
            original=text,
            updated=updated,
            masked_token_original=mask.original_token,
            masked_token_new=new_token,
            question=q.question,
            answer=q.answer,
            mask_kind=mask.mask_kind,
            spacy_used=mask.spacy_used,
            confidence=fill.chosen.score,
        )


# ── Helpers ───────────────────────────────────────────────────────────

class _Skip(Exception):
    """Internal control-flow signal used by :meth:`_synthesize_one_with_id`."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


def _coerce_chunk(raw: dict | tuple[str, str] | str, index: int) -> tuple[str, str]:
    """Normalise a chunk input into ``(chunk_id, text)``."""
    if isinstance(raw, str):
        return (f"chunk-{index:04d}", raw)
    if isinstance(raw, tuple) and len(raw) == 2:
        cid, text = raw
        return (str(cid), str(text))
    if isinstance(raw, dict):
        cid = str(raw.get("chunk_id") or raw.get("id") or f"chunk-{index:04d}")
        text = (
            raw.get("text")
            or raw.get("chunk")
            or raw.get("content")
            or raw.get("page_content")
            or ""
        )
        return (cid, str(text))
    raise TypeError(
        f"Unsupported chunk type {type(raw).__name__}; expected dict/tuple/str"
    )


def _substitute_mask(original: str, mask: MaskResult, new_token: str) -> str:
    """Replace the masked region in *original* with *new_token*.

    ``mask.mask_start`` is the char offset where the original token
    started (and where ``[MASK]`` would be inserted in the masked chunk).
    The end of the original token is ``mask_start + len(original_token)``;
    note that ``mask.mask_end`` points past ``[MASK]`` (length 6) in the
    masked chunk, NOT past the original token in the unmasked chunk.
    """
    original_end = mask.mask_start + len(mask.original_token)
    return (
        original[:mask.mask_start]
        + new_token
        + original[original_end:]
    )


def _span_of_new_token(
    original: str,
    mask: MaskResult,
    new_token: str,
    updated: str,
) -> tuple[int, int]:
    """Compute char offsets of *new_token* inside *updated*.

    The replacement sits exactly where ``[MASK]`` was in the masked chunk,
    i.e. between ``mask_start`` and ``mask_start + len(new_token)``.
    """
    start = mask.mask_start
    end = start + len(new_token)
    # Sanity check: the substring at that position must equal new_token.
    if updated[start:end] != new_token:
        # Fall back to a search (handles edge cases around whitespace).
        idx = updated.find(new_token, max(0, start - 5), start + len(new_token) + 10)
        if idx >= 0:
            return (idx, idx + len(new_token))
        # Last resort: first occurrence anywhere.
        idx = updated.find(new_token)
        if idx >= 0:
            return (idx, idx + len(new_token))
        raise _Skip(
            "substitution_misaligned",
            f"new token {new_token!r} not found at expected position",
        )
    return (start, end)


# ── JSONL helpers (used by batch.py) ──────────────────────────────────

def update_to_json_dict(update: UpdateWithGroundTruth) -> dict:
    """Serialise an :class:`UpdateWithGroundTruth` to the JSONL schema."""
    return {
        "chunk_id": update.chunk_id,
        "original": update.original,
        "updated": update.updated,
        "masked_token_original": update.masked_token_original,
        "masked_token_new": update.masked_token_new,
        "question": update.question,
        "answer": update.answer,
    }


def update_from_json_dict(d: dict) -> UpdateWithGroundTruth:
    """Inverse of :func:`update_to_json_dict` (deserialise)."""
    return UpdateWithGroundTruth(
        chunk_id=d["chunk_id"],
        original=d["original"],
        updated=d["updated"],
        masked_token_original=d["masked_token_original"],
        masked_token_new=d["masked_token_new"],
        question=d["question"],
        answer=d["answer"],
    )
