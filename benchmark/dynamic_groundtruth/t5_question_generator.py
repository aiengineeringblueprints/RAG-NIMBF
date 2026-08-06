"""T5 highlight-based question generator.

Uses ``valhalla/t5-base-qg-hl`` to read a chunk with one highlighted
answer span and emit a single question whose answer is that span.
Highlight format is ``<hl> ... </hl>`` around the answer.

This is the third stage of RAGPerf §3.2: DistilBERT produced a
replacement token; we substitute it back into the chunk, highlight it,
and ask T5 to produce a question whose answer is exactly the highlight.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


DEFAULT_T5_QG_MODEL = "valhalla/t5-base-qg-hl"
DEFAULT_MAX_LENGTH = 64          # tokens — questions are short
DEFAULT_MIN_LENGTH = 4           # tokens — reject "What?"
DEFAULT_MAX_QUESTION_WORDS = 30  # spec §3.2 quality safeguard

HL_OPEN = "<hl>"
HL_CLOSE = "</hl>"


# ── Public dataclasses ────────────────────────────────────────────────

@dataclass(frozen=True)
class QuestionResult:
    """A single generated question + answer span."""

    question: str
    answer: str
    raw_output: str


class T5QuestionError(RuntimeError):
    """Raised when T5 can't generate an acceptable question."""


class DependencyError(ImportError):
    """Raised when ``transformers`` or ``torch`` is not installed."""


# ── Module-level cache ────────────────────────────────────────────────

_MODEL_CACHE: dict[tuple[str, int], tuple[object, object]] = {}


def _cache_dir() -> str | None:
    return os.getenv("DYNAMIC_GTM_CACHE_DIR") or None


def _ensure_model(
    model_name: str,
    device: int,
) -> tuple[object, object]:
    """Lazily load ``(tokenizer, model)``.  Cached at module level."""
    key = (model_name, device)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        import torch  # noqa: F401  (required by transformers backend)
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise DependencyError(
            "transformers + torch are required for T5 question generation. "
            "Install them with `pip install transformers torch`."
        ) from exc

    cache_dir = _cache_dir()
    logger.debug(
        "Loading T5 QG model '%s' on device=%d (cache_dir=%s)",
        model_name, device, cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
    try:
        model = model.to(_torch_device_str(device))
    except Exception as exc:  # CPU fallback / device mapping issues
        logger.warning("Could not move T5 model to device %d: %s", device, exc)

    _MODEL_CACHE[key] = (tokenizer, model)
    return (tokenizer, model)


def _torch_device_str(device: int) -> str:
    return "cpu" if device < 0 else f"cuda:{device}"


# ── Highlight helpers ─────────────────────────────────────────────────

def _wrap_highlight(chunk: str, span_start: int, span_end: int) -> str:
    """Insert ``<hl>...</hl>`` around the answer span."""
    return (
        chunk[:span_start]
        + HL_OPEN
        + chunk[span_start:span_end]
        + HL_CLOSE
        + chunk[span_end:]
    )


# Strip leftover control tokens that occasionally leak from T5.
_LEAKY_TOKEN_RE = re.compile(r"<[^>]+>|^\s*(?:question|generate question)\s*[:\-]\s*", re.IGNORECASE)


def _clean_question(raw: str) -> str:
    """Normalise the question: strip tokens, collapse whitespace, capitalise."""
    if not raw:
        return ""
    cleaned = _LEAKY_TOKEN_RE.sub("", raw).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned and not cleaned[0].isupper():
        cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned and not cleaned.endswith(("?", ".", "!")):
        cleaned += "?"
    return cleaned


def _word_count(text: str) -> int:
    return len(text.split())


# ── Public API ────────────────────────────────────────────────────────

def generate_question(
    chunk_with_answer: str,
    answer_start: int,
    answer_end: int,
    *,
    model_name: str = DEFAULT_T5_QG_MODEL,
    device: int = -1,
    max_length: int = DEFAULT_MAX_LENGTH,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_question_words: int = DEFAULT_MAX_QUESTION_WORDS,
) -> QuestionResult:
    """Generate a question whose answer is the highlighted span.

    Parameters
    ----------
    chunk_with_answer
        The chunk after DistilBERT substitution — already contains the
        replacement token at the answer position.
    answer_start, answer_end
        Character offsets of the answer span inside *chunk_with_answer*.
    model_name
        HF model id, default ``valhalla/t5-base-qg-hl``.
    device
        ``-1`` for CPU, ``0`` for first GPU, etc.
    max_length, min_length
        Token-level bounds passed to ``model.generate``.
    max_question_words
        Reject questions longer than this (spec §3.2 safeguard, default 30).

    Returns
    -------
    QuestionResult

    Raises
    ------
    T5QuestionError
        If the model output is empty or fails quality gates.
    DependencyError
        If transformers/torch aren't installed.
    """
    if not (0 <= answer_start < answer_end <= len(chunk_with_answer)):
        raise T5QuestionError(
            f"Invalid answer span ({answer_start}, {answer_end}) for "
            f"chunk of length {len(chunk_with_answer)}"
        )

    highlighted = _wrap_highlight(chunk_with_answer, answer_start, answer_end)
    tokenizer, model = _ensure_model(model_name, device)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover — defensive
        raise DependencyError("torch is required for T5 generation") from exc

    inputs = tokenizer(
        highlighted,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            early_stopping=True,
        )

    raw = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    question = _clean_question(raw)

    if not question:
        raise T5QuestionError("T5 produced an empty question")

    word_count = _word_count(question)
    if word_count > max_question_words:
        raise T5QuestionError(
            f"T5 question too long ({word_count} words > {max_question_words}): "
            f"{question!r}"
        )
    if word_count < 3:
        raise T5QuestionError(
            f"T5 question too short ({word_count} words): {question!r}"
        )

    answer = chunk_with_answer[answer_start:answer_end]
    return QuestionResult(question=question, answer=answer, raw_output=raw)


def clear_model_cache() -> None:
    """Drop cached models.  Useful for tests / forced reloads."""
    _MODEL_CACHE.clear()
