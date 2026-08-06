"""DistilBERT masked-LM replacement generator.

Wraps ``distilbert-base-uncased`` so we can propose a plausible
replacement for the mask token chosen by :mod:`masker`.  Per RAGPerf
§3.2, the replacement should be:

* highly probable under the language model (so the resulting chunk is
  fluent), and
* **semantically distinct** from the original — otherwise the update is
  invisible and the question/answer pair trivially matches the original
  chunk too.

Module-level model cache ensures the model is loaded at most once per
process.  CPU fallback is always available; set ``CUDA_VISIBLE_DEVICES``
or pass ``device=-1`` to force it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from .masker import MASK_TOKEN

logger = logging.getLogger(__name__)


DEFAULT_DISTILBERT_MODEL = "distilbert-base-uncased"
DEFAULT_TOP_K = 10
DEFAULT_MIN_CONFIDENCE = 0.05  # per spec §3.2 quality safeguard


# ── Public dataclasses ────────────────────────────────────────────────

@dataclass(frozen=True)
class Candidate:
    """One fill-mask proposal."""

    token: str           # leading space stripped, lower-cased
    score: float         # softmax probability from the model
    raw_sequence: str    # full input with the candidate substituted in


@dataclass(frozen=True)
class FillMaskResult:
    """The chosen replacement plus alternatives for diagnostics."""

    chosen: Candidate
    alternatives: list[Candidate]


class DistilBERTError(RuntimeError):
    """Raised when the model can't produce any acceptable candidate."""


class DependencyError(ImportError):
    """Raised when ``transformers`` or ``torch`` is not installed."""


# ── Module-level cache ────────────────────────────────────────────────

# Two caches keyed by ``(model_name, device)``.  Loading a HF model twice
# is wasteful — a single instance is thread-safe for inference.
_PIPELINE_CACHE: dict[tuple[str, int], object] = {}


def _cache_dir() -> str | None:
    """Honour the ``DYNAMIC_GTM_CACHE_DIR`` env var for HF downloads."""
    return os.getenv("DYNAMIC_GTM_CACHE_DIR") or None


def _ensure_pipeline(
    model_name: str,
    device: int,
) -> object:
    """Lazily instantiate and cache a ``fill-mask`` pipeline.

    Raises :class:`DependencyError` with a clear message if the
    transformers stack isn't installed.
    """
    key = (model_name, device)
    cached = _PIPELINE_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        import torch  # noqa: F401  (required by transformers backend)
        from transformers import (
            AutoModelForMaskedLM,
            AutoTokenizer,
            pipeline,
        )
    except ImportError as exc:
        raise DependencyError(
            "transformers + torch are required for DistilBERT fill-mask. "
            "Install them with `pip install transformers torch`."
        ) from exc

    cache_dir = _cache_dir()
    logger.debug(
        "Loading DistilBERT '%s' on device=%d (cache_dir=%s)",
        model_name, device, cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)
    pipe = pipeline(
        task="fill-mask",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=DEFAULT_TOP_K,
    )
    _PIPELINE_CACHE[key] = pipe
    return pipe


# ── Helpers ───────────────────────────────────────────────────────────

def _normalise_token(token_str: str) -> str:
    """Strip the leading space that BERT tokenisers prepend to word pieces."""
    return token_str.strip().lower()


def _is_punctuation_or_stopword(token: str) -> bool:
    """Reject candidates that are pure punctuation or function words."""
    if not token:
        return True
    if not any(c.isalnum() for c in token):
        return True
    STOP = {
        "the", "a", "an", "this", "that", "these", "those",
        "is", "are", "was", "were", "be", "been", "being",
        "and", "or", "but", "of", "in", "on", "at", "to", "for",
        "with", "by", "from", "as", "it", "its", "his", "her",
    }
    return token.lower() in STOP


# ── Public API ────────────────────────────────────────────────────────

def fill_mask(
    masked_chunk: str,
    *,
    original_token: str | None = None,
    model_name: str = DEFAULT_DISTILBERT_MODEL,
    device: int = -1,
    top_k: int = DEFAULT_TOP_K,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> FillMaskResult:
    """Fill the single ``[MASK]`` token in *masked_chunk*.

    Parameters
    ----------
    masked_chunk
        Text containing exactly one ``[MASK]`` substring.
    original_token
        The original token that was masked, used to ensure semantic
        distinctness.  If ``None``, distinctness check is skipped.
    model_name
        HuggingFace model id.  Defaults to ``distilbert-base-uncased``.
    device
        ``-1`` for CPU, ``0`` for first GPU, etc.  Default ``-1``.
    top_k
        Number of candidates to request from the model.  Default 10.
    min_confidence
        Discard candidates with score below this threshold.

    Returns
    -------
    FillMaskResult

    Raises
    ------
    DistilBERTError
        If the model produced no acceptable candidate.
    DependencyError
        If transformers/torch aren't installed.
    """
    if MASK_TOKEN not in masked_chunk:
        raise DistilBERTError(
            "masked_chunk does not contain the [MASK] token"
        )
    if masked_chunk.count(MASK_TOKEN) > 1:
        raise DistilBERTError(
            "masked_chunk contains multiple [MASK] tokens; only one supported"
        )

    pipe = _ensure_pipeline(model_name, device)
    raw_preds = pipe(masked_chunk, top_k=top_k)

    candidates: list[Candidate] = []
    for pred in raw_preds:
        # Predictions from fill-mask pipeline are dicts.
        token_str = pred.get("token_str") or pred.get("token_text") or ""
        token = _normalise_token(token_str)
        score = float(pred.get("score", 0.0))
        if score < min_confidence:
            continue
        if _is_punctuation_or_stopword(token):
            continue
        candidates.append(Candidate(
            token=token,
            score=score,
            raw_sequence=pred.get("sequence", ""),
        ))

    if not candidates:
        raise DistilBERTError(
            "No fill-mask candidate passed the confidence / stopword filters"
        )

    chosen = _pick_distinct(candidates, original_token)
    return FillMaskResult(chosen=chosen, alternatives=candidates)


def _pick_distinct(
    candidates: list[Candidate],
    original_token: str | None,
) -> Candidate:
    """Pick the highest-scoring candidate that differs from the original.

    DistilBERT occasionally re-predicts the original token verbatim.  That
    is useless for an update operation — the masked replacement would
    equal the original, producing a no-op update.  We always prefer the
    top-scoring candidate but fall back to lower-ranked ones if the
    highest is the same string.
    """
    if original_token is None:
        return candidates[0]

    original_norm = original_token.strip().lower()
    for cand in candidates:
        if cand.token != original_norm:
            return cand

    # Everything matched the original — return the top-scoring one anyway;
    # the caller may still want to proceed or skip via quality gates.
    logger.warning(
        "All fill-mask candidates equal the original token '%s'; "
        "returning top-scoring candidate", original_norm,
    )
    return candidates[0]


def clear_model_cache() -> None:
    """Drop cached pipelines.  Useful for tests / forced reloads."""
    _PIPELINE_CACHE.clear()
