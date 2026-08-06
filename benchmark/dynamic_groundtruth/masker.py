"""Mask-target selection for dynamic ground-truth generation.

The masker is the first of three stages described in RAGPerf §3.2.
Given a chunk of text, it picks a single token to obscure, preferring
numeric values (years, percentages, magnitudes) over noun phrases.
The output is the chunk with one ``[MASK]`` token in place of the
original, plus metadata so downstream stages can locate and rewrite it.

Design
------
* spaCy is used when available because its POS tagger cleanly separates
  proper nouns (``PROPN``), common nouns (``NOUN``), and cardinal
  numbers (``NUM``).  The pipeline falls back to deterministic regex +
  suffix heuristics when spaCy is missing — that fallback is what the
  fast unit tests exercise.
* Only one mask per chunk is ever produced.  Multi-mask generation is
  out of scope (it would change the question-generation prompt format
  used by ``valhalla/t5-base-qg-hl``).
* Mask offsets are character-based into the returned ``masked_chunk``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── Public dataclasses ────────────────────────────────────────────────

@dataclass(frozen=True)
class MaskResult:
    """Result of selecting a mask target inside a chunk.

    Attributes
    ----------
    masked_chunk
        Copy of *chunk* with the chosen token replaced by ``"[MASK]"``.
        This is the string DistilBERT consumes.
    original_token
        The exact substring that was replaced (preserves case/punctuation).
    mask_start, mask_end
        Character offsets of the ``[MASK]`` token inside *masked_chunk*.
    mask_kind
        ``"number"`` or ``"noun"`` — useful for analytics / filtering.
    spacy_used
        Whether spaCy was used to pick the target (vs regex fallback).
    """

    masked_chunk: str
    original_token: str
    mask_start: int
    mask_end: int
    mask_kind: str
    spacy_used: bool


class MaskError(ValueError):
    """Raised when no suitable mask target exists in a chunk."""


# ── Regex fallback ────────────────────────────────────────────────────

# Numbers: 4-digit years, percentages, magnitudes with thousands separators.
# Order matters — years first, then percentages, then plain numbers.
_NUMBER_PATTERN = re.compile(
    r"""
    (?<![\w.])                  # not preceded by word char or dot (avoid IPs/dates split)
    (?:
        \d{4}                   # 4-digit year-like
        |                       # -- or --
        \d{1,3}(?:,\d{3})+      # thousands-separated: 1,000,000
        |                       # -- or --
        \d+(?:\.\d+)?           # plain int / decimal
    )
    %?                          # optional trailing percent
    (?![\w.])                   # not followed by word char or dot
    """,
    re.VERBOSE,
)

# Crude noun detection for the no-spaCy path.  We deliberately keep this
# conservative: a long, capitalised or all-lowercase token that doesn't
# look like a stopword / verb / adjective.  spaCy replaces this entirely
# when available.
_STOPWORDY_SUFFIXES = (
    "ing", "ed", "ly", "ous", "ive", "able", "ible", "ful",
    "less", "est", "er", "ors", "ism", "ist",
)
_PRONOUNS_AND_AUX = {
    "the", "a", "an", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "but", "as", "if", "than", "then",
    "he", "she", "it", "they", "we", "you", "i",
    "his", "her", "its", "their", "our", "your", "my",
    "him", "them", "us", "me",
    "who", "what", "when", "where", "why", "how", "which",
}
_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def _looks_like_noun_fallback(token: str) -> bool:
    """Heuristic noun detection used when spaCy is unavailable.

    Returns True for tokens that are:
    * not a pronoun / article / auxiliary verb (per the stopword set)
    * not ending in a verb/adjective suffix
    * at least 3 characters long
    """
    lower = token.lower()
    if len(lower) < 3:
        return False
    if lower in _PRONOUNS_AND_AUX:
        return False
    if any(lower.endswith(suf) and len(lower) > len(suf) + 2 for suf in _STOPWORDY_SUFFIXES):
        return False
    return True


def _fallback_select(chunk: str) -> tuple[str, int, int, str] | None:
    """Pick a mask target using regex + heuristics.

    Returns ``(original_token, start, end, kind)`` or ``None`` if no
    suitable target exists.
    """
    # 1) Numbers — first match wins.
    m = _NUMBER_PATTERN.search(chunk)
    if m:
        return (m.group(0), m.start(), m.end(), "number")

    # 2) Nouns via heuristic POS detection.
    for wm in _WORD_PATTERN.finditer(chunk):
        token = wm.group(0)
        if _looks_like_noun_fallback(token):
            return (token, wm.start(), wm.end(), "noun")

    return None


# ── spaCy path ────────────────────────────────────────────────────────

# Module-level cache so we don't reload spaCy per call.
_NLP_CACHE: dict[str, object] = {}


def _get_spacy_nlp(model_name: str = "en_core_web_sm") -> "object | None":
    """Load a spaCy pipeline, cached at module level.

    Returns ``None`` if spaCy or the model is unavailable.  Never raises.
    """
    if model_name in _NLP_CACHE:
        return _NLP_CACHE[model_name]

    try:
        import spacy  # type: ignore[import-not-found]
    except ImportError:
        logger.debug("spaCy not installed; falling back to regex masker")
        _NLP_CACHE[model_name] = None
        return None

    try:
        nlp = spacy.load(model_name, disable=["parser", "ner", "lemmatizer"])
    except Exception as exc:  # OSError if model not installed
        logger.debug(
            "spaCy model '%s' not available (%s); falling back to regex masker",
            model_name, exc,
        )
        _NLP_CACHE[model_name] = None
        return None

    _NLP_CACHE[model_name] = nlp
    return nlp


def _spacy_select(
    chunk: str,
    nlp: object,
) -> tuple[str, int, int, str] | None:
    """Pick a mask target using spaCy POS tags.

    Preference order: NUM (cardinals) → PROPN (named entities / proper
    nouns) → NOUN (common nouns).  First matching token wins.
    """
    doc = nlp(chunk)
    num_tok = None
    propn_tok = None
    noun_tok = None
    for tok in doc:
        text = tok.text
        if not text.strip():
            continue
        pos = tok.pos_
        if pos == "NUM" and num_tok is None:
            num_tok = (text, tok.idx, tok.idx + len(text), "number")
        elif pos == "PROPN" and propn_tok is None and len(text) >= 3:
            propn_tok = (text, tok.idx, tok.idx + len(text), "noun")
        elif pos == "NOUN" and noun_tok is None and len(text) >= 3:
            noun_tok = (text, tok.idx, tok.idx + len(text), "noun")

    return num_tok or propn_tok or noun_tok


# ── Top-level API ─────────────────────────────────────────────────────

# Mask token used by DistilBERT (and any BERT-style MLM).
MASK_TOKEN = "[MASK]"


def select_mask(
    chunk: str,
    *,
    prefer_spacy: bool = True,
    spacy_model: str = "en_core_web_sm",
) -> MaskResult:
    """Select a single mask target in *chunk*.

    Parameters
    ----------
    chunk
        Input text.  Must be non-empty.
    prefer_spacy
        Try spaCy first, fall back to regex.  Defaults to ``True``.
    spacy_model
        spaCy model name.  Defaults to the small English web model.

    Returns
    -------
    MaskResult

    Raises
    ------
    MaskError
        If no noun or number can be found in *chunk*.
    """
    if not chunk or not chunk.strip():
        raise MaskError("Cannot mask empty chunk")

    candidate: tuple[str, int, int, str] | None = None
    spacy_used = False

    if prefer_spacy:
        nlp = _get_spacy_nlp(spacy_model)
        if nlp is not None:
            candidate = _spacy_select(chunk, nlp)
            spacy_used = True

    if candidate is None:
        candidate = _fallback_select(chunk)

    if candidate is None:
        raise MaskError(
            "No numeric or noun mask target found in chunk "
            f"(len={len(chunk)})"
        )

    original_token, start, end, kind = candidate
    masked_chunk = chunk[:start] + MASK_TOKEN + chunk[end:]
    mask_start = start
    mask_end = start + len(MASK_TOKEN)

    return MaskResult(
        masked_chunk=masked_chunk,
        original_token=original_token,
        mask_start=mask_start,
        mask_end=mask_end,
        mask_kind=kind,
        spacy_used=spacy_used,
    )
