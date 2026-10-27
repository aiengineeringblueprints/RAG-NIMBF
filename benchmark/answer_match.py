"""Deterministic answer matching for datasets with labelled answer patterns.

Patterns are groups of alternatives: an answer matches when every group has
at least one alternative in the answer. Matching is case-insensitive,
Unicode-normalized, ignores Markdown emphasis and thousands separators, and
never matches inside a longer word or number. Alternatives prefixed with
``re:`` are regular expressions searched in the normalized (lower-cased)
answer.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

_THOUSANDS = re.compile(r"(?<=\d)[,\s](?=\d{3}(?!\d))")


def normalize_answer(text: str) -> str:
    """Normalize text once for repeated :func:`answer_matches_normalized` calls."""
    text = unicodedata.normalize("NFKC", text).replace("\u2212", "-")
    text = re.sub(r"[*`_]", "", text.lower())
    text = re.sub(r"\s+", " ", text)
    return _THOUSANDS.sub("", text)


def _contains(answer: str, alternative: str) -> bool:
    if alternative.startswith("re:"):
        return re.search(alternative[3:], answer) is not None
    needle = normalize_answer(alternative).strip()
    if not needle or needle not in answer:  # cheap pre-check before the regex
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", answer) is not None


def answer_matches(answer: str, patterns: Sequence[Sequence[str]]) -> bool:
    """Return whether ``answer`` satisfies every pattern group."""
    return answer_matches_normalized(normalize_answer(answer or ""), patterns)


def answer_matches_normalized(normalized: str, patterns: Sequence[Sequence[str]]) -> bool:
    """Like :func:`answer_matches` for text already passed through :func:`normalize_answer`."""
    if not patterns or not normalized:
        return False
    return all(
        any(_contains(normalized, alternative) for alternative in group)
        for group in patterns
    )
