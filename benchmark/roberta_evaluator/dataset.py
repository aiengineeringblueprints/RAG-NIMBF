"""RAGBench dataset loading and TRACe label extraction.

The RAGBench dataset (arXiv:2407.11005v2) is hosted on the Hugging Face
Hub under two equivalent ids — ``rungalileo/ragbench`` and
``galileo-ai/ragbench`` — with 12 subset configs (covidqa, cuad,
delucionqa, emanual, expertqa, finqa, hagrid, hotpotqa, msmarco,
pubmedqa, tatqa, techqa), each split into train / validation / test.

Each example exposes the four TRACe labels we want to predict:

    * ``utilization_score``  – float in [0, 1]
    * ``relevance_score``    – float in [0, 1]
    * ``completeness_score`` – float in [0, 1]
    * ``adherence_score``    – bool / {0, 1}

This module turns the raw HF rows into the flat ``(question, context,
response, labels)`` tuples the multi-task classifier consumes.

``datasets`` is imported lazily so the package can be imported on
machines that don't have HF stack installed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)

# Default HF dataset id. ``rungalileo/ragbench`` is the canonical name
# referenced in the RAGBench paper; ``galileo-ai/ragbench`` mirrors it.
RAGBENCH_HUB_ID = "rungalileo/ragbench"
RAGBENCH_MIRROR_HUB_ID = "galileo-ai/ragbench"
DEFAULT_SUBSETS = (
    "covidqa",
    "cuad",
    "delucionqa",
    "emanual",
    "expertqa",
    "finqa",
    "hagrid",
    "hotpotqa",
    "msmarco",
    "pubmedqa",
    "tatqa",
    "techqa",
)


@dataclass(frozen=True)
class TraceExample:
    """A single training/inference example.

    ``context`` is the concatenation of the retrieved documents for the
    example (the raw ``documents`` field is a list of [key, text] pairs).
    ``labels`` carries the four TRACe targets; any field that is None is
    treated as missing for the corresponding head (mask out in loss).
    """

    question: str
    context: str
    response: str
    labels: dict[str, float]


def _documents_to_context(documents: list | None) -> str:
    """Flatten RAGBench ``documents`` into a single context string.

    The field is a list of ``[key, text]`` pairs (or already-flat strings).
    The keys (e.g. ``"0a"``) are dropped; only the passage text is joined
    with newlines, matching how the original RAGBench paper feeds context
    to its evaluators.
    """
    if not documents:
        return ""
    chunks: list[str] = []
    for doc in documents:
        if isinstance(doc, (list, tuple)):
            if len(doc) >= 2:
                chunks.append(str(doc[1]))
            elif len(doc) == 1:
                chunks.append(str(doc[0]))
        else:
            chunks.append(str(doc))
    return "\n".join(chunks).strip()


def _coerce_float(value, default: float | None = None) -> float | None:
    """Best-effort conversion of HF label fields to float.

    RAGBench stores adherence as bool/int and the other TRACe labels as
    floats; some rows carry None.  Anything we cannot parse falls back to
    *default* so the row can still be emitted with that head masked.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def row_to_example(row: dict) -> TraceExample:
    """Convert a single RAGBench row to a :class:`TraceExample`.

    Field names follow the public RAGBench schema.  Both ``response`` and
    the older ``answer`` spelling are accepted so the loader is robust to
    minor schema drift between dataset revisions.
    """
    question = (row.get("question") or "").strip()
    response = (row.get("response") or row.get("answer") or "").strip()
    context = _documents_to_context(row.get("documents"))

    labels = {
        "utilization": _coerce_float(row.get("utilization_score")),
        "relevance": _coerce_float(row.get("relevance_score")),
        "completeness": _coerce_float(row.get("completeness_score")),
        "adherence": _coerce_float(row.get("adherence_score")),
    }
    return TraceExample(question=question, context=context, response=response, labels=labels)


def _load_hub(hub_id: str, subset: str, split: str, **load_kwargs):
    """Lazy ``datasets.load_dataset`` wrapper."""
    from datasets import load_dataset  # type: ignore[import-not-found]

    return load_dataset(hub_id, subset, split=split, **load_kwargs)


def load_ragbench_split(
    subset: str | Iterable[str] = DEFAULT_SUBSETS,
    split: str = "train",
    *,
    hub_id: str = RAGBENCH_HUB_ID,
    streaming: bool = False,
    **load_kwargs,
) -> Iterator[TraceExample]:
    """Yield :class:`TraceExample` objects from one or more RAGBench subsets.

    Parameters
    ----------
    subset
        A single subset name (e.g. ``"covidqa"``) or an iterable of them.
        Defaults to every published subset.
    split
        HF split name — ``train``, ``validation`` or ``test``.
    hub_id
        HF dataset id.  Falls back to :data:`RAGBENCH_MIRROR_HUB_ID` on
        ``ConnectionError``.
    streaming
        Use HF streaming mode (memory-friendly for big subsets).
    """
    if isinstance(subset, str):
        subsets = [subset]
    else:
        subsets = list(subset)

    last_error: Exception | None = None
    for hub in (hub_id, RAGBENCH_MIRROR_HUB_ID):
        for sub in subsets:
            try:
                ds = _load_hub(hub, sub, split, streaming=streaming, **load_kwargs)
            except Exception as exc:  # pragma: no cover - network/availability
                last_error = exc
                logger.warning(
                    "Failed to load RAGBench subset %r from %s: %s", sub, hub, exc
                )
                continue
            for row in ds:
                yield row_to_example(dict(row))
        if last_error is None:
            return  # all subsets loaded from this hub
    if last_error is not None:
        raise RuntimeError(
            f"Could not load RAGBench subsets {subsets!r} from {hub_id!r} or "
            f"mirror {RAGBENCH_MIRROR_HUB_ID!r}. Last error: {last_error}"
        )


def to_trace_tuples(
    examples: Iterable[TraceExample],
) -> list[dict]:
    """Flatten examples into JSON-serialisable dicts (training cache format)."""
    out: list[dict] = []
    for ex in examples:
        out.append(
            {
                "question": ex.question,
                "context": ex.context,
                "response": ex.response,
                "labels": dict(ex.labels),
            }
        )
    return out
