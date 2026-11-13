"""Canonical document ID resolution shared across the framework.

Every path that produces or consumes a document ID — gold-ID stamping on
per-sample datasets (``benchmark.dataset.assign_gold_doc_ids``), adapter
indexing (e.g. ``examples.enterprise_rag_plugin.adapter``), and adapter
retrieval metadata (``benchmark.adapters.mcp``) — must go through the same
resolver, so a retrieved chunk can actually be matched against its gold ID
by ``benchmark.gold_retrieval_metrics``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def canonical_doc_id(value: Any) -> str | None:
    """Normalise a raw source value into the canonical document ID.

    Path-like sources are reduced to their stem (``/docs/manual.pdf`` and
    ``manual.pdf`` both become ``manual``), matching how the MCP adapter and
    the local-corpus loaders name indexed documents. Dotted identifiers
    without a path shape keep their stem too — since both the gold and the
    retrieval side run through the same function, they stay comparable.
    ``None`` and whitespace-only values resolve to ``None``.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).stem or text


def resolve_dataset_doc_id(
    metadata: Mapping[str, Any] | None, index: int | None = None
) -> str | None:
    """Resolve the canonical document ID from sample metadata.

    Precedence mirrors the adapters' indexing: ``metadata["source"]`` wins
    over ``metadata["doc_id"]``; empty or ``None`` values fall through to the
    next candidate. When ``index`` is given, per-sample-context datasets fall
    back to ``dataset-doc-{index}`` — the same ID the adapters use when
    indexing a sample without an explicit source.
    """
    doc_id = None
    if metadata:
        doc_id = canonical_doc_id(metadata.get("source") or metadata.get("doc_id"))
    if doc_id:
        return doc_id
    if index is not None:
        return f"dataset-doc-{index}"
    return None
