"""Canonical document ID resolution shared across the framework.

Every path that produces or consumes a document ID — gold-ID stamping on
per-sample datasets (``benchmark.dataset.assign_gold_doc_ids``), adapter
indexing (e.g. ``examples.enterprise_rag_plugin.adapter``), and adapter
retrieval metadata (``benchmark.adapters.mcp``) — must go through the same
resolver, so a retrieved chunk can actually be matched against its gold ID
by ``benchmark.gold_retrieval_metrics``.
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Mapping

# Trailing suffixes that mark a value as a raw file name. Only these are
# stripped during canonicalisation; unknown dotted suffixes (e.g. ``.v1`` in
# ``release.v1``, or versioned names like ``report.1``) are part of the
# document identity and are preserved, so distinct documents never collapse
# into the same ID.
_DOCUMENT_SUFFIXES = {
    ".pdf",
    ".md",
    ".txt",
    ".doc",
    ".docx",
    ".html",
    ".htm",
    ".csv",
    ".json",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}


def canonical_doc_id(value: Any) -> str | None:
    """Normalise a raw source value into the canonical document ID.

    Guarantees:

    - **Idempotent**: ``canonical_doc_id(canonical_doc_id(x)) ==
      canonical_doc_id(x)``. Already-canonical IDs pass through unchanged,
      so gold stamping, indexing, and retrieval metadata can apply the
      function repeatedly in sequence without drifting apart.
    - **Collision-free**: distinct documents never map to the same ID.
      Values that look like file paths (contain a path separator) keep the
      full normalised path, so ``/a/manual.pdf`` and ``/b/manual.pdf`` stay
      distinct. Suffix stripping only removes *known document extensions*
      from bare file names, so ``report.1`` and ``report.2`` keep their
      identity while ``manual.pdf`` → ``manual``.

    ``None`` and whitespace-only values resolve to ``None``.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "/" in text or "\\" in text:
        # A real path: keep it fully, normalised, so same-named files in
        # different directories remain distinguishable.
        path = (
            PureWindowsPath(text)
            if "\\" in text
            else PurePosixPath(text)
        )
        return path.as_posix()
    lowered = text.lower()
    for suffix in _DOCUMENT_SUFFIXES:
        if lowered.endswith(suffix) and len(text) > len(suffix):
            return text[: -len(suffix)]
    return text


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


def resolve_raw_doc_id(
    metadata: Mapping[str, Any] | None, index: int | None = None
) -> str | None:
    """Resolve the document ID from metadata *without* canonicalisation.

    Used for shared-corpus data, whose existing IDs (from the dataset
    loaders or the user's own manifest) must pass through unchanged — only
    missing IDs fall back to ``dataset-doc-{index}``.
    """
    if metadata:
        for key in ("source", "doc_id"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value)
    if index is not None:
        return f"dataset-doc-{index}"
    return None
