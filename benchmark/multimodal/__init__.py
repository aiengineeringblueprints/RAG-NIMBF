"""Multi-modal ingestion for the RAG benchmarking framework.

Adds three ingestion paths inspired by RAGPerf §3.3.1 / §4.1:

- **PDF/Image OCR** (``pdf_ocr_docling``, ``image_ocr_docling``):
  Docling -> per-page text -> standard text pipeline.
- **PDF/Image Visual** (``colpali_visual``):
  ColPali -> multi-vector embeddings per page -> ColBERT-style max-sim ranker.
  Stored in a dedicated JSON-blob collection (NOT compatible with the standard
  Chroma text collection).
- **Audio ASR** (``audio_whisper``):
  Whisper -> transcript -> standard text pipeline.

All heavy ML dependencies (``docling``, ``colpali-engine``, ``faster-whisper``,
``openai-whisper``, ``torch``) are *lazy* — importing this package does not
require any of them. Missing deps raise a clear :class:`OptionalDependencyError`
when the relevant code path is actually invoked.

This is a Tier-3 functional prototype (see NOTES_G_Multimodal.md). The text
pipeline is unchanged; multimodal chunkers produce ordinary ``{context,
metadata}`` dicts that flow through the existing chunk -> embed -> retrieve
-> generate path.
"""

from __future__ import annotations

from benchmark.multimodal.registry import (
    MULTIMODAL_CHUNKERS,
    MULTIMODAL_EMBEDDERS,
    OptionalDependencyError,
    available_multimodal_chunkers,
    available_multimodal_embedders,
    is_chunker_multimodal,
    is_embedder_multimodal,
    register_multimodal_chunker,
    register_multimodal_embedder,
)

__all__ = [
    "MULTIMODAL_CHUNKERS",
    "MULTIMODAL_EMBEDDERS",
    "OptionalDependencyError",
    "available_multimodal_chunkers",
    "available_multimodal_embedders",
    "is_chunker_multimodal",
    "is_embedder_multimodal",
    "register_multimodal_chunker",
    "register_multimodal_embedder",
]
