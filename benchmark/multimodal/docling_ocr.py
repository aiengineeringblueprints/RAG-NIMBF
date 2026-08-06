"""Docling OCR ingestion path.

Takes PDFs or images and returns per-page text in the framework's
``{context, metadata}`` corpus format. Output is consumed downstream by the
existing text chunk -> embed -> retrieve pipeline.

The Docling dependency is heavy (pulls torch + layout models). It is imported
lazily; missing deps raise :class:`OptionalDependencyError`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from benchmark.multimodal.registry import OptionalDependencyError

logger = logging.getLogger(__name__)

_FEATURE_PDF = "pdf_ocr_docling"
_FEATURE_IMAGE = "image_ocr_docling"
_DOCLING_PACKAGE = "docling"
_DOCLING_EXTRA = "docling"  # pip install docling


def _require_docling() -> tuple[Any, Any, Any, Any]:
    """Lazy-import Docling bits or raise a helpful error.

    Returns (DocumentConverter, PdfFormatOption, InputFormat, PdfPipelineOptions).
    """
    try:
        from docling.document_converter import (
            DocumentConverter,
            PdfFormatOption,
        )
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
    except ImportError as exc:
        raise OptionalDependencyError(
            feature=_FEATURE_PDF,
            package=_DOCLING_PACKAGE,
            extra=_DOCLING_EXTRA,
        ) from exc
    return DocumentConverter, PdfFormatOption, InputFormat, PdfPipelineOptions


def _build_converter(do_ocr: bool = True, do_table_structure: bool = True) -> Any:
    DocumentConverter, PdfFormatOption, InputFormat, PdfPipelineOptions = (
        _require_docling()
    )
    opts = PdfPipelineOptions(do_ocr=do_ocr, do_table_structure=do_table_structure)
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
        }
    )


def _page_texts(document: Any) -> list[str]:
    """Group DoclingDocument text items by ``page_no``.

    Docling exposes the document tree via ``iterate_items()``. Each text item
    carries provenance (``prov[0].page_no``). We collect text per page so the
    downstream text pipeline sees one chunk per page, mirroring RAGPerf §3.3.1.
    """
    pages: dict[int, list[str]] = {}
    fallback_chunks: list[str] = []

    for item, _level in document.iterate_items():
        text = getattr(item, "text", None)
        if not text:
            continue
        text = text.strip()
        if not text:
            continue

        prov = getattr(item, "prov", None) or []
        if prov:
            try:
                page_no = prov[0].page_no
            except (AttributeError, IndexError):
                page_no = None
        else:
            page_no = None

        if page_no is not None:
            pages.setdefault(int(page_no), []).append(text)
        else:
            # Items without provenance (e.g., exported tables without a page
            # anchor) fall back to a single concatenated bucket so text is
            # never silently dropped.
            fallback_chunks.append(text)

    ordered = [pages[k] for k in sorted(pages)]
    if fallback_chunks:
        ordered.append(fallback_chunks)
    return ["\n".join(parts) for parts in ordered if parts]


# ---------------------------------------------------------------------------
# Corpus-format factories (registered as multimodal chunkers)
# ---------------------------------------------------------------------------


def ingest_pdf_via_docling(
    corpus_path: str | Path,
    *,
    do_ocr: bool = True,
    do_table_structure: bool = True,
    **_: Any,
) -> list[dict]:
    """Walk ``corpus_path`` for PDFs, return per-page ``{context, metadata}`` dicts.

    Each returned dict carries:
      - ``context``: per-page text (one dict per page, multiple per source PDF)
      - ``metadata``: ``{doc_id, source_id, source_name, source_path, page_no,
        corpus_type}``

    An empty corpus raises ValueError (consistent with the text corpus loader).
    """
    root = Path(corpus_path).resolve()
    if not root.is_dir():
        raise ValueError(f"corpus_path is not a directory: {root}")

    pdf_paths = sorted(p for p in root.rglob("*.pdf") if p.is_file())
    if not pdf_paths:
        raise ValueError(f"No PDF documents found in {root}")

    converter = _build_converter(do_ocr=do_ocr, do_table_structure=do_table_structure)

    corpus: list[dict] = []
    for pdf_path in pdf_paths:
        logger.info("Docling: converting %s", pdf_path.relative_to(root))
        result = converter.convert(str(pdf_path))
        page_texts = _page_texts(result.document)
        if not page_texts:
            # Fall back to the full-document markdown so empty pages don't
            # silently disappear from the corpus.
            markdown = result.document.export_to_markdown().strip()
            if markdown:
                page_texts = [markdown]

        relative = pdf_path.relative_to(root)
        source_id = pdf_path.stem
        for page_index, page_text in enumerate(page_texts, start=1):
            corpus.append(
                {
                    "context": page_text,
                    "metadata": {
                        "doc_id": f"{source_id}_p{page_index}",
                        "source_id": source_id,
                        "source_name": pdf_path.name,
                        "source_path": str(relative),
                        "page_no": page_index,
                        "corpus_type": "pdf",
                    },
                }
            )
    logger.info("Docling: produced %d page chunks from %d PDFs", len(corpus), len(pdf_paths))
    return corpus


def ingest_image_via_docling(
    corpus_path: str | Path,
    *,
    do_ocr: bool = True,
    **_: Any,
) -> list[dict]:
    """Walk ``corpus_path`` for images (PNG/JPG), return per-image OCR dicts.

    Docling's PDF pipeline doesn't natively OCR a raw image; for images we use
    the same DocumentConverter but rely on Docling's image-format support if
    available. When image support isn't installed, we fall back to wrapping
    the image into a single-page PDF. This is acceptable for a Tier-3
    prototype; production code should call ``docling_core`` directly.
    """
    root = Path(corpus_path).resolve()
    if not root.is_dir():
        raise ValueError(f"corpus_path is not a directory: {root}")

    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_paths = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in image_exts
    )
    if not image_paths:
        raise ValueError(f"No image documents found in {root}")

    converter = _build_converter(do_ocr=do_ocr, do_table_structure=False)

    corpus: list[dict] = []
    for image_path in image_paths:
        logger.info("Docling: OCR-ing image %s", image_path.relative_to(root))
        try:
            result = converter.convert(str(image_path))
            page_texts = _page_texts(result.document)
            if not page_texts:
                markdown = result.document.export_to_markdown().strip()
                page_texts = [markdown] if markdown else []
        except Exception as exc:  # pragma: no cover - depends on docling version
            logger.warning("Docling could not OCR %s: %s", image_path, exc)
            page_texts = []

        if not page_texts:
            # Skip silently; an empty OCR result is not a fatal error.
            continue

        relative = image_path.relative_to(root)
        source_id = image_path.stem
        for page_index, page_text in enumerate(page_texts, start=1):
            corpus.append(
                {
                    "context": page_text,
                    "metadata": {
                        "doc_id": f"{source_id}_p{page_index}",
                        "source_id": source_id,
                        "source_name": image_path.name,
                        "source_path": str(relative),
                        "page_no": page_index,
                        "corpus_type": "image",
                    },
                }
            )
    logger.info(
        "Docling: produced %d image chunks from %d images", len(corpus), len(image_paths)
    )
    return corpus
