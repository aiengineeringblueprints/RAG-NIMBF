"""Registry for multi-modal chunker / embedder names.

The benchmark's text pipeline routes through ``benchmark.chunking.get_chunker``
and ``benchmark.embedding.get_embedding_model``. Multi-modal ingestion needs
different machinery (Docling / ColPali / Whisper), so we keep a parallel
registry here and expose helpers that the text-side code can call to decide
whether a given strategy name belongs to the multi-modal world.

Heavy ML deps are imported lazily inside the factory callables, never at
registry-construction time.
"""

from __future__ import annotations

from typing import Callable

# ---------------------------------------------------------------------------
# Optional-dependency error
# ---------------------------------------------------------------------------


class OptionalDependencyError(ImportError):
    """Raised when a multi-modal code path needs a dep that isn't installed.

    Carries the ``feature`` name and the suggested ``pip_install`` command so
    callers can surface a helpful message in logs / CLI output.
    """

    def __init__(self, feature: str, package: str, extra: str | None = None) -> None:
        install_target = extra if extra is not None else package
        hint = (
            f"This benchmark path needs the optional dependency '{package}'. "
            f"Install it with:  pip install {install_target}"
        )
        super().__init__(f"[{feature}] {hint}")
        self.feature = feature
        self.package = package
        self.install_target = install_target


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

# A multimodal chunker factory turns a corpus path + kwargs into a list of
# ``{context, metadata}`` dicts — the same shape the text pipeline expects
# from a corpus loader. The factory is invoked lazily.
MultimodalChunkerFactory = Callable[..., list[dict]]


# A multimodal embedder factory returns an opaque object whose only contract
# is ``embed_documents(paths) -> list[MultiVectorChunk]`` and a ``retrieve``
# helper. Currently only ``colpali_visual`` registers here.
MultimodalEmbedderFactory = Callable[..., object]


MULTIMODAL_CHUNKERS: dict[str, MultimodalChunkerFactory] = {}
MULTIMODAL_EMBEDDERS: dict[str, MultimodalEmbedderFactory] = {}


def register_multimodal_chunker(name: str, factory: MultimodalChunkerFactory) -> None:
    """Register a multimodal chunker factory under ``name``."""
    MULTIMODAL_CHUNKERS[name] = factory


def register_multimodal_embedder(name: str, factory: MultimodalEmbedderFactory) -> None:
    """Register a multimodal embedder factory under ``name``."""
    MULTIMODAL_EMBEDDERS[name] = factory


def available_multimodal_chunkers() -> tuple[str, ...]:
    return tuple(sorted(MULTIMODAL_CHUNKERS))


def available_multimodal_embedders() -> tuple[str, ...]:
    return tuple(sorted(MULTIMODAL_EMBEDDERS))


def is_chunker_multimodal(strategy: str) -> bool:
    return strategy in MULTIMODAL_CHUNKERS


def is_embedder_multimodal(embedder: str) -> bool:
    return embedder in MULTIMODAL_EMBEDDERS


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------
# Imports below are safe — they only register factory *callables*, they do
# not import docling/colpali/whisper at module load. The factories themselves
# defer heavy imports until invocation.

from benchmark.multimodal.docling_ocr import (  # noqa: E402
    ingest_pdf_via_docling,
    ingest_image_via_docling,
)
from benchmark.multimodal.whisper_asr import ingest_audio_via_whisper  # noqa: E402
from benchmark.multimodal.colpali_embed import (  # noqa: E402
    build_colpali_visual_embedder,
)


register_multimodal_chunker("pdf_ocr_docling", ingest_pdf_via_docling)
register_multimodal_chunker("image_ocr_docling", ingest_image_via_docling)
register_multimodal_chunker("audio_whisper", ingest_audio_via_whisper)


register_multimodal_embedder("colpali_visual", build_colpali_visual_embedder)
