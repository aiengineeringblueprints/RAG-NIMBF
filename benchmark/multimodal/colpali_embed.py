"""ColPali visual embedder — multi-vector page embeddings + ColBERT max-sim.

Unlike the text pipeline (one vector per chunk in Chroma), ColPali produces a
*matrix* of token-level embeddings per page. We store each matrix as a JSON
blob in a dedicated collection and retrieve via ColBERT-style max-sim.

This is a Tier-3 functional prototype. The ColPali dependency
(``colpali-engine``) is heavy (pulls torch + transformers + a ColQwen2 model
download). If it isn't installed, ``build_colpali_visual_embedder`` raises
:class:`OptionalDependencyError`. Tests mock the heavy calls; the interface
itself is fully implemented so downstream code can be wired up against it.

Storage / retrieval contract
----------------------------
``build_colpali_visual_embedder`` returns an object with:

- ``embed_documents(page_paths) -> list[MultiVectorChunk]``
- ``embed_query(text) -> torch.Tensor``  (one query embedding matrix)
- ``build_index(chunks) -> ColPaliIndex``  (in-memory; persists as JSON)
- ``ColPaliIndex.load(path)`` / ``ColPaliIndex.save(path)``
- ``ColPaliIndex.retrieve(query_embedding, top_k) -> list[(chunk, score)]``

The index uses brute-force max-sim — fine for prototype-scale corpora.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from benchmark.multimodal.registry import OptionalDependencyError

logger = logging.getLogger(__name__)

_FEATURE = "colpali_visual"
_COLPALI_PACKAGE = "colpali-engine"
_COLPALI_EXTRA = "colpali-engine"
_DEFAULT_MODEL = "vidore/colqwen2-v1.0"


# ---------------------------------------------------------------------------
# Optional-dependency gate
# ---------------------------------------------------------------------------


def _require_colpali() -> tuple[Any, Any, Any]:
    """Lazy-import ColQwen2 model + processor, raise if unavailable."""
    try:
        import torch  # noqa: F401  (presence check)
        from colpali_engine.models import ColQwen2, ColQwen2Processor  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OptionalDependencyError(
            feature=_FEATURE,
            package=_COLPALI_PACKAGE,
            extra=_COLPALI_EXTRA,
        ) from exc
    return torch, ColQwen2, ColQwen2Processor


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OptionalDependencyError(
            feature=_FEATURE,
            package="torch",
            extra="torch",
        ) from exc
    return torch


def _pdf_to_page_images(pdf_path: Path) -> list[Any]:
    """Render a PDF to a list of PIL images (one per page).

    Uses ``pdf2image`` if available; falls back to ``fitz`` (PyMuPDF). Both are
    optional — ColPali users typically have one installed.
    """
    try:
        from pdf2image import convert_from_path  # type: ignore[import-not-found]
        return convert_from_path(str(pdf_path))
    except ImportError:
        pass
    try:
        import fitz  # type: ignore[import-not-found]  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=120)
            mode = "rgb" if pix.n < 4 else "rgba"
            images.append(_pixmap_to_pil(pix, mode))
        doc.close()
        return images
    except ImportError as exc:
        raise OptionalDependencyError(
            feature=_FEATURE,
            package="pdf2image or PyMuPDF",
            extra="pdf2image",
        ) from exc


def _pixmap_to_pil(pix: Any, mode: str) -> Any:
    from PIL import Image  # type: ignore[import-not-found]
    return Image.frombytes(mode, [pix.width, pix.height], pix.samples)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class MultiVectorChunk:
    """One ColPali page embedding + the metadata to surface at retrieval time."""

    doc_id: str
    page_no: int
    source_name: str
    source_path: str
    # Token-level embeddings: list[list[float]] with shape (n_tokens, dim).
    # JSON-serialisable so we can persist as a blob.
    token_vectors: list[list[float]]
    # Optional text fallback (e.g. Docling OCR of the same page) used when the
    # generator needs text context.
    page_text: str = ""

    def to_json(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "page_no": self.page_no,
            "source_name": self.source_name,
            "source_path": self.source_path,
            "token_vectors": self.token_vectors,
            "page_text": self.page_text,
        }

    @classmethod
    def from_json(cls, data: dict) -> "MultiVectorChunk":
        return cls(
            doc_id=str(data["doc_id"]),
            page_no=int(data["page_no"]),
            source_name=str(data["source_name"]),
            source_path=str(data["source_path"]),
            token_vectors=[list(v) for v in data["token_vectors"]],
            page_text=str(data.get("page_text", "")),
        )


# ---------------------------------------------------------------------------
# ColBERT-style max-sim index (brute-force, in-memory, JSON-persisted)
# ---------------------------------------------------------------------------


@dataclass
class ColPaliIndex:
    """In-memory multi-vector index with brute-force max-sim retrieval.

    Scale: prototype-friendly (hundreds to low thousands of pages). For larger
    corpora swap in a dedicated multi-vector ANN (e.g. Vespa, Milvus) behind
    this interface.
    """

    chunks: list[MultiVectorChunk] = field(default_factory=list)

    def add(self, chunk: MultiVectorChunk) -> None:
        self.chunks.append(chunk)

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chunks": [c.to_json() for c in self.chunks]}
        with out.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp)

    @classmethod
    def load(cls, path: str | Path) -> "ColPaliIndex":
        with Path(path).open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return cls(chunks=[MultiVectorChunk.from_json(c) for c in payload["chunks"]])

    def retrieve(
        self,
        query_embedding: Any,
        top_k: int = 5,
    ) -> list[tuple[MultiVectorChunk, float]]:
        """Return top-k ``(chunk, score)`` pairs via ColBERT max-sim.

        ``query_embedding`` may be either a ``torch.Tensor`` of shape
        ``(q_tokens, dim)`` or a plain ``list[list[float]]``.
        """
        if not self.chunks:
            return []

        torch = _require_torch()
        q = _as_tensor(torch, query_embedding)  # (q_tokens, dim)

        scored: list[tuple[MultiVectorChunk, float]] = []
        for chunk in self.chunks:
            doc = torch.tensor(chunk.token_vectors, dtype=q.dtype)  # (d_tokens, dim)
            if doc.numel() == 0:
                scored.append((chunk, 0.0))
                continue
            # ColBERT max-sim: for each query token, take the max similarity
            # over doc tokens; sum across query tokens.
            sim = torch.matmul(q, doc.t())  # (q_tokens, d_tokens)
            max_per_query_token = sim.max(dim=1).values
            score = float(max_per_query_token.sum().item())
            scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def _as_tensor(torch: Any, embedding: Any) -> Any:
    if isinstance(embedding, torch.Tensor):
        return embedding
    return torch.tensor(embedding, dtype=torch.float32)


# ---------------------------------------------------------------------------
# ColPali embedder (the object registered as ``colpali_visual``)
# ---------------------------------------------------------------------------


class ColPaliVisualEmbedder:
    """Wraps a ColQwen2 model/processor and exposes embed + retrieve helpers."""

    def __init__(
        self,
        *,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
        torch_dtype: str | None = None,
    ) -> None:
        torch, ColQwen2, ColQwen2Processor = _require_colpali()
        self.torch = torch
        self.model_name = model_name
        self.device = device

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            None: torch.float32,
        }
        torch_dtype_resolved = dtype_map.get(torch_dtype, torch.float32)

        logger.info(
            "ColPali: loading ColQwen2 '%s' on %s (%s)",
            model_name,
            device,
            torch_dtype or "float32",
        )
        self.model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch_dtype_resolved,
            device_map=device,
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(model_name)

    # -- embedding ----------------------------------------------------------

    def embed_images(self, images: Iterable[Any]) -> Any:
        """Embed an iterable of PIL images. Returns a list of (n_tokens, dim) tensors."""
        images = list(images)
        batch = self.processor.process_images(images).to(self.model.device)
        with self.torch.no_grad():
            embeddings = self.model(**batch)
        # Each entry in ``embeddings`` is one page's token matrix.
        return [e.cpu() for e in embeddings]

    def embed_query(self, text: str) -> Any:
        batch = self.processor.process_queries([text]).to(self.model.device)
        with self.torch.no_grad():
            embedding = self.model(**batch)
        return embedding[0].cpu()  # (q_tokens, dim)

    def embed_documents(
        self,
        page_specs: list[dict],
    ) -> list[MultiVectorChunk]:
        """Embed a list of ``{path, page_no, source_name, source_path, doc_id,
        page_text?}`` specs into MultiVectorChunks.

        ``path`` may point at an image (PNG/JPG) or a PDF. PDFs are rasterised
        page-by-page; the caller may pre-specify which page via ``page_no`` or
        leave it to be assigned sequentially.
        """
        chunks: list[MultiVectorChunk] = []
        for spec in page_specs:
            path = Path(spec["path"])
            if path.suffix.lower() == ".pdf":
                page_images = _pdf_to_page_images(path)
            else:
                from PIL import Image  # type: ignore[import-not-found]
                page_images = [Image.open(path)]

            embeddings = self.embed_images(page_images)
            for idx, (emb, _img) in enumerate(zip(embeddings, page_images)):
                page_no = int(spec.get("page_no", idx + 1))
                token_vectors = emb.tolist()
                chunks.append(
                    MultiVectorChunk(
                        doc_id=f"{spec['doc_id']}_p{page_no}",
                        page_no=page_no,
                        source_name=str(spec.get("source_name", path.name)),
                        source_path=str(spec.get("source_path", path)),
                        token_vectors=token_vectors,
                        page_text=str(spec.get("page_text", "")),
                    )
                )
        return chunks

    def build_index(self, chunks: list[MultiVectorChunk]) -> ColPaliIndex:
        return ColPaliIndex(chunks=list(chunks))


# ---------------------------------------------------------------------------
# Factory (registered as the ``colpali_visual`` embedder)
# ---------------------------------------------------------------------------


def build_colpali_visual_embedder(**kwargs: Any) -> ColPaliVisualEmbedder:
    """Construct a ColPaliVisualEmbedder from kwargs (lazy heavy imports).

    Raises :class:`OptionalDependencyError` if ``colpali-engine`` / ``torch``
    are not installed. Tests mock this entry point.
    """
    return ColPaliVisualEmbedder(**kwargs)
