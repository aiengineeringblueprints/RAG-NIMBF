"""Contracts for black-box document parser integrations (ADR-005)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ParsedPage:
    """One parsed page: Markdown plus optional structured blocks."""

    page_number: int
    markdown: str
    blocks: tuple[dict[str, Any], ...] | None = None


@dataclass(frozen=True)
class ParseResult:
    """Normalized output of parsing one document.

    ``pages`` carries per-page Markdown.  Structured blocks are optional:
    Markdown-only parsers are fully supported and simply leave ``blocks``
    unset on every page.
    """

    document_id: str
    pages: tuple[ParsedPage, ...] = ()
    parser_name: str = ""
    parser_version: str | None = None
    total_seconds: float = 0.0
    raw_response: Any = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def markdown(self) -> str:
        """Return the full-document Markdown (pages joined in order)."""
        return "\n\n".join(page.markdown for page in self.pages)


class DocumentParser(Protocol):
    """A black-box document parser that turns documents into Markdown."""

    @property
    def name(self) -> str:
        """Registry-facing parser identity."""
        ...

    @property
    def parser_version(self) -> str | None:
        """Optional parser version pinned into run metadata."""
        ...

    def parse(self, document: dict, config: Any = None) -> ParseResult:
        """Parse one document and return per-page Markdown.

        ``document`` shape: ``{"document_id": str, "pages": [{"page_number":
        int, "image_base64": str | None, "image_media_type": str,
        "text": str | None}], "metadata": {...}}``.
        """
        ...
