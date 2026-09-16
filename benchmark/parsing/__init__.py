"""Document parser adapter registry (ADR-005).

Parsers are first-class citizens beside the RAG adapter registry:
register a factory by name with :func:`register_parser_adapter` and
resolve it from a benchmark config with :func:`get_parser_adapter`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from benchmark.parsing.alignment import (
    MATCH_ALGORITHM_LICENSE,
    MATCH_ALGORITHM_NAME,
    MATCH_ALGORITHM_SOURCE,
    MATCH_ALGORITHM_VERSION,
    AlignedPair,
    alignment_run_metadata,
    quick_match,
    split_paragraphs,
)
from benchmark.parsing.base import DocumentParser, ParsedPage, ParseResult
from benchmark.parsing.http import HttpParserAdapter
from benchmark.parsing.metrics import (
    PageTextMetrics,
    ParsingTextMetricsResult,
    cer,
    compute_aligned_parsing_text_metrics,
    compute_parsing_text_metrics,
    normalize_text,
    normalized_edit_distance,
    wer,
)
from benchmark.parsing.plugin import load_parser_class
from benchmark.parsing.table_metrics import (
    PageTableMetrics,
    ParsingTableMetricsResult,
    compute_parsing_table_metrics,
)
from benchmark.parsing.teds import (
    TEDS,
    TEDS_ALGORITHM_LICENSE,
    TEDS_ALGORITHM_NAME,
    TEDS_ALGORITHM_SOURCE,
    TEDS_ALGORITHM_VERSION,
    teds,
    teds_run_metadata,
    teds_structure_only,
)
from benchmark.parsing.tables import (
    extract_markdown_tables,
    markdown_tables_to_html,
)

ParserAdapterFactory = Callable[[Any], DocumentParser]

PARSER_ADAPTER_REGISTRY: dict[str, ParserAdapterFactory] = {}

__all__ = [
    "MATCH_ALGORITHM_LICENSE",
    "MATCH_ALGORITHM_NAME",
    "MATCH_ALGORITHM_SOURCE",
    "MATCH_ALGORITHM_VERSION",
    "PARSER_ADAPTER_REGISTRY",
    "TEDS",
    "TEDS_ALGORITHM_LICENSE",
    "TEDS_ALGORITHM_NAME",
    "TEDS_ALGORITHM_SOURCE",
    "TEDS_ALGORITHM_VERSION",
    "AlignedPair",
    "DocumentParser",
    "HttpParserAdapter",
    "PageTableMetrics",
    "PageTextMetrics",
    "ParseResult",
    "ParsedPage",
    "ParserAdapterFactory",
    "ParsingTableMetricsResult",
    "ParsingTextMetricsResult",
    "alignment_run_metadata",
    "cer",
    "compute_aligned_parsing_text_metrics",
    "compute_parsing_table_metrics",
    "compute_parsing_text_metrics",
    "extract_markdown_tables",
    "get_corpus_parser",
    "get_parser_adapter",
    "load_parser_class",
    "markdown_tables_to_html",
    "normalize_text",
    "normalized_edit_distance",
    "quick_match",
    "register_parser_adapter",
    "split_paragraphs",
    "teds",
    "teds_run_metadata",
    "teds_structure_only",
    "wer",
]


def register_parser_adapter(name: str, factory: ParserAdapterFactory) -> None:
    """Register a document parser factory by config name."""
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Parser adapter name must not be empty")
    PARSER_ADAPTER_REGISTRY[normalized_name] = factory


register_parser_adapter("http", HttpParserAdapter.from_config)


def get_corpus_parser(config: Any) -> DocumentParser | None:
    """Resolve the corpus-ingestion parser (OCR-07), or None when unset.

    Unlike :func:`get_parser_adapter`, this is registry-only: the
    ``corpus_parser`` knob names a registered parser adapter, and the
    parser_* settings (HTTP endpoint, version, ...) parameterize it.
    """
    corpus_parser_name = str(
        getattr(config, "corpus_parser", "") or ""
    ).strip().lower()
    if not corpus_parser_name:
        return None
    try:
        factory = PARSER_ADAPTER_REGISTRY[corpus_parser_name]
    except KeyError as exc:
        available = ", ".join(sorted(PARSER_ADAPTER_REGISTRY))
        raise ValueError(
            f"Unsupported CORPUS_PARSER={config.corpus_parser!r}. "
            f"Available: {available}"
        ) from exc
    return factory(config)


def get_parser_adapter(config: Any) -> DocumentParser | None:
    """Resolve the configured parser, or None when no parser is configured.

    An explicit plugin module/attribute pair takes precedence over the
    registry name, so local parser classes can be used without registry
    indirection.
    """
    plugin_module = getattr(config, "parser_plugin_module", None)
    plugin_attribute = getattr(config, "parser_plugin_attribute", None)
    if plugin_module:
        if not plugin_attribute:
            raise ValueError(
                "PARSER_PLUGIN_ATTRIBUTE is required when "
                "PARSER_PLUGIN_MODULE is set"
            )
        return load_parser_class(plugin_module, plugin_attribute, config)

    parser_name = str(getattr(config, "parser_adapter", "") or "").strip().lower()
    if not parser_name:
        return None
    try:
        factory = PARSER_ADAPTER_REGISTRY[parser_name]
    except KeyError as exc:
        available = ", ".join(sorted(PARSER_ADAPTER_REGISTRY))
        raise ValueError(
            f"Unsupported PARSER_ADAPTER={config.parser_adapter!r}. "
            f"Available: {available}"
        ) from exc
    return factory(config)
