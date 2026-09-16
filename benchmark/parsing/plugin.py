"""In-process Python-plugin parser flavor."""

from __future__ import annotations

import importlib
from typing import Any

from benchmark.parsing.base import DocumentParser


def load_parser_class(
    module_name: str, attribute: str, config: Any = None
) -> DocumentParser:
    """Import ``module_name`` and instantiate its parser class.

    The class receives the benchmark config as an optional constructor
    argument, mirroring the RAG plugin convention.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"Cannot import parser plugin module {module_name!r}: {exc}"
        ) from exc
    cls = getattr(module, attribute, None)
    if cls is None:
        raise ValueError(
            f"Parser plugin module {module_name!r} has no attribute "
            f"{attribute!r}"
        )
    try:
        return cls(config)
    except TypeError:
        return cls()
