"""Prompt template registry for the RAG benchmark generation step."""

from benchmark.prompt_templates.types import PromptTemplate
from benchmark.prompt_templates.concise import CONCISE
from benchmark.prompt_templates.detailed import DETAILED

BUILTIN_TEMPLATES: dict[str, PromptTemplate] = {
    t.name: t for t in (CONCISE, DETAILED)
}


def get_template(name: str) -> PromptTemplate:
    """Look up a prompt template by name.

    Raises ``KeyError`` if *name* is not in the built-in registry.
    """
    if name not in BUILTIN_TEMPLATES:
        available = ", ".join(sorted(BUILTIN_TEMPLATES))
        raise KeyError(
            f"Unknown prompt template '{name}'. Available: {available}"
        )
    return BUILTIN_TEMPLATES[name]
