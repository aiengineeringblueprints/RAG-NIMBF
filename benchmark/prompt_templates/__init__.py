"""Prompt template registry for the RAG benchmark generation step."""

from benchmark.prompt_templates.types import PromptTemplate
from benchmark.prompt_templates.abstain_strict import ABSTAIN_STRICT
from benchmark.prompt_templates.chain_of_note import CHAIN_OF_NOTE
from benchmark.prompt_templates.cited import CITED
from benchmark.prompt_templates.concise import CONCISE
from benchmark.prompt_templates.cot import COT
from benchmark.prompt_templates.detailed import DETAILED
from benchmark.prompt_templates.finqa import FINQA
from benchmark.prompt_templates.minimal import MINIMAL
from benchmark.prompt_templates.no_context import NO_CONTEXT
from benchmark.prompt_templates.question_first import QUESTION_FIRST

BUILTIN_TEMPLATES: dict[str, PromptTemplate] = {
    t.name: t
    for t in (
        CONCISE, DETAILED, FINQA,
        MINIMAL, NO_CONTEXT, COT, CITED,
        ABSTAIN_STRICT, QUESTION_FIRST, CHAIN_OF_NOTE,
    )
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
