"""Minimal prompt template — bare instruction baseline without rules or examples."""

from benchmark.prompt_templates.types import PromptTemplate

SYSTEM_PROMPT = "Answer the question using the provided context."

HUMAN_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

MINIMAL = PromptTemplate(
    name="minimal",
    system_prompt=SYSTEM_PROMPT,
    human_template=HUMAN_TEMPLATE,
)
