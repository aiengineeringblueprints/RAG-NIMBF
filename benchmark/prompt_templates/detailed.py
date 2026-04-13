"""Detailed prompt template — full sentence answers with reasoning."""

from benchmark.prompt_templates.types import PromptTemplate

SYSTEM_PROMPT = (
    "Answer the question based only on the provided context. "
    "Provide a clear, complete answer in full sentences. "
    "Include relevant calculations or reasoning steps when applicable."
)

HUMAN_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

DETAILED = PromptTemplate(
    name="detailed",
    system_prompt=SYSTEM_PROMPT,
    human_template=HUMAN_TEMPLATE,
)
