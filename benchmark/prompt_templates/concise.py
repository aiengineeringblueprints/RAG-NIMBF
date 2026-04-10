"""Concise prompt template — raw values only, matching FinQA ground truth style."""

from benchmark.prompt_templates.types import PromptTemplate

SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context. "
    "Return ONLY the raw value — a number, percentage, ratio, or yes/no. "
    "Do NOT include units, explanations, reasoning, or full sentences. "
    "Examples: 494.0 | 0.12 | -0.46 | 1 | 5.8"
)

HUMAN_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

CONCISE = PromptTemplate(
    name="concise",
    system_prompt=SYSTEM_PROMPT,
    human_template=HUMAN_TEMPLATE,
)
