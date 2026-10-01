"""Question-first prompt template — question before and after the context (sandwich)."""

from benchmark.prompt_templates.concise import SYSTEM_PROMPT
from benchmark.prompt_templates.types import PromptTemplate

# Same rules as ``concise`` so only the question placement differs.
HUMAN_TEMPLATE = (
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Using the context above, answer the question: {question}\n\n"
    "Answer:"
)

QUESTION_FIRST = PromptTemplate(
    name="question_first",
    system_prompt=SYSTEM_PROMPT,
    human_template=HUMAN_TEMPLATE,
)
