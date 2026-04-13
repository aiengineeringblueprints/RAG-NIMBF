"""Concise prompt template — raw values only, matching FinQA ground truth style."""

from benchmark.prompt_templates.types import PromptTemplate

SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context.\n"
    "RULES:\n"
    "- Return a single number, nothing else — no units, no % sign, no explanation.\n"
    "- For percentage questions, return the DECIMAL fraction, not the percentage. "
    "E.g. if the answer is 12%, return 0.12. If -46.8%, return -0.468.\n"
    "- For yes/no questions, return 1 for yes, 0 for no.\n"
    "- For absolute values, return the raw number. E.g. 494.0 or 100000000.0\n"
    "- Preserve negative signs when the value decreased.\n"
    "Examples: 494.0 | 0.12 | -0.468 | 1 | 148.36 | 100000000.0"
)

HUMAN_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

CONCISE = PromptTemplate(
    name="concise",
    system_prompt=SYSTEM_PROMPT,
    human_template=HUMAN_TEMPLATE,
)
