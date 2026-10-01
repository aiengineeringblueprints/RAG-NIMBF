"""No-context prompt template — closed-book baseline; retrieved chunks are not shown."""

from benchmark.prompt_templates.types import PromptTemplate

SYSTEM_PROMPT = (
    "Answer the question from your own knowledge.\n"
    "RULES:\n"
    "- Give a short, direct, factual answer — one sentence maximum.\n"
    "- Do NOT explain your reasoning or use markdown formatting.\n"
    "- If you do not know the answer, say 'I cannot answer this question.'"
)

HUMAN_TEMPLATE = "Question: {question}\n\nAnswer:"

NO_CONTEXT = PromptTemplate(
    name="no_context",
    system_prompt=SYSTEM_PROMPT,
    human_template=HUMAN_TEMPLATE,
)
