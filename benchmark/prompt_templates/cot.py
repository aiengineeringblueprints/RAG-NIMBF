"""Chain-of-thought prompt template — step-by-step reasoning, then a FINAL answer line."""

from benchmark.prompt_templates.types import PromptTemplate

SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context.\n"
    "RULES:\n"
    "1. First, reason step by step: identify the relevant facts in the context "
    "and explain how they lead to the answer. Keep the reasoning brief.\n"
    "2. On the LAST line, write EXACTLY 'FINAL: ' followed by a short, direct, "
    "factual answer — one sentence maximum.\n"
    "3. If the context does not contain the answer, write "
    "'FINAL: I cannot answer from the provided context.'\n"
    "\n"
    "Example:\n"
    "The context states that the Star Wars score was composed by John Williams.\n"
    "FINAL: John Williams."
)

HUMAN_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\n\nReasoning:"

COT = PromptTemplate(
    name="cot",
    system_prompt=SYSTEM_PROMPT,
    human_template=HUMAN_TEMPLATE,
    final_answer_line=True,
)
