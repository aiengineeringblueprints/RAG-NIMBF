"""Concise prompt template — short factual answers, matching WikiQA ground truth style."""

from benchmark.prompt_templates.types import PromptTemplate

SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context.\n"
    "RULES:\n"
    "- Give a short, direct, factual answer — one sentence maximum.\n"
    "- Do NOT explain your reasoning, add context, or use markdown formatting.\n"
    "- Do NOT say 'Based on the context' or similar preamble.\n"
    "- If the context does not contain the answer, say 'I cannot answer from the provided context.'\n"
    "Examples:\n"
    "Q: What circuit court is Maryland? → The Circuit Courts of Maryland are the state trial courts of general jurisdiction.\n"
    "Q: How many members are in the House of Representatives? → The total number of voting representatives is fixed at 435.\n"
    "Q: Who wrote the music for Star Wars? → John Williams."
)

HUMAN_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

CONCISE = PromptTemplate(
    name="concise",
    system_prompt=SYSTEM_PROMPT,
    human_template=HUMAN_TEMPLATE,
)
