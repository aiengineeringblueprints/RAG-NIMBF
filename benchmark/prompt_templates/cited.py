"""Cited prompt template — numbered context chunks, inline [n] citations per claim."""

from benchmark.prompt_templates.types import PromptTemplate

SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context. "
    "The context consists of numbered passages like [1], [2].\n"
    "RULES:\n"
    "- Give a short, direct, factual answer — one or two sentences.\n"
    "- After every claim, cite the supporting passage number(s) in square "
    "brackets, e.g. 'John Williams composed the score [2].'\n"
    "- Only make claims that a cited passage supports.\n"
    "- Do NOT say 'Based on the context' or similar preamble.\n"
    "- If no passage contains the answer, say 'I cannot answer from the provided context.'"
)

HUMAN_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

CITED = PromptTemplate(
    name="cited",
    system_prompt=SYSTEM_PROMPT,
    human_template=HUMAN_TEMPLATE,
    number_contexts=True,
    strip_citations=True,
)
