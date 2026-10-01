"""Strict-abstention prompt template — refuse unless the context fully supports the answer."""

from benchmark.prompt_templates.types import PromptTemplate

SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context.\n"
    "RULES:\n"
    "- First decide whether the context contains enough information to answer "
    "the question completely. Do NOT use outside knowledge, and do NOT guess "
    "from partially related passages.\n"
    "- If the context is sufficient, give a short, direct, factual answer — "
    "one sentence maximum, without preamble or explanation.\n"
    "- If the context is insufficient, answer EXACTLY: "
    "'I cannot answer from the provided context.'\n"
    "Examples:\n"
    "Context: John Williams composed the music for Star Wars.\n"
    "Q: Who wrote the music for Star Wars? → John Williams.\n"
    "Context: Star Wars was released in 1977 and directed by George Lucas.\n"
    "Q: Who wrote the music for Star Wars? → I cannot answer from the provided context."
)

HUMAN_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

ABSTAIN_STRICT = PromptTemplate(
    name="abstain_strict",
    system_prompt=SYSTEM_PROMPT,
    human_template=HUMAN_TEMPLATE,
)
