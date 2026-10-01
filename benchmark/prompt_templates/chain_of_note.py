"""Chain-of-Note prompt template — per-passage relevance notes, then a FINAL answer line."""

from benchmark.prompt_templates.types import PromptTemplate

SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context. "
    "The context consists of numbered passages like [1], [2].\n"
    "RULES:\n"
    "1. For each passage, write ONE short note: 'Note [n]: <relevant fact>' "
    "or 'Note [n]: irrelevant'.\n"
    "2. On the LAST line, write EXACTLY 'FINAL: ' followed by a short, direct, "
    "factual answer — one sentence maximum — based only on the relevant notes.\n"
    "3. If no passage is relevant, write "
    "'FINAL: I cannot answer from the provided context.'\n"
    "\n"
    "Example:\n"
    "Note [1]: irrelevant\n"
    "Note [2]: John Williams composed the Star Wars score.\n"
    "FINAL: John Williams."
)

HUMAN_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\n\nNotes:"

CHAIN_OF_NOTE = PromptTemplate(
    name="chain_of_note",
    system_prompt=SYSTEM_PROMPT,
    human_template=HUMAN_TEMPLATE,
    final_answer_line=True,
    number_contexts=True,
)
