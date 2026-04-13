"""FinQA prompt template — derivation trace + final value for financial Q&A."""

from benchmark.prompt_templates.types import PromptTemplate

SYSTEM_PROMPT = (
    "You are a financial data analyst. Answer the question using ONLY the provided context.\n"
    "RULES:\n"
    "1. First, write ONE short line showing which values from the context you used "
    "and the calculation. Example: 'Liability 2014: 494 minus Liability 2013: 0 = 494'\n"
    "2. On the LAST line, write EXACTLY 'FINAL: ' followed by the raw answer value.\n"
    "3. For percentage questions, the FINAL value must be a DECIMAL fraction.\n"
    "   If the answer is 12%, write FINAL: 0.12. If -46.8%, write FINAL: -0.468.\n"
    "4. For yes/no questions, write FINAL: 1 (yes) or FINAL: 0 (no).\n"
    "5. For absolute values, write the raw number. E.g. FINAL: 494.0 or FINAL: 100000000.0\n"
    "6. Preserve negative signs when the value decreased.\n"
    "7. Do NOT include units, % signs, or $ signs in the FINAL value.\n"
    "\n"
    "Examples:\n"
    "Liability changed from 0 in 2013 to 494 in 2014.\n"
    "FINAL: 494.0\n"
    "\n"
    "Euro impact 3.5 divided by total 13.0 equals 0.269.\n"
    "FINAL: 0.269\n"
    "\n"
    "Operating profit 473 / net sales 7170 = 0.066.\n"
    "FINAL: 0.066"
)

HUMAN_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

FINQA = PromptTemplate(
    name="finqa",
    system_prompt=SYSTEM_PROMPT,
    human_template=HUMAN_TEMPLATE,
)
