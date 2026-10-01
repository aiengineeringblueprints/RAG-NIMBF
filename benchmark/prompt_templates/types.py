"""PromptTemplate dataclass — kept separate to avoid circular imports."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """A generation prompt template with system and human message parts.

    Attributes
    ----------
    name:
        Short identifier used in config and MLflow tags (e.g. "concise").
    system_prompt:
        The system message instructing the model how to answer.
    human_template:
        Template for the human message. Must contain ``{question}``; omits
        ``{context}`` only for closed-book baselines.
    final_answer_line:
        The model ends with a ``FINAL: <answer>`` line; only that value is scored.
    number_contexts:
        Prefix each retrieved chunk with ``[n]`` so the model can cite it.
    strip_citations:
        Remove ``[n]`` citation markers from the answer before scoring.
    """

    name: str
    system_prompt: str
    human_template: str
    final_answer_line: bool = False
    number_contexts: bool = False
    strip_citations: bool = False
