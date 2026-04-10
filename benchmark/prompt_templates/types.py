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
        Template for the human message. Must contain ``{context}`` and
        ``{question}`` placeholders.
    """

    name: str
    system_prompt: str
    human_template: str
