"""Common contracts for black-box RAG system integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RagSystemOutput:
    """Normalized output returned by any benchmarked RAG system."""

    answer: str
    contexts: list[str] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)
    raw_response: dict[str, Any] | None = None
    ttft_seconds: float = 0.0
    total_seconds: float = 0.0
    token_count: int = 0
    tokens_per_second: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None
    gpu_usage: dict[str, float] | None = None
    raw_content: str = ""
    raw_reasoning: str | None = None
    answer_valid: bool = True


class RagSystemAdapter(Protocol):
    """A black-box RAG system that can answer benchmark samples."""

    name: str

    def prepare(
        self,
        config: Any,
        data: list[dict],
        corpus: list[dict] | None = None,
    ) -> None:
        """Prepare the system before per-sample queries run."""

    def answer(self, sample: dict, config: Any) -> RagSystemOutput:
        """Return a normalized answer and optional retrieval evidence."""
