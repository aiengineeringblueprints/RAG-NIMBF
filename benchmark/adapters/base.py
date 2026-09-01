"""Common contracts for black-box RAG system integrations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from benchmark.adapters.components import ComponentBundle


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
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterCapabilities:
    """Features a managed RAG adapter can guarantee to the benchmark.

    Capabilities are deliberately explicit.  A requested experiment must not
    silently assume that a provider applied an embedding, chunking, or
    reranking setting that its API cannot control.
    """

    ingestion: bool = False
    retrieval: bool = False
    generation: bool = True
    chunk_configuration: bool = False
    embedding_configuration: bool = False
    reranking: bool = False
    references: bool = False
    token_usage: bool = False
    cleanup: bool = False


@dataclass(frozen=True)
class PreparedTarget:
    """Provider resources prepared for one concrete benchmark configuration."""

    target_id: str | None = None
    dataset_ids: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    """Provider-independent representation of one ranked evidence chunk."""

    text: str
    rank: int
    chunk_id: str | None = None
    document_id: str | None = None
    source_id: str | None = None
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, Any]:
        """Return metadata in the shape consumed by existing metrics."""
        result = dict(self.metadata)
        if self.chunk_id is not None:
            result.setdefault("chunk_id", self.chunk_id)
        if self.document_id is not None:
            result.setdefault("document_id", self.document_id)
        if self.source_id is not None:
            result.setdefault("source_id", self.source_id)
            # Existing gold-document metrics use ``doc_id``. Prefer the
            # caller-stable source ID over a provider-generated document ID.
            result.setdefault("doc_id", self.source_id)
        elif self.document_id is not None:
            result.setdefault("doc_id", self.document_id)
        if self.score is not None:
            result.setdefault("score", self.score)
        result.setdefault("rank", self.rank)
        return result


@dataclass(frozen=True)
class RetrievalResult:
    """Normalized result of a retrieval-only provider operation."""

    chunks: tuple[RetrievedChunk, ...] = ()
    total_seconds: float = 0.0
    raw_response: Any = None

    @property
    def contexts(self) -> list[str]:
        return [chunk.text for chunk in self.chunks]

    @property
    def metadata(self) -> list[dict[str, Any]]:
        return [chunk.as_metadata() for chunk in self.chunks]


@dataclass(frozen=True)
class AdapterGenerationResult:
    """Normalized managed-provider generation result.

    ``retrieval`` carries the exact evidence used for generation.  This keeps
    retrieval evaluation possible without encoding provider-specific response
    objects in the benchmark core.
    """

    answer: str
    retrieval: RetrievalResult = field(default_factory=RetrievalResult)
    raw_response: Any = None
    ttft_seconds: float = 0.0
    total_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None
    gpu_usage: dict[str, float] | None = None
    raw_content: str = ""
    raw_reasoning: str | None = None
    answer_valid: bool = True
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_legacy_output(self) -> RagSystemOutput:
        """Convert to the original answer contract used by reporting code."""
        token_count = self.output_tokens
        tokens_per_second = (
            token_count / self.total_seconds
            if token_count and self.total_seconds > 0
            else 0.0
        )
        return RagSystemOutput(
            answer=self.answer,
            contexts=self.retrieval.contexts,
            metadata=self.retrieval.metadata,
            raw_response=(
                self.raw_response if isinstance(self.raw_response, dict) else None
            ),
            ttft_seconds=self.ttft_seconds,
            total_seconds=self.total_seconds,
            token_count=token_count,
            tokens_per_second=tokens_per_second,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens or self.input_tokens + self.output_tokens,
            estimated_cost_usd=self.estimated_cost_usd,
            gpu_usage=self.gpu_usage,
            raw_content=self.raw_content or self.answer,
            raw_reasoning=self.raw_reasoning,
            answer_valid=self.answer_valid and bool(self.answer.strip()),
            diagnostics=dict(self.diagnostics),
        )


class RagSystemAdapter(Protocol):
    """A black-box RAG system that can answer benchmark samples.

    Two optional methods support component injection. Adapters that do not
    implement them are treated as pure black-box — the Framework will not
    attempt to inject components.
    """

    name: str

    def supports_components(self) -> dict[str, bool]:
        """Declare which ComponentBundle slots this adapter accepts.

        Keys: any of {chunker, embedder, retriever, reranker, llm, prompt}.
        Value True means the adapter will use a Framework-provided slot if
        given one. Method is optional; absence = accepts nothing.
        """
        ...

    def set_components(self, bundle: "ComponentBundle") -> None:
        """Receive Framework-built components. Called once before prepare().

        Optional; absence = pure black-box adapter.
        """
        ...

    def prepare(
        self,
        config: Any,
        data: list[dict],
        corpus: list[dict] | None = None,
    ) -> None:
        """Prepare the system before per-sample queries run."""

    def answer(self, sample: dict, config: Any) -> RagSystemOutput:
        """Return a normalized answer and optional retrieval evidence."""


class ManagedRagSystemAdapter(Protocol):
    """Full lifecycle contract for configurable external RAG systems."""

    name: str

    def capabilities(self) -> AdapterCapabilities:
        """Declare provider features before resources are created."""
        ...

    def prepare(
        self,
        config: Any,
        data: list[dict],
        corpus: list[dict] | None = None,
    ) -> PreparedTarget:
        """Create or resolve resources used by the rest of this run."""
        ...

    def retrieve(
        self,
        target: PreparedTarget,
        sample: dict,
        config: Any,
    ) -> RetrievalResult:
        """Retrieve normalized ranked evidence for one sample."""
        ...

    def generate(
        self,
        target: PreparedTarget,
        sample: dict,
        config: Any,
        retrieval: RetrievalResult | None = None,
    ) -> AdapterGenerationResult:
        """Generate one answer, optionally using a prior retrieval result."""
        ...

    def cleanup(self, target: PreparedTarget, config: Any) -> None:
        """Release ephemeral provider resources when the run owns them."""
        ...
