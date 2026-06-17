"""Framework-built components offered to external RAG adapters via injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING

try:
    from benchmark.chunking import get_chunker
    from benchmark.embedding import get_embedding_model
    from benchmark.generation import get_llm
    from benchmark.prompt_templates import get_template
    from benchmark.reranker import get_reranker
except ModuleNotFoundError:  # pragma: no cover - depends on optional deps
    # Lazy fallback so the module is importable in test envs without mlflow.
    get_chunker = None  # type: ignore[assignment]
    get_embedding_model = None  # type: ignore[assignment]
    get_llm = None  # type: ignore[assignment]
    get_template = None  # type: ignore[assignment]
    get_reranker = None  # type: ignore[assignment]

VALID_SLOTS: frozenset[str] = frozenset(
    {"chunker", "embedder", "retriever", "reranker", "llm", "prompt"}
)


@dataclass(frozen=True)
class ComponentBundle:
    """Optional LangChain-compatible components built from .env.

    All fields default to None. Adapters pick the slots they support; the
    rest stay None. None means "Framework did not build this slot" or
    "adapter declined injection via supports_components".

    Frozen so a bundle can be hashed/cached and so adapters cannot mutate
    the Framework-built instance — they receive it, store it, read it.
    """
    chunker: Optional[Any] = None  # langchain TextSplitter
    embedder: Optional[Any] = None  # langchain Embeddings
    retriever_factory: Optional[Callable[[list[dict]], Any]] = None
    reranker: Optional[Any] = None
    llm: Optional[Any] = None  # langchain BaseChatModel
    prompt_template: Optional[str] = None


if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.embeddings import Embeddings
    from langchain_text_splitters import TextSplitter


def _parse_accepts(raw: str) -> set[str]:
    """Parse comma-separated RAG_ADAPTER_ACCEPTS into a normalized set.

    Raises ValueError on unknown slot names to fail fast at config load.
    """
    if not raw or not raw.strip():
        return set()
    slots = {part.strip().lower() for part in raw.split(",") if part.strip()}
    unknown = slots - VALID_SLOTS
    if unknown:
        raise ValueError(
            f"unknown component slot(s): {sorted(unknown)}. "
            f"Valid: {sorted(VALID_SLOTS)}"
        )
    return slots


def build_components(config) -> ComponentBundle:
    """Build a ComponentBundle honoringings RAG_ADAPTER_ACCEPTS.

    Slots the user did not list stay None. If config.rag_adapter_accepts is
    empty, returns an empty bundle. Construction is kwargs-style so the
    bundle stays frozen.
    """
    accepts = _parse_accepts(getattr(config, "rag_adapter_accepts", ""))
    fields: dict[str, Any] = {}

    if "chunker" in accepts and getattr(config, "chunking_strategy", ""):
        fields["chunker"] = get_chunker(
            config.chunking_strategy,
            int(config.chunk_size or 0),
            int(config.chunk_overlap or 0),
        )

    if "embedder" in accepts and getattr(config, "embedding_model", ""):
        fields["embedder"] = get_embedding_model(
            config.embedding_model,
            config.embedding_base_url(),
            config.embedding_api_key(),
            provider=config.embedding_provider,
        )

    if "llm" in accepts and getattr(config, "llm_model", ""):
        fields["llm"] = get_llm(
            provider=config.llm_provider,
            model_name=config.llm_model,
            base_url=config.llm_base_url(),
            api_key=config.llm_api_key(),
            max_new_tokens=getattr(config, "max_new_tokens", 256),
        )

    if "reranker" in accepts and getattr(config, "reranker_model", None):
        fields["reranker"] = get_reranker(config.reranker_model)

    if "prompt" in accepts and getattr(config, "prompt_template", ""):
        fields["prompt_template"] = get_template(config.prompt_template)

    if "retriever" in accepts and getattr(config, "embedding_model", ""):
        fields["retriever_factory"] = _make_retriever_factory(config)

    return ComponentBundle(**fields)


def _make_retriever_factory(config):
    """Return a closure that builds a retriever once the corpus is available.

    The factory defers vector-store construction until prepare() has the
    corpus chunks. This matches the Framework's existing two-stage pattern
    (index → query) but lets the adapter decide when to call it.
    """
    def factory(chunks: list):
        # Import locally to avoid a heavy import at module load.
        from benchmark.retrieval import build_vector_store

        cache_k = f"injected_{config.embedding_model}_{config.chunk_size}_{config.chunk_overlap}_{config.chunking_strategy}"
        collection_name = f"rag_injected_{config.vector_db_backend}_{cache_k[:24]}"
        return build_vector_store(
            chunks,
            config.embedding_model,
            collection_name,
            ollama_base_url=config.embedding_base_url(),
            ollama_api_key=config.embedding_api_key(),
            cache_key=cache_k,
            embedding_provider=config.embedding_provider,
            vector_db_backend=config.vector_db_backend,
            lancedb_path=getattr(config, "lancedb_path", ".lancedb"),
            create_if_missing=True,
        )
    return factory
