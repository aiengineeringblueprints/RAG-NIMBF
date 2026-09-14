"""Framework-built components offered to external RAG adapters via injection.

The injection policy (decide-whether-and-what-to-inject) lives here, beside
the component factory. The orchestrator only hands over the adapter and a
component bundle; it never re-derives capabilities from the config and no
code path mutates the config object. ``rag_adapter_accepts`` remains a
user-intent override that can only narrow what the adapter declares.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
    chunker: Any | None = None  # langchain TextSplitter
    embedder: Any | None = None  # langchain Embeddings
    retriever_factory: Callable[[list[dict]], Any] | None = None
    reranker: Any | None = None
    llm: Any | None = None  # langchain BaseChatModel
    prompt_template: str | None = None




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


def build_components(config, accepts: set[str] | None = None) -> ComponentBundle:
    """Build a ComponentBundle for the given component slots.

    ``accepts`` is the resolved set of slots to populate. When omitted it
    falls back to the user-intent ``RAG_ADAPTER_ACCEPTS`` config field. The
    config object is only ever read — never mutated.
    """
    if accepts is None:
        accepts = _parse_accepts(getattr(config, "rag_adapter_accepts", ""))
    fields: dict[str, Any] = {}

    if "chunker" in accepts and getattr(config, "chunking_strategy", ""):
        chunker_kwargs: dict[str, Any] = {}
        if config.chunking_strategy == "semantic":
            # Semantic splitters need embeddings; mirror the internal
            # pipeline's construction so an injected chunker behaves the
            # same as a self-built one.
            chunker_kwargs = dict(
                embeddings=get_embedding_model(
                    config.embedding_model,
                    config.embedding_base_url(),
                    config.embedding_api_key(),
                    provider=config.embedding_provider,
                ),
                breakpoint_threshold_type=config.semantic_breakpoint_type,
                breakpoint_threshold_amount=config.semantic_breakpoint_amount,
            )
        fields["chunker"] = get_chunker(
            config.chunking_strategy,
            int(config.chunk_size or 0),
            int(config.chunk_overlap or 0),
            **chunker_kwargs,
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


def resolve_injection_slots(adapter: Any, config: Any) -> set[str]:
    """Resolve the injection policy for an adapter.

    Capability is derived from the adapter itself: ``supports_components()``
    declares which slots it can consume; adapters without the component
    protocol (pure black-box) accept nothing. The user-intent config field
    ``rag_adapter_accepts`` may only narrow the capability — it can never
    inject a slot the adapter did not declare. The config object is read
    only, never mutated.
    """
    if not hasattr(adapter, "set_components"):
        return set()
    capabilities = (
        adapter.supports_components() if hasattr(adapter, "supports_components") else {}
    )
    capability = {k for k, v in (capabilities or {}).items() if v}
    user_intent = _parse_accepts(getattr(config, "rag_adapter_accepts", ""))
    if user_intent:
        return capability & user_intent
    return capability


def build_injected_components(adapter: Any, config: Any) -> ComponentBundle:
    """Resolve the policy for ``adapter`` and build the resulting bundle."""
    return build_components(config, accepts=resolve_injection_slots(adapter, config))


def inject_components(adapter: Any, bundle: ComponentBundle, console: Any = None) -> None:
    """Hand a Framework-built component bundle to an adapter, if it accepts.

    Pure handover: the policy was already applied when the bundle was built
    (see ``resolve_injection_slots``). Adapters without ``set_components``
    are skipped; empty bundles are not delivered.
    """
    if not hasattr(adapter, "set_components"):
        return

    populated = {
        k: v
        for k, v in {
            "chunker": bundle.chunker,
            "embedder": bundle.embedder,
            "retriever": bundle.retriever_factory,
            "reranker": bundle.reranker,
            "llm": bundle.llm,
            "prompt": bundle.prompt_template,
        }.items()
        if v is not None
    }
    if not populated:
        return

    adapter.set_components(bundle)
    message = f"Injected components: {', '.join(sorted(populated))}"
    if console is not None:
        console.print(f"  [dim]{message}[/dim]")
    else:
        logging.getLogger(__name__).info(message)


def _make_retriever_factory(config):
    """Return a closure that builds a retriever once the corpus is available.

    The factory defers vector-store construction until prepare() has the
    corpus chunks. This matches the Framework's existing two-stage pattern
    (index → query) but lets the adapter decide when to call it.
    """
    def factory(chunks: list):
        # Import locally to avoid a heavy import at module load.
        from benchmark.retrieval import (
            build_vector_store,
            corpus_fingerprint_from_documents,
            index_cache_key,
        )

        cache_k = index_cache_key(
            config.embedding_model,
            config.chunk_size,
            config.chunk_overlap,
            config.chunking_strategy,
            dataset_name=getattr(config, "dataset_name", ""),
            embedding_provider=config.embedding_provider,
            dataset_subset=getattr(config, "dataset_subset", ""),
            dataset_sample_size=getattr(config, "dataset_sample_size", None),
            corpus_fingerprint=corpus_fingerprint_from_documents(chunks),
            vector_db_backend=config.vector_db_backend,
        )
        collection_name = f"rag_{config.vector_db_backend}_{cache_k[:24]}"
        return build_vector_store(
            chunks,
            config.embedding_model,
            collection_name=collection_name,
            ollama_base_url=config.embedding_base_url(),
            ollama_api_key=config.embedding_api_key(),
            cache_key=cache_k,
            embedding_provider=config.embedding_provider,
            vector_db_backend=config.vector_db_backend,
            lancedb_path=getattr(config, "lancedb_path", ".lancedb"),
            create_if_missing=getattr(config, "benchmark_stage", "all") != "query",
        )
    return factory
