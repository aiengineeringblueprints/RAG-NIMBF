"""Managed adapter wrapping the built-in (internal) RAG pipeline.

The adapter exposes the framework's own chunk → embed → index → retrieve →
rerank → generate pipeline through the managed lifecycle contract, so the
registry can hand out a real adapter for ``internal`` instead of ``None``.
The orchestrator keeps its dedicated internal branch until that branch is
removed; this adapter is the expand half of expand–contract.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from benchmark.adapters.base import (
    AdapterCapabilities,
    AdapterGenerationResult,
    PreparedTarget,
    RetrievalResult,
    RetrievedChunk,
)
from benchmark.adapters.components import ComponentBundle
from benchmark.chunking import chunk_documents, get_chunker

logger = logging.getLogger(__name__)


class InternalRagAdapter:
    """Run the built-in RAG pipeline behind the managed adapter lifecycle."""

    name = "internal"

    def __init__(self, generator: Callable[..., Any] | None = None) -> None:
        # Lazy imports keep construction cheap and avoid import cycles.
        from benchmark.generation import generate_answer

        self._generator = generator or generate_answer
        self._bundle = ComponentBundle()
        self._store: Any = None
        self._custom_generator = generator is not None

    @classmethod
    def from_config(
        cls, config: Any, generator: Callable[..., Any] | None = None
    ) -> InternalRagAdapter:
        return cls(generator=generator)

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            ingestion=True,
            retrieval=True,
            generation=True,
            chunk_configuration=True,
            embedding_configuration=True,
            reranking=True,
            references=True,
            token_usage=True,
            cleanup=True,
        )

    def supports_components(self) -> dict[str, bool]:
        return {
            "chunker": True,
            "embedder": True,
            "retriever": True,
            "reranker": True,
            "llm": True,
            "prompt": True,
        }

    def set_components(self, bundle: ComponentBundle) -> None:
        self._bundle = bundle

    # ------------------------------------------------------------------
    # Component resolution: injected bundle slots take precedence over
    # components built from the experiment config.
    # ------------------------------------------------------------------

    def _chunker(self, config: Any):
        if self._bundle.chunker is not None:
            return self._bundle.chunker
        chunker_kwargs: dict = {}
        if config.chunking_strategy == "semantic":
            from benchmark.embedding import get_embedding_model

            chunker_kwargs = {
                "embeddings": get_embedding_model(
                    config.embedding_model,
                    config.embedding_base_url(),
                    config.embedding_api_key(),
                    provider=config.embedding_provider,
                ),
                "breakpoint_threshold_type": config.semantic_breakpoint_type,
                "breakpoint_threshold_amount": config.semantic_breakpoint_amount,
            }
        return get_chunker(
            config.chunking_strategy,
            config.chunk_size or 0,
            config.chunk_overlap or 0,
            **chunker_kwargs,
        )

    def _embedder(self, config: Any):
        if self._bundle.embedder is not None:
            return self._bundle.embedder
        from benchmark.embedding import get_embedding_model

        return get_embedding_model(
            config.embedding_model,
            config.embedding_base_url(),
            config.embedding_api_key(),
            provider=config.embedding_provider,
        )

    def _llm(self, config: Any):
        if self._bundle.llm is not None:
            return self._bundle.llm
        from benchmark.generation import get_llm

        return get_llm(
            provider=config.llm_provider,
            model_name=config.llm_model,
            base_url=config.llm_base_url(),
            api_key=config.llm_api_key(),
            max_new_tokens=config.max_new_tokens,
        )

    def _generation_llm(self, config: Any):
        """LLM handed to the generator.

        With a stubbed (injected) generator no model is built — the stub
        decides whether it needs one, keeping lifecycle tests cheap.
        """
        if self._custom_generator and self._bundle.llm is None:
            return None
        return self._llm(config)

    def _reranker(self, config: Any):
        if self._bundle.reranker is not None:
            return self._bundle.reranker
        if not getattr(config, "reranker_model", None):
            return None
        from benchmark.reranker import get_reranker

        return get_reranker(config.reranker_model)

    def _prompt(self, config: Any):
        if self._bundle.prompt_template is not None:
            return self._bundle.prompt_template
        from benchmark.prompt_templates import get_template

        return get_template(config.prompt_template)

    def _build_store(self, config: Any, chunks: list, fingerprint: str):
        if self._bundle.retriever_factory is not None:
            return self._bundle.retriever_factory(chunks)
        if self._bundle.embedder is not None:
            from langchain_community.vectorstores import InMemoryVectorStore

            return InMemoryVectorStore.from_documents(chunks, self._bundle.embedder)

        from benchmark.retrieval import build_vector_store

        cache_k = target_cache_key(config, fingerprint)
        return build_vector_store(
            chunks,
            config.embedding_model,
            f"rag_{config.vector_db_backend}_{cache_k[:24]}",
            ollama_base_url=config.embedding_base_url(),
            ollama_api_key=config.embedding_api_key(),
            cache_key=cache_k,
            embedding_provider=config.embedding_provider,
            vector_db_backend=config.vector_db_backend,
            lancedb_path=config.lancedb_path,
            create_if_missing=getattr(config, "benchmark_stage", "all") != "query",
        )

    # ------------------------------------------------------------------
    # Managed lifecycle
    # ------------------------------------------------------------------

    def prepare(
        self,
        config: Any,
        data: list[dict],
        corpus: list[dict] | None = None,
    ) -> PreparedTarget:
        chunker = self._chunker(config)
        chunk_source = corpus if corpus else data
        chunks = chunk_documents(chunker, chunk_source)
        fingerprint = cache_corpus_fingerprint(chunk_source)
        logger.info("Internal adapter chunked corpus into %d pieces", len(chunks))

        self._store = self._build_store(config, chunks, fingerprint)
        cache_key = target_cache_key(config, fingerprint)
        return PreparedTarget(
            target_id=f"rag_{config.vector_db_backend}_{cache_key[:24]}",
            dataset_ids=(str(config.dataset_name),),
            metadata={
                "cache_key": cache_key,
                "corpus_fingerprint": fingerprint,
                "chunk_count": len(chunks),
            },
        )

    def retrieve(
        self,
        target: PreparedTarget,
        sample: dict,
        config: Any,
    ) -> RetrievalResult:
        if self._store is None:
            raise RuntimeError(
                "Internal adapter retrieve() called before prepare()."
            )
        query = sample["question"]
        llm = None
        if config.retrieval_use_hyde:
            from benchmark.retrieval import expand_query_with_hyde

            llm = self._llm(config)
            query = expand_query_with_hyde(llm, sample["question"])

        docs = self._search(query, config, llm)
        docs = self._rerank(sample, docs, config)
        return _to_retrieval_result(docs)

    def generate(
        self,
        target: PreparedTarget,
        sample: dict,
        config: Any,
        retrieval: RetrievalResult | None = None,
    ) -> AdapterGenerationResult:
        llm = self._generation_llm(config)
        if config.retrieval_mode == "direct":
            retrieval_result = RetrievalResult(
                chunks=(RetrievedChunk(text=sample["context"], rank=1),)
            )
        else:
            retrieval_result = retrieval or self.retrieve(target, sample, config)
        contexts = retrieval_result.contexts

        prompt = self._prompt(config)
        result = self._generator(
            llm,  # type: ignore[arg-type]  # stubbed generators may take None
            sample["question"],
            contexts,
            system_prompt=getattr(prompt, "system_prompt", ""),
            human_template=getattr(prompt, "human_template", ""),
            strip_mode=config.llm_answer_strip_mode,
            value_fallback=config.llm_answer_value_fallback,
            ground_truth=sample.get("ground_truth"),
            prompt_template_name=config.prompt_template,
            cost_model_name=config.llm_model,
        )
        return AdapterGenerationResult(
            answer=result.answer,
            retrieval=retrieval_result,
            ttft_seconds=result.ttft_seconds,
            total_seconds=result.total_seconds,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
            estimated_cost_usd=result.estimated_cost_usd,
            raw_content=result.raw_content,
            raw_reasoning=result.raw_reasoning,
            answer_valid=result.answer_valid,
        )

    def cleanup(self, target: PreparedTarget, config: Any) -> None:
        # The built-in vector store is process-local and/or persisted on
        # disk under a content-addressed collection; releasing the
        # reference is all the cleanup the internal pipeline needs.
        self._store = None

    # ------------------------------------------------------------------
    # Search internals
    # ------------------------------------------------------------------

    def _search(self, query: str, config: Any, llm: Any) -> list:
        from benchmark.retrieval import retrieve, retrieve_multihop

        if getattr(config, "retrieval_multihop", False):
            return retrieve_multihop(
                self._store,
                llm if llm is not None else self._llm(config),
                query,
                config.retrieval_top_k,
                rounds=config.retrieval_multihop_rounds,
                retrieval_strategy=config.retrieval_strategy,
                fetch_k=config.retrieval_fetch_k,
                mmr_lambda=config.retrieval_mmr_lambda,
            )
        return retrieve(
            self._store,
            query,
            config.retrieval_top_k,
            retrieval_strategy=config.retrieval_strategy,
            fetch_k=config.retrieval_fetch_k,
            mmr_lambda=config.retrieval_mmr_lambda,
        )

    def _rerank(self, sample: dict, docs: list, config: Any) -> list:
        reranker = self._reranker(config)
        if reranker is None:
            return docs
        return reranker.rerank(
            sample["question"], docs, config.reranker_top_k,
        )


def cache_corpus_fingerprint(chunk_source: list[dict]) -> str:
    """Corpus fingerprint over raw dataset items (shared with main.py)."""
    from benchmark.retrieval import corpus_fingerprint

    return corpus_fingerprint(chunk_source)


def target_cache_key(config: Any, corpus_fingerprint: str) -> str:
    """Derive the shared index cache key for this experiment's corpus."""
    from benchmark.retrieval import index_cache_key

    return index_cache_key(
        config.embedding_model,
        config.chunk_size,
        config.chunk_overlap,
        config.chunking_strategy,
        dataset_name=config.dataset_name,
        embedding_provider=config.embedding_provider,
        dataset_subset=config.dataset_subset,
        dataset_sample_size=config.dataset_sample_size,
        corpus_fingerprint=corpus_fingerprint,
        vector_db_backend=config.vector_db_backend,
    )


def _to_retrieval_result(docs: list) -> RetrievalResult:
    return RetrievalResult(
        chunks=tuple(
            RetrievedChunk(
                text=doc.page_content,
                rank=rank,
                metadata=dict(doc.metadata) if doc.metadata else {},
            )
            for rank, doc in enumerate(docs, start=1)
        )
    )
