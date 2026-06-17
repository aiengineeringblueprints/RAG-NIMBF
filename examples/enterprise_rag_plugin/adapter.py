"""Reference adapter for Enterprise_RAG_Blueprint.

Demonstrates component-injection contract with the real Blueprint API:
- Framework-built ``embedder`` is passed as ``existing_embeddings`` to
  ``create_retriever``.
- Framework-built ``llm`` is passed as ``model`` to ``load_chain``.
- Blueprint's own chunker / prompt / reranker are kept (Blueprint has
  opinionated PromptKey + Chroma-only stack), so those slots are False.
- Retrieval ``top_k`` is read from ``config.retrieval_top_k``.

Activation (bash):

    PYTHONPATH=.:Enterprise_RAG_Blueprint \\
    RAG_ADAPTER_MODULES=enterprise_rag_plugin.adapter \\
    RAG_SYSTEM_ADAPTER=enterprise_rag \\
    RAG_ADAPTER_ACCEPTS=embedder,llm \\
    BENCHMARK_CONFIG_FILE=experiments/enterprise_rag_demo.yaml \\
    python main.py

Requires the Blueprint env (``EMBEDDING_MODEL``, ``INDEX_NAME``,
``VECTORDB_DIR``, ``RETRIEVER_DISABLE_FILTER`` etc.) so the Blueprint's own
``os.environ.get`` calls inside ``chain.retriever`` resolve.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from benchmark.adapters import register_rag_adapter
from benchmark.adapters.base import RagSystemOutput
from benchmark.adapters.components import ComponentBundle

log = logging.getLogger(__name__)


class EnterpriseRagAdapter:
    name = "enterprise_rag"

    def __init__(self, config: Any):
        self.config = config
        self.bundle: ComponentBundle = ComponentBundle()
        self.chain = None
        self.retriever = None
        self.prompt_key = None

    def supports_components(self) -> dict[str, bool]:
        return {
            "chunker": False,   # Blueprint chunkt im loader
            "embedder": True,   # bundle.embedder -> existing_embeddings
            "retriever": False, # Blueprint baut eigenen Chroma-Retriever
            "reranker": False,  # Blueprint hat keinen separaten Reranker-Slot
            "llm": True,        # bundle.llm -> model
            "prompt": False,    # Blueprint nutzt PromptKey-System
        }

    def set_components(self, bundle: ComponentBundle) -> None:
        self.bundle = bundle

    def prepare(
        self,
        config: Any,
        data: list[dict],
        corpus: list[dict] | None = None,
    ) -> None:
        from chain.load_chain import load_chain
        from chain.prompts.promt_manager import PromptKey
        from chain.retriever import create_retriever

        top_k = int(getattr(config, "retrieval_top_k", 3) or 3)

        self.retriever = create_retriever(
            returned_docs=top_k,
            similarity_threshold=float(
                getattr(config, "retriever_similarity_threshold", 0.5) or 0.5
            ),
            existing_embeddings=self.bundle.embedder,
        )

        self.prompt_key = PromptKey.SOURCE
        self.chain = load_chain(
            context=self.retriever,
            prompt_key=self.prompt_key,
            model=self.bundle.llm,
            include_doc_names=True,
        )

    def answer(self, sample: dict, config: Any) -> RagSystemOutput:
        from chain.load_chain import rag_chain

        if self.chain is None or self.retriever is None:
            raise RuntimeError("EnterpriseRagAdapter.prepare() was not called")

        question = sample.get("question") or sample.get("query") or ""
        t0 = time.perf_counter()
        answer_str, sources = rag_chain(
            question=question,
            chain=self.chain,
            retriever=self.retriever,
            show_sources=True,
        )
        elapsed = time.perf_counter() - t0

        # sources is list[str] formatted as "src->content" (or None)
        contexts: list[str] = []
        metadata: list[dict] = []
        if sources:
            for entry in sources:
                if isinstance(entry, str) and "->" in entry:
                    src, _, content = entry.partition("->")
                    contexts.append(content)
                    metadata.append({"source": src})
                elif isinstance(entry, str):
                    contexts.append(entry)
                    metadata.append({"source": "unknown"})
                else:
                    doc = entry
                    contexts.append(getattr(doc, "page_content", str(doc)))
                    metadata.append(getattr(doc, "metadata", {}) or {})

        return RagSystemOutput(
            answer=str(answer_str or ""),
            contexts=contexts,
            metadata=metadata,
            total_seconds=elapsed,
        )


register_rag_adapter("enterprise_rag", lambda c: EnterpriseRagAdapter(c))
