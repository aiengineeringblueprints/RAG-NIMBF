"""Reference adapter for Enterprise_RAG_Blueprint.

Demonstrates the component-injection contract: accepts Framework chunker,
embedder, reranker, and LLM, but keeps its own retriever and prompt.

Activation (bash):

    PYTHONPATH=examples \\
    RAG_ADAPTER_MODULES=enterprise_rag_plugin.adapter \\
    RAG_SYSTEM_ADAPTER=enterprise_rag \\
    RAG_ADAPTER_ACCEPTS=chunker,embedder,llm,reranker \\
    BENCHMARK_CONFIG_FILE=experiments/enterprise_rag_demo.yaml \\
    python main.py

NOTE: import paths ``chain.retriever.EnterpriseRetriever`` and
``chain.load_chain.build_chain`` must match the actual Enterprise_RAG_Blueprint
codebase at integration time. Adjust the imports below if they differ.
"""

from __future__ import annotations

import time
from typing import Any

from benchmark.adapters import register_rag_adapter
from benchmark.adapters.base import RagSystemOutput
from benchmark.adapters.components import ComponentBundle


class EnterpriseRagAdapter:
    name = "enterprise_rag"

    def __init__(self, config: Any):
        self.config = config
        self.bundle: ComponentBundle = ComponentBundle()
        self.chain = None
        self.retriever = None

    def supports_components(self) -> dict[str, bool]:
        return {
            "chunker": True,
            "embedder": True,
            "retriever": False,
            "reranker": True,
            "llm": True,
            "prompt": False,
        }

    def set_components(self, bundle: ComponentBundle) -> None:
        self.bundle = bundle

    def prepare(
        self,
        config: Any,
        data: list[dict],
        corpus: list[dict] | None = None,
    ) -> None:
        from chain.retriever import EnterpriseRetriever
        from chain.load_chain import build_chain

        docs = corpus if corpus is not None else data
        self.retriever = EnterpriseRetriever(corpus=docs)

        if self.bundle.chunker is not None:
            self.retriever.chunker = self.bundle.chunker
        if self.bundle.embedder is not None:
            self.retriever.embedder = self.bundle.embedder
        if self.bundle.reranker is not None:
            self.retriever.reranker = self.bundle.reranker

        llm = self.bundle.llm
        self.chain = build_chain(llm=llm, retriever=self.retriever)

    def answer(self, sample: dict, config: Any) -> RagSystemOutput:
        if self.chain is None:
            raise RuntimeError("EnterpriseRagAdapter.prepare() was not called")
        t0 = time.perf_counter()
        result = self.chain.invoke(sample["question"])
        elapsed = time.perf_counter() - t0

        docs = result.get("source_documents", []) if isinstance(result, dict) else []
        return RagSystemOutput(
            answer=result.get("answer", "") if isinstance(result, dict) else str(result),
            contexts=[getattr(d, "page_content", str(d)) for d in docs],
            metadata=[
                getattr(d, "metadata", {}) or {"doc_id": None} for d in docs
            ],
            total_seconds=elapsed,
        )


register_rag_adapter("enterprise_rag", lambda c: EnterpriseRagAdapter(c))
