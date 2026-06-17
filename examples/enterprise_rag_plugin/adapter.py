"""Reference adapter for Enterprise_RAG_Blueprint.

Demonstrates component-injection contract with the real Blueprint API:
- The benchmark dataset corpus is indexed into the Blueprint Chroma collection
  during prepare(), so the YAML-selected dataset is the only retrieval base.
- Framework-built chunker is used to split the dataset corpus.
- Framework-built embedder is passed as existing_embeddings to
  create_retriever.
- Framework-built llm is passed as model to load_chain.
- Blueprint's own prompt / reranker are kept (Blueprint has opinionated
  PromptKey + Chroma-only stack), so those slots are False.
- Retrieval top_k is read from config.retrieval_top_k.

Activation (bash):

    PYTHONPATH=.:Enterprise_RAG_Blueprint:Enterprise_RAG_Blueprint/chain \
    RAG_ADAPTER_MODULES=examples.enterprise_rag_plugin.adapter \
    BENCHMARK_CONFIG_FILE=experiments/enterprise_rag_demo.yaml \
    python main.py

Requires the Blueprint env (``EMBEDDING_MODEL``, ``INDEX_NAME``,
``VECTORDB_DIR``, ``RETRIEVER_DISABLE_FILTER`` etc.) so the Blueprint's own
``os.environ.get`` calls inside ``chain.retriever`` resolve.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from langchain_core.documents import Document

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
            "chunker": True,    # YAML-selected dataset corpus is chunked here
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

        self._replace_blueprint_index(config, data=data, corpus=corpus)

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

    def _replace_blueprint_index(
        self,
        config: Any,
        data: list[dict],
        corpus: list[dict] | None = None,
    ) -> None:
        """Replace the Blueprint Chroma collection with the benchmark corpus."""
        if self.bundle.embedder is None:
            raise ValueError(
                "EnterpriseRagAdapter requires an injected embedder. "
                "Use RAG_ADAPTER_MODULES and RAG_SYSTEM_ADAPTER=enterprise_rag "
                "with a config that provides an embedding model."
            )

        chunker = self.bundle.chunker
        if chunker is None:
            from benchmark.chunking import get_chunker

            chunker = get_chunker(
                getattr(config, "chunking_strategy", "recursive") or "recursive",
                int(getattr(config, "chunk_size", 512) or 512),
                int(getattr(config, "chunk_overlap", 64) or 64),
            )

        source_docs = corpus if corpus else data
        documents = self._documents_from_dataset(source_docs)
        chunks = chunker.split_documents(documents)
        chunks = [chunk for chunk in chunks if chunk.page_content.strip()] or documents
        for index, chunk in enumerate(chunks):
            chunk.metadata = self._sanitize_metadata(
                {
                    **(chunk.metadata or {}),
                    "chunk_index": index,
                    "category": "General",
                    "category_bitmask": 1,
                }
            )

        from langchain_chroma import Chroma

        index_name = os.environ.get("INDEX_NAME", "langchain-test-index")
        vector_db_dir = os.environ.get("VECTORDB_DIR", "/data/vectordb")
        os.makedirs(vector_db_dir, exist_ok=True)

        vectorstore = Chroma(
            collection_name=index_name,
            embedding_function=self.bundle.embedder,
            persist_directory=vector_db_dir,
        )
        try:
            vectorstore.delete_collection()
        except Exception:
            log.debug("Blueprint collection %s did not need deletion", index_name)

        vectorstore = Chroma(
            collection_name=index_name,
            embedding_function=self.bundle.embedder,
            persist_directory=vector_db_dir,
        )
        vectorstore.add_documents(chunks)

        persist_fn = getattr(vectorstore, "persist", None)
        client = getattr(vectorstore, "_client", None)
        if callable(persist_fn):
            persist_fn()
        elif client is not None and callable(getattr(client, "persist", None)):
            client.persist()

        log.info(
            "Indexed %s dataset chunks into Blueprint Chroma collection %s at %s",
            len(chunks),
            index_name,
            vector_db_dir,
        )

    @staticmethod
    def _documents_from_dataset(items: list[dict]) -> list[Document]:
        documents: list[Document] = []
        for index, item in enumerate(items):
            context = item.get("context", "")
            if isinstance(context, list):
                context = "\n".join(str(part) for part in context)
            text = str(context).strip()
            if not text:
                continue

            raw_metadata = item.get("metadata") or {}
            metadata = {
                **raw_metadata,
                "source": raw_metadata.get(
                    "source",
                    raw_metadata.get("doc_id", f"dataset-doc-{index}"),
                ),
            }
            documents.append(
                Document(
                    page_content=text,
                    metadata=EnterpriseRagAdapter._sanitize_metadata(metadata),
                )
            )

        if not documents:
            raise ValueError("Dataset did not provide any non-empty context to index")
        return documents

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[str(key)] = value
            else:
                sanitized[str(key)] = str(value)
        return sanitized

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
