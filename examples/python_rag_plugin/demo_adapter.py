"""Reference black-box RAG adapter implemented as a Python plugin.

Purpose: demonstrate the full ``RagSystemOutput`` contract end-to-end with
zero external dependencies (no LLM, no embedding model, no network). The
retriever is a length-normalized TF cosine match over the corpus; the
generator is an extractive sentence picker. Swap those two methods for your
own retriever/generator and keep the rest of the skeleton.

Activation (bash):

    PYTHONPATH=examples \\
    RAG_ADAPTER_MODULES=python_rag_plugin.demo_adapter \\
    RAG_SYSTEM_ADAPTER=demo_python \\
    BENCHMARK_CONFIG_FILE=experiments/external_rag_demo.yaml \\
    python main.py

The plugin registers itself on import via :func:`register_rag_adapter`.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from benchmark.adapters import register_rag_adapter
from benchmark.adapters.base import RagSystemOutput


_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass
class _Doc:
    doc_id: str
    text: str
    tokens: list[str]
    norm: float


class DemoPythonRagAdapter:
    """In-memory TF retriever + extractive answer.

    Fills every :class:`RagSystemOutput` field so the framework can compute
    the full metric suite (RAGAS faithfulness/context precision/recall,
    nDCG/recall@k via ``metadata[].doc_id``, latency, token accounting).
    """

    name = "demo_python"

    def __init__(self, config: Any):
        self.config = config
        self.docs: list[_Doc] = []
        self.top_k: int = int(getattr(config, "retrieval_top_k", 5) or 5)

    def prepare(
        self,
        config: Any,
        data: list[dict],
        corpus: list[dict] | None = None,
    ) -> None:
        source = corpus if corpus else data
        seen: set[str] = set()
        total_tokens = 0
        for i, rec in enumerate(source or []):
            text = str(
                rec.get("context")
                or rec.get("text")
                or rec.get("content")
                or ""
            )
            if not text:
                continue
            meta = rec.get("metadata") or {}
            doc_id = str(
                meta.get("doc_id")
                or rec.get("doc_id")
                or f"doc-{i}"
            )
            if doc_id in seen:
                continue
            seen.add(doc_id)
            tokens = _tokenize(text)
            if not tokens:
                continue
            tf = Counter(tokens)
            norm = sum(c * c for c in tf.values()) ** 0.5
            self.docs.append(
                _Doc(doc_id=doc_id, text=text, tokens=tokens, norm=norm or 1.0)
            )
            total_tokens += len(tokens)

    def _retrieve(self, question: str, top_k: int) -> list[tuple[float, _Doc]]:
        q_tokens = _tokenize(question)
        if not q_tokens or not self.docs:
            return []
        q_tf = Counter(q_tokens)
        q_norm = sum(c * c for c in q_tf.values()) ** 0.5 or 1.0
        scored: list[tuple[float, _Doc]] = []
        for doc in self.docs:
            d_tf = Counter(doc.tokens)
            dot = sum(q_tf[t] * d_tf.get(t, 0) for t in q_tf)
            if dot <= 0:
                continue
            scored.append((dot / (q_norm * doc.norm), doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[: max(0, top_k)]

    @staticmethod
    def _extract_answer(question: str, contexts: list[str]) -> str:
        if not contexts:
            return ""
        q_tokens = set(_tokenize(question))
        best_sent = ""
        best_overlap = 0
        for ctx in contexts[:1]:
            for sent in _SENT_SPLIT_RE.split(ctx):
                s_tokens = _tokenize(sent)
                if not s_tokens:
                    continue
                overlap = sum(1 for t in s_tokens if t in q_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sent = sent.strip()
        return best_sent

    def answer(self, sample: dict, config: Any) -> RagSystemOutput:
        start = time.perf_counter()
        question = str(sample.get("question") or "")
        hits = self._retrieve(question, self.top_k)
        contexts = [d.text for _, d in hits]
        metadata = [
            {"doc_id": d.doc_id, "score": float(s)} for s, d in hits
        ]
        answer_text = self._extract_answer(question, contexts)
        elapsed = time.perf_counter() - start
        in_tokens = sum(len(c.split()) for c in contexts) + len(question.split())
        out_tokens = len(answer_text.split())
        return RagSystemOutput(
            answer=answer_text,
            contexts=contexts,
            metadata=metadata,
            ttft_seconds=elapsed * 0.5,
            total_seconds=elapsed,
            token_count=in_tokens + out_tokens,
            tokens_per_second=(
                (out_tokens / elapsed) if elapsed > 0 and out_tokens else 0.0
            ),
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            total_tokens=in_tokens + out_tokens,
            raw_content=answer_text,
            answer_valid=bool(answer_text.strip()),
        )


register_rag_adapter(
    "demo_python", lambda config: DemoPythonRagAdapter(config)
)
