"""Reference-point RAG adapters: no-retrieval, random-retrieval, oracle.

These isolate what retrieval actually contributes:

- ``no_retrieval``    closed-book: the LLM answers without any context
                      (lower bound / parametric-knowledge baseline)
- ``random_retrieval`` contexts drawn uniformly at random from the corpus
                      (retrieval can only hurt if the system is worse than this)
- ``oracle_retrieval`` the gold context from the dataset row
                      (upper bound: exposes retrieval headroom)

All three share the framework's generator and evaluator, so their results are
directly comparable with ``internal`` runs in the same experiment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from benchmark.adapters.base import RagSystemOutput
from benchmark.generation import (
    extract_concise_fallback,
    generate_answer,
    get_llm,
)
from benchmark.prompt_templates import get_template

_CLOSED_BOOK_SYSTEM_PROMPT = (
    "Answer the question from your own knowledge. "
    "Return ONLY the raw value — a number, percentage, ratio, or yes/no. "
    "Do NOT include units, explanations, reasoning, or full sentences."
)
_CLOSED_BOOK_HUMAN_TEMPLATE = "Question: {question}\n\nAnswer:"


@dataclass
class NoRetrievalAdapter:
    """Closed-book LLM baseline: no contexts are passed to the generator."""

    name: str = "no_retrieval"
    llm: Any = None
    prompt_template: Any = None

    @classmethod
    def from_config(cls, config: Any) -> "NoRetrievalAdapter":
        return cls(
            llm=get_llm(
                provider=config.llm_provider,
                model_name=config.llm_model,
                base_url=config.llm_base_url(),
                api_key=config.llm_api_key(),
                max_new_tokens=config.max_new_tokens,
            ),
            prompt_template=get_template(config.prompt_template),
        )

    def prepare(self, config: Any, data: list[dict], corpus=None) -> None:
        del config, data, corpus

    def cleanup(self, target: Any, config: Any) -> None:
        del target, config

    def answer(self, sample: dict, config: Any) -> RagSystemOutput:
        generated = generate_answer(
            self.llm,
            str(sample["question"]),
            [],
            system_prompt=_CLOSED_BOOK_SYSTEM_PROMPT,
            human_template=_CLOSED_BOOK_HUMAN_TEMPLATE,
            strip_mode=config.llm_answer_strip_mode,
            value_fallback=config.llm_answer_value_fallback,
            ground_truth=sample.get("ground_truth"),
            prompt_template_name=config.prompt_template,
            cost_model_name=config.llm_model,
        )
        answer = generated.answer
        if config.llm_answer_value_fallback and answer:
            answer = extract_concise_fallback(answer) or answer
        return RagSystemOutput(
            answer=answer,
            contexts=[],
            metadata=[],
            total_seconds=generated.total_seconds,
            ttft_seconds=generated.ttft_seconds,
            token_count=generated.token_count,
            tokens_per_second=generated.tokens_per_second,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
            total_tokens=generated.total_tokens,
            estimated_cost_usd=generated.estimated_cost_usd,
            gpu_usage=generated.gpu_usage,
            raw_content=generated.raw_content,
            raw_reasoning=generated.raw_reasoning,
            answer_valid=generated.answer_valid,
            diagnostics={"baseline": "no_retrieval"},
        )


@dataclass
class RandomRetrievalAdapter:
    """Random-context baseline: samples top_k documents uniformly at random.

    Deterministic per sample: the RNG is seeded from the question text, so a
    config reproduces the same contexts across runs.
    """

    name: str = "random_retrieval"
    llm: Any = None
    prompt_template: Any = None
    seed: int = 42
    _corpus: list[dict] = field(default_factory=list, init=False, repr=False)

    @classmethod
    def from_config(cls, config: Any) -> "RandomRetrievalAdapter":
        return cls(
            llm=get_llm(
                provider=config.llm_provider,
                model_name=config.llm_model,
                base_url=config.llm_base_url(),
                api_key=config.llm_api_key(),
                max_new_tokens=config.max_new_tokens,
            ),
            prompt_template=get_template(config.prompt_template),
        )

    def prepare(self, config: Any, data: list[dict], corpus=None) -> None:
        self._corpus = list(corpus or [])
        if not self._corpus:
            raise ValueError(
                "random_retrieval requires a corpus; use a shared-corpus dataset "
                "or set dataset.corpus_path"
            )

    def cleanup(self, target: Any, config: Any) -> None:
        del target, config
        self._corpus = []

    def _sample_contexts(self, sample: dict, top_k: int) -> list[str]:
        rng = random.Random(f"{self.seed}:{sample['question']}")
        picks = rng.sample(
            range(len(self._corpus)), k=min(top_k, len(self._corpus))
        )
        contexts: list[str] = []
        for index in picks:
            doc = self._corpus[index]
            text = doc.get("context") if isinstance(doc, dict) else doc
            contexts.append(str(text))
        return contexts

    def answer(self, sample: dict, config: Any) -> RagSystemOutput:
        contexts = self._sample_contexts(sample, int(config.retrieval_top_k))
        generated = generate_answer(
            self.llm,
            str(sample["question"]),
            contexts,
            system_prompt=self.prompt_template.system_prompt,
            human_template=self.prompt_template.human_template,
            strip_mode=config.llm_answer_strip_mode,
            value_fallback=config.llm_answer_value_fallback,
            ground_truth=sample.get("ground_truth"),
            prompt_template_name=config.prompt_template,
            cost_model_name=config.llm_model,
        )
        metadata = [
            {"rank": rank + 1, "doc_id": "random", "baseline": "random_retrieval"}
            for rank in range(len(contexts))
        ]
        return RagSystemOutput(
            answer=generated.answer,
            contexts=contexts,
            metadata=metadata,
            total_seconds=generated.total_seconds,
            ttft_seconds=generated.ttft_seconds,
            token_count=generated.token_count,
            tokens_per_second=generated.tokens_per_second,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
            total_tokens=generated.total_tokens,
            estimated_cost_usd=generated.estimated_cost_usd,
            gpu_usage=generated.gpu_usage,
            raw_content=generated.raw_content,
            raw_reasoning=generated.raw_reasoning,
            answer_valid=generated.answer_valid,
            diagnostics={"baseline": "random_retrieval"},
        )


@dataclass
class OracleRetrievalAdapter:
    """Oracle baseline: retrieves exactly the gold context of the question."""

    name: str = "oracle_retrieval"
    llm: Any = None
    prompt_template: Any = None

    @classmethod
    def from_config(cls, config: Any) -> "OracleRetrievalAdapter":
        return cls(
            llm=get_llm(
                provider=config.llm_provider,
                model_name=config.llm_model,
                base_url=config.llm_base_url(),
                api_key=config.llm_api_key(),
                max_new_tokens=config.max_new_tokens,
            ),
            prompt_template=get_template(config.prompt_template),
        )

    def prepare(self, config: Any, data: list[dict], corpus=None) -> None:
        del config, data, corpus

    def cleanup(self, target: Any, config: Any) -> None:
        del target, config

    def answer(self, sample: dict, config: Any) -> RagSystemOutput:
        contexts = [str(sample["context"])] if sample.get("context") else []
        if not contexts:
            raise ValueError(
                "oracle_retrieval requires a non-empty context on every sample"
            )
        generated = generate_answer(
            self.llm,
            str(sample["question"]),
            contexts,
            system_prompt=self.prompt_template.system_prompt,
            human_template=self.prompt_template.human_template,
            strip_mode=config.llm_answer_strip_mode,
            value_fallback=config.llm_answer_value_fallback,
            ground_truth=sample.get("ground_truth"),
            prompt_template_name=config.prompt_template,
            cost_model_name=config.llm_model,
        )
        gold_id = (sample.get("metadata") or {}).get("gold_doc_id", "oracle")
        metadata = [
            {"rank": 1, "doc_id": gold_id, "baseline": "oracle_retrieval"}
        ]
        return RagSystemOutput(
            answer=generated.answer,
            contexts=contexts,
            metadata=metadata,
            total_seconds=generated.total_seconds,
            ttft_seconds=generated.ttft_seconds,
            token_count=generated.token_count,
            tokens_per_second=generated.tokens_per_second,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
            total_tokens=generated.total_tokens,
            estimated_cost_usd=generated.estimated_cost_usd,
            gpu_usage=generated.gpu_usage,
            raw_content=generated.raw_content,
            raw_reasoning=generated.raw_reasoning,
            answer_valid=generated.answer_valid,
            diagnostics={"baseline": "oracle_retrieval"},
        )
