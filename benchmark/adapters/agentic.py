"""Agentic RAG adapter: runs the framework's local tool suite in a
ReAct-style loop and records per-call diagnostics for tool-call metrics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from benchmark.adapters.base import RagSystemOutput
from benchmark.agentic.metrics import ToolCallRecord
from benchmark.agentic.tools import build_tool_suite

_DEFAULT_SYSTEM_PROMPT = (
    "Answer the user's question using the available tools when evidence is "
    "needed. Search the corpus first; rewrite queries, re-rank, or grade "
    "relevance when helpful. Use web search only as a fallback. Return only "
    "the final answer when you have enough evidence."
)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def _message_usage(message: Any) -> tuple[int, int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0) or (
        input_tokens + output_tokens
    )
    return input_tokens, output_tokens, total_tokens


@dataclass
class AgenticRagAdapter:
    """Benchmark adapter running an agent over the local agentic tool suite.

    The LLM decides which tools to call each round; every call is recorded
    as a ``ToolCallRecord`` so run-level tool-call metrics can be computed
    with ``benchmark.agentic.metrics.aggregate_tool_call_metrics``.
    """

    name: str = "agentic"
    llm: Any = None
    tools: list[Any] = field(default_factory=list)
    max_rounds: int = 4
    system_prompt: str | None = None
    default_top_k: int = 4
    llm_answer_strip_mode: str = "tags_only"
    llm_answer_value_fallback: bool = True
    prompt_template: Any = None
    _retriever: Any = None
    _reranker: Any = None
    _components: Any = None

    @classmethod
    def from_config(cls, config: Any) -> "AgenticRagAdapter":
        from benchmark.generation import get_llm
        from benchmark.prompt_templates import get_template

        return cls(
            llm=get_llm(
                provider=config.llm_provider,
                model_name=config.llm_model,
                base_url=config.llm_base_url(),
                api_key=config.llm_api_key(),
                max_new_tokens=config.max_new_tokens,
            ),
            max_rounds=int(getattr(config, "agentic_max_rounds", 4)),
            system_prompt=getattr(config, "agentic_system_prompt", None),
            default_top_k=int(getattr(config, "retrieval_top_k", 4)),
            llm_answer_strip_mode=config.llm_answer_strip_mode,
            llm_answer_value_fallback=config.llm_answer_value_fallback,
            prompt_template=get_template(config.prompt_template),
        )

    def supports_components(self) -> dict[str, bool]:
        return {"retriever": True, "reranker": True, "llm": True}

    def set_components(self, bundle: Any) -> None:
        self._components = bundle

    def prepare(self, config: Any, data: list[dict], corpus=None) -> None:
        del data
        bundle = self._components
        if bundle is None:
            from benchmark.adapters.components import build_components

            bundle = build_components(config)
            self._components = bundle
        factory = getattr(bundle, "retriever_factory", None)
        self._retriever = factory(corpus or []) if factory else None
        self._reranker = getattr(bundle, "reranker", None)
        self.tools = self._build_tools(config)
        if bundle.llm is not None and self.llm is None:
            self.llm = bundle.llm

    def _build_tools(self, config: Any) -> list[Any]:
        if self._retriever is None:
            return []

        vectorstore = self._retriever

        def retrieve(query: str, top_k: int, filters: dict | None = None):
            return vectorstore.similarity_search(query, k=top_k, filter=filters)

        reranker_fn = None
        if self._reranker is not None:
            def reranker_fn(query, documents, top_k):
                return self._reranker.rerank(query, documents, top_k)

        llm_fn = None
        if self.llm is not None:
            def llm_fn(prompt: str) -> str:
                return str(self.llm.invoke(prompt))

        return build_tool_suite(
            retriever=retrieve,
            reranker=reranker_fn,
            llm=llm_fn,
            default_top_k=self.default_top_k,
        )

    def cleanup(self, target: Any, config: Any) -> None:
        del target, config

    def answer(self, sample: dict, config: Any) -> RagSystemOutput:
        del config
        if self.llm is None:
            raise RuntimeError("AgenticRagAdapter has no LLM initialized")

        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        schemas = [
            tool.tool_schema if hasattr(tool, "tool_schema") else tool
            for tool in self.tools
        ]
        model = self.llm.bind_tools(schemas)
        messages: list[Any] = [
            SystemMessage(content=self.system_prompt or _DEFAULT_SYSTEM_PROMPT),
            HumanMessage(content=str(sample["question"])),
        ]
        tools_by_name = {tool.name: tool for tool in self.tools}
        tool_names = sorted(tools_by_name)

        contexts: list[str] = []
        metadata: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        input_tokens = output_tokens = total_tokens = 0
        model_seconds = 0.0
        final_text = ""
        rounds_used = 0

        try:
            for round_number in range(1, self.max_rounds + 1):
                rounds_used = round_number
                started = time.perf_counter()
                response = model.invoke(messages)
                model_seconds += time.perf_counter() - started
                usage = _message_usage(response)
                input_tokens += usage[0]
                output_tokens += usage[1]
                total_tokens += usage[2]
                messages.append(response)

                tool_calls = list(getattr(response, "tool_calls", ()) or ())
                if not tool_calls:
                    final_text = _message_text(response).strip()
                    break

                for tool_call in tool_calls:
                    call_name = str(tool_call.get("name", ""))
                    arguments = tool_call.get("args", {}) or {}
                    started = time.perf_counter()
                    ok = True
                    error: str | None = None
                    result_text = ""
                    try:
                        if call_name not in tools_by_name:
                            raise ValueError(
                                f"LLM requested unknown tool {call_name!r}; "
                                f"available: {tool_names}"
                            )
                        result_text = str(
                            tools_by_name[call_name].invoke(arguments)
                        )
                    except Exception as exc:  # noqa: BLE001 - record and continue
                        ok = False
                        error = str(exc)
                    seconds = time.perf_counter() - started
                    record = ToolCallRecord(
                        tool=call_name,
                        round=round_number,
                        arguments=dict(arguments),
                        ok=ok,
                        error=error,
                        total_seconds=seconds,
                        result_chars=len(result_text),
                    )
                    records.append(record.as_dict())
                    if ok and result_text and call_name in {
                        "retrieve",
                        "retrieve_with_filter",
                        "retrieve_alternate",
                        "hyde_retrieve",
                    }:
                        offset = len(contexts)
                        contexts.append(result_text)
                        metadata.append(
                            {"tool": call_name, "rank": offset + 1}
                        )
                    messages.append(
                        ToolMessage(
                            content=result_text
                            if ok
                            else f"Error: {error}",
                            tool_call_id=str(tool_call.get("id", call_name)),
                        )
                    )
            else:
                raise RuntimeError(
                    f"Agentic agent exceeded {self.max_rounds} tool rounds"
                )
        finally:
            diagnostics: dict[str, Any] = {
                "execution_mode": "agentic",
                "agent_rounds": rounds_used,
                "agent_tool_calls": len(records),
                "agent_rounds_exhausted": False,
                "agent_retrieved": bool(contexts),
                "agent_model_seconds": model_seconds,
                "agent_tokens_input": input_tokens,
                "agent_tokens_output": output_tokens,
                "agent_tokens_total": total_tokens,
                "agent_tools_used": sorted(
                    {str(r["tool"]) for r in records}
                ),
                "tool_call_records": records,
            }
            self._last_diagnostics = diagnostics
            answer = final_text

        if not final_text:
            diagnostics["empty_response"] = True
        if self.llm_answer_value_fallback and answer:
            from benchmark.generation import extract_concise_fallback

            answer = extract_concise_fallback(answer) or answer

        return RagSystemOutput(
            answer=answer,
            contexts=contexts,
            metadata=metadata,
            total_seconds=model_seconds + sum(
                float(r["total_seconds"]) for r in records
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            raw_content=final_text,
            answer_valid=bool(final_text),
            diagnostics=diagnostics,
        )
