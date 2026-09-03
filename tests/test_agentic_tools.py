"""Tests for the agentic RAG tool suite (benchmark/agentic/tools.py)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from langchain_core.documents import Document
from langchain_core.tools import BaseTool

from benchmark.agentic import metrics as agentic_metrics
from benchmark.agentic import tools as agentic_tools
from benchmark.adapters import agentic as agentic_adapter_module


class FakeAgentModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.schemas = None

    def bind_tools(self, schemas):
        self.schemas = schemas
        return self

    def invoke(self, messages):
        del messages
        return self.responses.pop(0)


@dataclass
class FakeRetriever:
    """Records calls and returns scripted documents."""

    docs: list[Document] = field(default_factory=list)
    calls: list[tuple[str, int]] = field(default_factory=list)
    filters: list[dict | None] = field(default_factory=list)

    def __call__(
        self, query: str, top_k: int = 4, filters: dict | None = None
    ) -> list[Document]:
        self.calls.append((query, top_k))
        self.filters.append(filters)
        return self.docs[:top_k]


def test_retrieve_tool_returns_documents_and_records_query() -> None:
    retriever = FakeRetriever(
        docs=[
            Document(page_content="chunk one", metadata={"source": "a.md"}),
            Document(page_content="chunk two", metadata={"source": "b.md"}),
        ]
    )
    tool = agentic_tools.build_retrieve_tool(retriever, default_top_k=4)

    result = tool.invoke({"query": "what is reward hacking"})

    assert retriever.calls == [("what is reward hacking", 4)]
    assert "chunk one" in result
    assert "chunk two" in result


def test_retrieve_tool_respects_top_k_argument() -> None:
    retriever = FakeRetriever(
        docs=[Document(page_content=f"chunk {i}") for i in range(5)]
    )
    tool = agentic_tools.build_retrieve_tool(retriever, default_top_k=4)

    result = tool.invoke({"query": "q", "top_k": 2})

    assert retriever.calls == [("q", 2)]
    assert result.count("chunk") == 2


def test_retrieve_with_filter_tool_passes_metadata_filter() -> None:
    retriever = FakeRetriever(
        docs=[Document(page_content="filtered chunk", metadata={"source": "c.md"})]
    )
    tool = agentic_tools.build_retrieve_with_filter_tool(retriever, default_top_k=4)

    result = tool.invoke(
        {"query": "q", "filters": '{"source": "c.md"}', "top_k": 3}
    )

    assert retriever.calls[0] == ("q", 3)
    assert retriever.filters == [{"source": "c.md"}]
    assert "filtered chunk" in result


def test_retrieve_with_filter_tool_invalid_json_returns_error() -> None:
    retriever = FakeRetriever()
    tool = agentic_tools.build_retrieve_with_filter_tool(retriever)

    result = tool.invoke({"query": "q", "filters": "not-json"})

    assert "Error" in result
    assert retriever.calls == []


def test_retrieve_alternate_tool_uses_secondary_retriever() -> None:
    primary = FakeRetriever(docs=[Document(page_content="primary chunk")])
    alternate = FakeRetriever(
        docs=[Document(page_content="alternate chunk", metadata={"source": "alt"})]
    )
    tool = agentic_tools.build_retrieve_alternate_tool(alternate, default_top_k=4)

    result = tool.invoke({"query": "q", "top_k": 2})

    assert alternate.calls == [("q", 2)]
    assert primary.calls == []
    assert "alternate chunk" in result


def test_hyde_retrieve_tool_searches_with_hypothetical_document() -> None:
    retriever = FakeRetriever(
        docs=[Document(page_content="hyde chunk", metadata={"source": "h.md"})]
    )

    def hypothesize(query: str) -> str:
        return f"hypothetical answer to: {query}"

    tool = agentic_tools.build_hyde_retrieve_tool(
        hypothesize, retriever, default_top_k=4
    )

    result = tool.invoke({"query": "q"})

    assert retriever.calls == [("hypothetical answer to: q", 4)]
    assert "hyde chunk" in result


def test_rewrite_query_tool_returns_rewritten_query() -> None:
    calls: list[str] = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return "reward hacking types LLM alignment"

    tool = agentic_tools.build_rewrite_query_tool(fake_llm)

    result = tool.invoke({"query": "ways models game rewards"})

    assert len(calls) == 1
    assert "ways models game rewards" in calls[0]
    assert result == "reward hacking types LLM alignment"


def test_rerank_chunks_tool_reorders_documents() -> None:
    def fake_reranker(
        query: str, documents: list[Document], top_k: int
    ) -> list[Document]:
        assert query == "q"
        return list(reversed(documents))[:top_k]

    tool = agentic_tools.build_rerank_chunks_tool(fake_reranker)

    result = tool.invoke(
        {
            "query": "q",
            "documents": ["doc a", "doc b", "doc c"],
            "top_k": 2,
        }
    )

    assert result == "doc c\ndoc b"


def test_rerank_chunks_tool_empty_documents() -> None:
    tool = agentic_tools.build_rerank_chunks_tool(lambda q, d, k: d)

    result = tool.invoke({"query": "q", "documents": []})

    assert result == "No documents found."


def test_grade_relevance_tool_reports_verdict() -> None:
    calls: list[tuple[str, str]] = []

    def fake_grader(query: str, document: str) -> bool:
        calls.append((query, document))
        return document == "relevant doc"

    tool = agentic_tools.build_grade_relevance_tool(fake_grader)

    result = tool.invoke({"query": "q", "document": "relevant doc"})

    assert calls == [("q", "relevant doc")]
    assert result == "relevant"


def test_grade_relevance_tool_negative_verdict() -> None:
    tool = agentic_tools.build_grade_relevance_tool(lambda q, d: False)

    result = tool.invoke({"query": "q", "document": "junk"})

    assert result == "not relevant"


def test_grade_answer_tool_supported_verdict() -> None:
    calls: list[dict] = []

    def fake_checker(answer: str, contexts: list[str]) -> bool:
        calls.append({"answer": answer, "contexts": contexts})
        return True

    tool = agentic_tools.build_grade_answer_tool(fake_checker)

    result = tool.invoke(
        {"answer": "42", "contexts": ["the answer is 42"], "question": "what?"}
    )

    assert calls == [{"answer": "42", "contexts": ["the answer is 42"]}]
    assert result == "supported"


def test_grade_answer_tool_unsupported_verdict() -> None:
    tool = agentic_tools.build_grade_answer_tool(lambda a, c: False)

    result = tool.invoke({"answer": "made up", "contexts": []})

    assert result == "unsupported"


def test_web_search_tool_returns_results() -> None:
    calls: list[str] = []

    def fake_search(query: str) -> str:
        calls.append(query)
        return "search result snippet"

    tool = agentic_tools.build_web_search_tool(fake_search)

    result = tool.invoke({"query": "langchain agentic rag"})

    assert calls == ["langchain agentic rag"]
    assert "search result snippet" in result


def test_fetch_url_tool_returns_page_text() -> None:
    calls: list[str] = []

    def fake_fetch(url: str) -> str:
        calls.append(url)
        return "<html>page body</html>"

    tool = agentic_tools.build_fetch_url_tool(fake_fetch)

    result = tool.invoke({"url": "https://example.com/post"})

    assert calls == ["https://example.com/post"]
    assert result == "<html>page body</html>"


def test_fetch_url_tool_blocks_disallowed_domain() -> None:
    tool = agentic_tools.build_fetch_url_tool(
        lambda url: "should not be called",
        allowed_domains=["https://allowed.com"],
    )

    result = tool.invoke({"url": "https://evil.com/x"})

    assert "Error" in result
    assert "allowed.com" in result


EXPECTED_TOOL_NAMES = {
    "retrieve",
    "retrieve_with_filter",
    "retrieve_alternate",
    "hyde_retrieve",
    "rewrite_query",
    "rerank_chunks",
    "grade_relevance",
    "grade_answer",
    "web_search",
    "fetch_url",
}


def test_build_tool_suite_full() -> None:
    retriever = FakeRetriever(docs=[Document(page_content="x")])
    tools = agentic_tools.build_tool_suite(
        retriever=retriever,
        alternate_retriever=retriever,
        reranker=lambda q, d, k: d,
        llm=lambda p: "rewritten",
        hypothesize=lambda q: "hypothetical",
        grader=lambda q, d: True,
        answer_checker=lambda a, c: True,
        search_fn=lambda q: "results",
        fetch_fn=lambda u: "page",
        allowed_domains=["https://example.com"],
    )

    names = {t.name for t in tools}
    assert names == EXPECTED_TOOL_NAMES
    assert all(isinstance(t, BaseTool) for t in tools)


def test_build_tool_suite_minimal() -> None:
    retriever = FakeRetriever()
    tools = agentic_tools.build_tool_suite(retriever=retriever)

    names = {t.name for t in tools}
    assert names == {"retrieve", "retrieve_with_filter"}
    assert all(isinstance(t, BaseTool) for t in tools)


def _record(name="retrieve", ok=True, error=None, args=None, seconds=0.1):
    return agentic_metrics.ToolCallRecord(
        tool=name,
        round=1,
        arguments=args or {},
        ok=ok,
        error=error,
        total_seconds=seconds,
        result_chars=10,
    )


def test_compute_tool_call_metrics_counts_and_rates() -> None:
    records = [
        _record("retrieve", args={"query": "q1"}),
        _record("retrieve", args={"query": "q1"}),
        _record("rewrite_query", args={"query": "q1"}),
        _record("retrieve", ok=False, error="boom"),
    ]

    m = agentic_metrics.compute_tool_call_metrics(records)

    assert m["tool_calls_total"] == 4
    assert m["calls_by_tool"] == {"retrieve": 3, "rewrite_query": 1}
    assert m["invalid_call_count"] == 1
    assert m["invalid_call_rate"] == pytest.approx(0.25)
    assert m["redundant_call_count"] == 1
    assert m["redundant_call_rate"] == pytest.approx(0.25)
    assert m["tool_seconds_total"] == pytest.approx(0.4)


def test_compute_tool_call_metrics_selection_against_expected() -> None:
    records = [
        _record("retrieve"),
        _record("retrieve"),
        _record("rewrite_query"),
        _record("fetch_url"),
    ]
    available = ["retrieve", "retrieve_with_filter", "rewrite_query"]

    m = agentic_metrics.compute_tool_call_metrics(
        records, expected_tools=["retrieve", "retrieve_with_filter"],
        available_tools=available,
    )

    assert m["hallucinated_tool_count"] == 1
    assert m["hallucinated_tool_rate"] == pytest.approx(0.25)
    assert m["tool_selection_precision"] == pytest.approx(0.5)
    assert m["tool_selection_recall"] == pytest.approx(0.5)


def test_compute_tool_call_metrics_empty() -> None:
    m = agentic_metrics.compute_tool_call_metrics([])
    assert m["tool_calls_total"] == 0
    assert m["invalid_call_rate"] == 0.0


def test_aggregate_tool_call_metrics_over_diagnostics() -> None:
    diagnostics = [
        {"tool_call_records": [_record("retrieve"), _record()]},
        {"tool_call_records": []},
        {},
    ]

    m = agentic_metrics.aggregate_tool_call_metrics(diagnostics)

    assert m["tool_calls_total"] == 2
    assert m["tool_calls_mean"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# AgenticRagAdapter
# ---------------------------------------------------------------------------


@dataclass
class FakeResponse:
    """AIMessage-like response with optional tool calls."""

    content: str
    tool_calls: list = field(default_factory=list)
    usage_metadata: dict = field(default_factory=lambda: {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7})


def test_agentic_adapter_runs_tool_loop_and_returns_answer() -> None:
    retriever = FakeRetriever(
        docs=[Document(page_content="chunk one", metadata={"source": "a.md"})]
    )
    retrieve_tool = agentic_tools.build_retrieve_tool(retriever)
    model = FakeAgentModel(
        [
            FakeResponse(
                "searching",
                tool_calls=[
                    {"name": "retrieve", "args": {"query": "q1"}, "id": "c1"}
                ],
            ),
            FakeResponse("42"),
        ]
    )
    adapter = agentic_adapter_module.AgenticRagAdapter(
        llm=model, tools=[retrieve_tool], max_rounds=4
    )

    out = adapter.answer({"question": "what is the answer?"}, config=object())

    assert out.answer == "42"
    assert len(out.contexts) == 1
    assert "chunk one" in out.contexts[0]
    assert retriever.calls == [("q1", 4)]
    assert out.diagnostics["agent_rounds"] == 2
    assert out.diagnostics["agent_retrieved"] is True
    records = out.diagnostics["tool_call_records"]
    assert len(records) == 1
    assert records[0]["tool"] == "retrieve"
    assert records[0]["ok"] is True


def test_agentic_adapter_records_invalid_tool_call() -> None:
    model = FakeAgentModel(
        [
            FakeResponse(
                "hallucinating",
                tool_calls=[
                    {"name": "nonexistent_tool", "args": {}, "id": "c1"}
                ],
            ),
            FakeResponse("fallback answer"),
        ]
    )
    adapter = agentic_adapter_module.AgenticRagAdapter(
        llm=model, tools=[], max_rounds=4
    )

    out = adapter.answer({"question": "q"}, config=object())

    assert out.answer == "fallback answer"
    records = out.diagnostics["tool_call_records"]
    assert records[0]["ok"] is False
    assert "nonexistent_tool" in records[0]["error"]


def test_agentic_adapter_exhausted_rounds_raises() -> None:
    call = FakeResponse(
        "calling",
        tool_calls=[{"name": "retrieve", "args": {"query": "q"}, "id": "c1"}],
    )
    model = FakeAgentModel([call, call])
    adapter = agentic_adapter_module.AgenticRagAdapter(
        llm=model,
        tools=[agentic_tools.build_retrieve_tool(FakeRetriever())],
        max_rounds=2,
    )

    with pytest.raises(RuntimeError, match="rounds"):
        adapter.answer({"question": "q"}, config=object())


def test_comparison_rows_include_tool_call_metrics() -> None:
    from benchmark.reporting.comparisons import _comparison_rows
    from types import SimpleNamespace

    per_sample = [
        SimpleNamespace(
            ragas_scores={"faithfulness": 0.9},
            custom_scores={},
            adapter_diagnostics={
                "execution_mode": "agentic",
                "agent_rounds": 2,
                "tool_call_records": [
                    agentic_metrics.ToolCallRecord(
                        tool="retrieve", round=1, ok=True, total_seconds=0.2
                    ).as_dict(),
                    agentic_metrics.ToolCallRecord(
                        tool="retrieve", round=2, ok=False, error="x"
                    ).as_dict(),
                ],
            },
        ),
        SimpleNamespace(
            ragas_scores={"faithfulness": 0.8},
            custom_scores={},
            adapter_diagnostics={"execution_mode": "fixed"},
        ),
    ]

    rows = _comparison_rows(per_sample)

    assert rows[0]["tool_calls_total"] == 2.0
    assert rows[0]["invalid_call_count"] == 1.0
    assert rows[0]["tool_seconds_total"] == pytest.approx(0.2)
    assert "tool_calls_total" not in rows[1]


def test_agentic_adapter_registered_and_from_config() -> None:
    from benchmark.adapters import get_rag_adapter
    from types import SimpleNamespace

    config = SimpleNamespace(
        rag_system_adapter="agentic",
        llm_provider="openai",
        llm_model="gpt-x",
        llm_base_url=lambda: "http://x",
        llm_api_key=lambda: None,
        max_new_tokens=64,
        agentic_max_rounds=6,
        agentic_system_prompt="custom prompt",
        retrieval_top_k=5,
        llm_answer_strip_mode="tags_only",
        llm_answer_value_fallback=True,
        prompt_template="concise",
    )

    # from_config builds the LLM eagerly -> would hit the network only on use;
    # get_chat_model construction itself must not raise.
    adapter = get_rag_adapter(config)

    assert adapter is not None
    assert adapter.name == "agentic"
    assert adapter.max_rounds == 6
    assert adapter.system_prompt == "custom prompt"
    assert adapter.default_top_k == 5


def test_agentic_adapter_failed_tool_call_recorded_and_continues() -> None:
    def boom(query: str, top_k: int = 4, filters=None):
        raise ValueError("vector store down")

    tool = agentic_tools.build_retrieve_tool(boom)
    model = FakeAgentModel(
        [
            FakeResponse("try", tool_calls=[{"name": "retrieve", "args": {"query": "q"}, "id": "c1"}]),
            FakeResponse("gave up"),
        ]
    )
    adapter = agentic_adapter_module.AgenticRagAdapter(
        llm=model, tools=[tool], max_rounds=4
    )

    out = adapter.answer({"question": "q"}, config=object())

    assert out.answer == "gave up"
    records = out.diagnostics["tool_call_records"]
    assert records[0]["ok"] is False
    assert "vector store down" in records[0]["error"]
