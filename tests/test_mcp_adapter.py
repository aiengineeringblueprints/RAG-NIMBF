from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from benchmark.adapters.mcp import McpRagAdapter
from benchmark.generation import GenerationResult
from benchmark.prompt_templates.types import PromptTemplate


class FakeToolResult:
    def __init__(self, *, structured=None, texts=(), is_error=False):
        self.structuredContent = structured
        self.content = [SimpleNamespace(text=text) for text in texts]
        self.isError = is_error

    def model_dump(self, **kwargs):
        del kwargs
        return {
            "structuredContent": self.structuredContent,
            "isError": self.isError,
        }


class FakeRunner:
    def __init__(self, responses=(), tools=("search",), failures=0, **kwargs):
        del kwargs
        self.responses = list(responses)
        self.failures = failures
        self.closed = False
        self.calls = []
        self.tools = [
            SimpleNamespace(
                name=name,
                description=f"{name} tool",
                inputSchema={"type": "object", "properties": {}},
            )
            for name in tools
        ]

    def start(self):
        return 0.125

    def list_tools(self):
        return self.tools

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.failures:
            self.failures -= 1
            raise TimeoutError("temporary timeout")
        return self.responses.pop(0)

    def close(self):
        self.closed = True


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
class Config:
    retrieval_top_k: int = 3
    llm_answer_strip_mode: str = "tags_only"
    llm_answer_value_fallback: bool = True
    prompt_template: str = "concise"
    llm_model: str = "model"


def _adapter(
    *responses,
    result_mode="answer",
    result_field="answer",
    execution_mode="fixed",
    tools=("search",),
    failures=0,
    continue_on_error=True,
):
    runner = FakeRunner(responses, tools=tools, failures=failures)
    adapter = McpRagAdapter(
        transport="stdio",
        tool_name="search",
        result_mode=result_mode,
        execution_mode=execution_mode,
        timeout_seconds=5,
        question_argument="query",
        top_k_argument="top_k",
        tool_arguments={"collection": "docs"},
        result_field=result_field,
        command="python",
        command_args=["server.py"],
        max_retries=1,
        retry_backoff_seconds=0,
        continue_on_error=continue_on_error,
        session_factory=lambda **kwargs: runner,
    )
    return adapter, runner


def test_prepare_rejects_missing_tool_and_closes_session():
    adapter, runner = _adapter(tools=("other",))

    with pytest.raises(ValueError, match="server tools: other"):
        adapter.prepare(Config(), [])

    assert runner.closed is True


def test_answer_mode_reuses_session_and_records_cold_and_warm_calls():
    adapter, runner = _adapter(
        FakeToolResult(structured={"answer": "42", "source": "NF-001.md"}),
        FakeToolResult(structured={"answer": "43", "source": "NF-002.md"}),
    )
    adapter.prepare(Config(), [])

    first = adapter.answer({"question": "First?"}, Config())
    second = adapter.answer({"question": "Second?"}, Config())

    assert first.answer == "42"
    assert second.answer == "43"
    assert first.metadata[0]["doc_id"] == "NF-001"
    assert first.diagnostics["cold_start"] is True
    assert first.diagnostics["connection_seconds"] == 0.125
    assert second.diagnostics["cold_start"] is False
    assert second.diagnostics["connection_seconds"] == 0.0
    assert len(runner.calls) == 2


def test_context_mode_generates_with_mcp_evidence(monkeypatch):
    result = FakeToolResult(
        structured={
            "contexts": [{"text": "Evidence", "source": "NF-001.md", "score": 2.5}]
        }
    )
    adapter, _ = _adapter(result, result_mode="context", result_field="contexts")
    adapter.llm = object()
    adapter.prompt_template = PromptTemplate("test", "system", "{context} {question}")
    adapter.prepare(Config(), [])
    captured = {}

    def fake_generate(llm, question, contexts, **kwargs):
        captured.update(llm=llm, question=question, contexts=contexts, kwargs=kwargs)
        return GenerationResult(
            answer="42",
            ttft_seconds=0.1,
            total_seconds=0.2,
            token_count=1,
            tokens_per_second=5.0,
            gpu_usage=None,
            input_tokens=10,
            output_tokens=1,
            total_tokens=11,
            raw_content="42",
        )

    monkeypatch.setattr("benchmark.adapters.mcp.generate_answer", fake_generate)

    output = adapter.answer({"question": "What?", "ground_truth": "42"}, Config())

    assert captured["contexts"] == ["Evidence"]
    assert output.contexts == ["Evidence"]
    assert output.metadata[0]["doc_id"] == "NF-001"
    assert output.metadata[0]["score"] == 2.5
    assert output.input_tokens == 10
    assert output.diagnostics["generation_seconds"] == 0.2


def test_timeout_is_retried_and_counted():
    adapter, runner = _adapter(FakeToolResult(structured={"answer": "42"}), failures=1)
    adapter.prepare(Config(), [])

    output = adapter.answer({"question": "What?"}, Config())

    assert output.answer == "42"
    assert output.diagnostics["retry_count"] == 1
    assert output.diagnostics["tool_calls"][0]["timeout_attempts"] == 1
    assert "TimeoutError" in output.diagnostics["tool_calls"][0]["attempt_errors"][0]
    assert len(runner.calls) == 2


def test_tool_errors_are_recorded_without_fabricating_answer():
    adapter, _ = _adapter(
        FakeToolResult(texts=["access denied"], is_error=True),
        continue_on_error=True,
    )
    adapter.prepare(Config(), [])

    output = adapter.answer({"question": "What?"}, Config())

    assert output.answer == ""
    assert output.answer_valid is False
    assert output.diagnostics["error_type"] == "RuntimeError"
    assert output.diagnostics["partial_completion"] is True


def test_agentic_mode_can_call_multiple_discovered_tools():
    search_result = FakeToolResult(
        structured={"contexts": [{"text": "Evidence", "source": "NF-001.md"}]}
    )
    adapter, runner = _adapter(
        search_result,
        result_mode="context",
        result_field="contexts",
        execution_mode="agentic",
        tools=("search", "lookup"),
    )
    adapter.llm = FakeAgentModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search", "args": {"query": "capacity"}, "id": "1"}
                ],
            ),
            AIMessage(
                content="42",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 1,
                    "total_tokens": 11,
                },
            ),
        ]
    )
    adapter.prepare(Config(), [])

    output = adapter.answer({"question": "What capacity?"}, Config())

    assert output.answer == "42"
    assert output.contexts == ["Evidence"]
    assert output.metadata[0]["doc_id"] == "NF-001"
    assert output.diagnostics["agent_rounds"] == 2
    assert runner.calls[0][1]["top_k"] == 3
    assert {schema["name"] for schema in adapter.llm.schemas} == {"search", "lookup"}


def _evidence(text="Evidence"):
    return FakeToolResult(structured={"contexts": [{"text": text}]})


def _call(name, call_id, **args):
    return {"name": name, "args": args, "id": call_id}


def _agentic_adapter(
    *tool_results, model_responses, tools=("search", "lookup"), schemas=None
):
    adapter, runner = _adapter(
        *tool_results,
        result_mode="context",
        result_field="contexts",
        execution_mode="agentic",
        tools=tools,
    )
    for tool in runner.tools:
        if schemas and tool.name in schemas:
            tool.inputSchema = schemas[tool.name]
    adapter.llm = FakeAgentModel(model_responses)
    adapter.prepare(Config(), [])
    return adapter, runner


def test_aggregate_reports_tool_calls_and_agent_rounds_per_question():
    adapter, _ = _agentic_adapter(
        _evidence(),
        _evidence(),
        _evidence(),
        _evidence(),
        model_responses=[
            # Question 1: one call, answered in round 2.
            AIMessage(content="", tool_calls=[_call("search", "1", query="a")]),
            AIMessage(content="first"),
            # Question 2: two calls, then one more, answered in round 3.
            AIMessage(
                content="",
                tool_calls=[
                    _call("search", "2", query="b"),
                    _call("lookup", "3", id="x"),
                ],
            ),
            AIMessage(content="", tool_calls=[_call("lookup", "4", id="y")]),
            AIMessage(content="second"),
        ],
    )

    outputs = [
        adapter.answer({"question": "One?"}, Config()),
        adapter.answer({"question": "Two?"}, Config()),
    ]
    metrics = adapter.aggregate_metrics([o.diagnostics for o in outputs])

    assert metrics["tool_calls_per_sample_mean"] == 2.0
    assert metrics["tool_calls_per_sample_max"] == 3
    assert metrics["agent_rounds_mean"] == 2.5
    assert metrics["agent_rounds_max"] == 3


def test_aggregate_counts_repeated_identical_calls_within_a_question():
    adapter, _ = _agentic_adapter(
        _evidence(),
        _evidence(),
        _evidence(),
        _evidence(),
        model_responses=[
            AIMessage(content="", tool_calls=[_call("search", "1", query="a")]),
            # Same tool + same arguments again: a loop symptom.
            AIMessage(content="", tool_calls=[_call("search", "2", query="a")]),
            # Same tool, different arguments: legitimate follow-up.
            AIMessage(content="", tool_calls=[_call("search", "3", query="b")]),
            AIMessage(content="first"),
            # A new question may repeat the earlier question's call.
            AIMessage(content="", tool_calls=[_call("search", "4", query="a")]),
            AIMessage(content="second"),
        ],
    )

    outputs = [
        adapter.answer({"question": "One?"}, Config()),
        adapter.answer({"question": "Two?"}, Config()),
    ]
    metrics = adapter.aggregate_metrics([o.diagnostics for o in outputs])

    assert metrics["duplicate_tool_call_count"] == 1
    assert metrics["duplicate_tool_call_rate"] == 0.25


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_aggregate_reports_tool_latency_percentiles(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr("benchmark.adapters.mcp.time", clock)
    durations = [float(seconds) for seconds in range(1, 11)]
    adapter, runner = _adapter(
        *[FakeToolResult(structured={"answer": "42"}) for _ in durations]
    )
    original_call = runner.call_tool

    def timed_call(name, arguments):
        clock.now += durations[len(runner.calls)]
        return original_call(name, arguments)

    runner.call_tool = timed_call
    adapter.prepare(Config(), [])

    diagnostics = [
        adapter.answer({"question": f"Q{i}?"}, Config()).diagnostics
        for i in range(len(durations))
    ]
    metrics = adapter.aggregate_metrics(diagnostics)

    # Nearest-rank percentiles over the per-call latencies 1s..10s.
    assert metrics["tool_latency_p50_seconds"] == 5.0
    assert metrics["tool_latency_p95_seconds"] == 10.0
    assert metrics["tool_latency_max_seconds"] == 10.0


def test_aggregate_separates_tool_errors_from_transport_failures():
    # Two failing attempts exhaust max_retries=1 for the first question.
    adapter, _ = _adapter(
        FakeToolResult(texts=["access denied"], is_error=True),
        FakeToolResult(structured={"answer": "42"}),
        failures=2,
    )
    adapter.prepare(Config(), [])

    diagnostics = [
        adapter.answer({"question": f"Q{i}?"}, Config()).diagnostics
        for i in range(3)
    ]
    metrics = adapter.aggregate_metrics(diagnostics)

    assert [d.get("failure_kind") for d in diagnostics] == [
        "transport",
        "tool_error",
        None,
    ]
    assert metrics["transport_failure_count"] == 1
    assert metrics["tool_error_count"] == 1
    assert metrics["agent_failure_count"] == 0
    assert metrics["failure_count"] == 2


def test_aggregate_counts_agent_failures_separately():
    adapter, _ = _agentic_adapter(
        model_responses=[
            AIMessage(content="", tool_calls=[_call("delete_everything", "1")]),
        ],
        tools=("search",),
    )

    output = adapter.answer({"question": "What?"}, Config())
    metrics = adapter.aggregate_metrics([output.diagnostics])

    assert output.diagnostics["failure_kind"] == "agent"
    assert metrics["agent_failure_count"] == 1
    assert metrics["transport_failure_count"] == 0


def test_schema_invalid_arguments_are_recorded_but_still_sent():
    search_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
        "required": ["query"],
    }
    adapter, runner = _agentic_adapter(
        _evidence(),
        _evidence(),
        model_responses=[
            # Wrong argument name: the schema requires "query".
            AIMessage(content="", tool_calls=[_call("search", "1", q="capacity")]),
            AIMessage(content="", tool_calls=[_call("search", "2", query="capacity")]),
            AIMessage(content="42"),
        ],
        schemas={"search": search_schema},
    )

    output = adapter.answer({"question": "What capacity?"}, Config())
    metrics = adapter.aggregate_metrics([output.diagnostics])

    calls = output.diagnostics["tool_calls"]
    assert [call["schema_valid"] for call in calls] == [False, True]
    assert "query" in calls[0]["schema_errors"][0]
    assert len(runner.calls) == 2
    assert metrics["schema_invalid_call_count"] == 1
    assert metrics["schema_invalid_call_rate"] == 0.5


def test_agentic_mode_reports_tool_schema_overhead_per_model_call():
    adapter, _ = _agentic_adapter(
        _evidence(),
        model_responses=[
            AIMessage(content="", tool_calls=[_call("search", "1", query="a")]),
            AIMessage(content="42"),
        ],
        tools=("search",),
    )
    # Compact JSON of the tool definitions the model sees on every call.
    exposed = (
        '[{"description":"search tool","name":"search",'
        '"parameters":{"properties":{},"type":"object"}}]'
    )

    output = adapter.answer({"question": "What?"}, Config())
    metrics = adapter.aggregate_metrics([output.diagnostics])

    assert output.diagnostics["model_calls"] == 2
    assert output.diagnostics["tool_schema_chars"] == 2 * len(exposed)
    assert metrics["tool_schema_chars_total"] == 2 * len(exposed)
    assert metrics["tool_schema_chars_per_sample_mean"] == 2 * len(exposed)


def test_fixed_mode_aggregate_has_no_agent_round_metrics():
    adapter, _ = _adapter(FakeToolResult(structured={"answer": "42"}))
    adapter.prepare(Config(), [])

    output = adapter.answer({"question": "What?"}, Config())
    metrics = adapter.aggregate_metrics([output.diagnostics])

    assert metrics["tool_calls_per_sample_mean"] == 1.0
    assert metrics["agent_rounds_mean"] is None
    assert metrics["tool_schema_chars_total"] is None


def test_agentic_mode_rejects_disallowed_tool_call():
    adapter, _ = _adapter(
        result_mode="context",
        result_field="contexts",
        execution_mode="agentic",
        tools=("search",),
        continue_on_error=True,
    )
    adapter.llm = FakeAgentModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "delete_everything", "args": {}, "id": "1"}],
            )
        ]
    )
    adapter.prepare(Config(), [])

    output = adapter.answer({"question": "What?"}, Config())

    assert output.answer_valid is False
    assert "disallowed" in output.diagnostics["error"]
