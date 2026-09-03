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
    assert output.diagnostics["agent_rounds"] == 1
    assert output.diagnostics["agent_tools_used"] == []
    assert output.diagnostics["agent_retrieved"] is False


def test_agentic_mode_records_round_accounting():
    search_result = FakeToolResult(
        structured={"contexts": [{"text": "Evidence", "source": "NF-001.md"}]}
    )
    adapter, _ = _adapter(
        search_result,
        result_mode="context",
        result_field="contexts",
        execution_mode="agentic",
        tools=("search",),
    )
    adapter.llm = FakeAgentModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search", "args": {"query": "capacity"}, "id": "1"}
                ],
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 1,
                    "total_tokens": 11,
                },
            ),
            AIMessage(
                content="42",
                usage_metadata={
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "total_tokens": 7,
                },
            ),
        ]
    )
    adapter.prepare(Config(), [])

    output = adapter.answer({"question": "What capacity?"}, Config())

    diagnostics = output.diagnostics
    assert diagnostics["agent_rounds"] == 2
    assert diagnostics["agent_tool_calls"] == 1
    assert diagnostics["agent_tools_used"] == ["search"]
    assert diagnostics["agent_round_log"] == [{"round": 1, "tools": ["search"]}]
    assert diagnostics["agent_retrieved"] is True
    assert diagnostics["agent_rounds_exhausted"] is False
    assert diagnostics["agent_tokens_input"] == 15
    assert diagnostics["agent_tokens_output"] == 3
    assert diagnostics["agent_tokens_total"] == 18
    assert diagnostics["agent_model_seconds"] >= 0.0


def test_agentic_mode_records_exhausted_rounds():
    adapter, _ = _adapter(
        FakeToolResult(structured={"contexts": [{"text": "Evidence"}]}),
        FakeToolResult(structured={"contexts": [{"text": "Evidence"}]}),
        result_mode="context",
        result_field="contexts",
        execution_mode="agentic",
        tools=("search",),
    )
    loop_response = AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {"query": "x"}, "id": "1"}],
        usage_metadata={
            "input_tokens": 4,
            "output_tokens": 1,
            "total_tokens": 5,
        },
    )
    adapter.llm = FakeAgentModel([loop_response, loop_response])
    adapter.max_agent_rounds = 2
    adapter.prepare(Config(), [])

    output = adapter.answer({"question": "What?"}, Config())

    assert output.answer_valid is False
    assert "exceeded 2 tool rounds" in output.diagnostics["error"]
    assert output.diagnostics["agent_rounds"] == 2
    assert output.diagnostics["agent_rounds_exhausted"] is True
    assert output.diagnostics["partial_completion"] is True
    assert output.diagnostics["agent_retrieved"] is True
    assert output.diagnostics["agent_tokens_total"] == 10
    assert output.diagnostics["agent_round_log"] == [
        {"round": 1, "tools": ["search"]},
        {"round": 2, "tools": ["search"]},
    ]


def test_aggregate_agent_metrics_merges_into_run_metrics():
    from benchmark.adapters.mcp import aggregate_agent_metrics

    fixed_diagnostics = {
        "execution_mode": "fixed",
        "tool_calls": [{"tool": "search", "total_seconds": 0.5}],
    }
    agentic_diagnostics = [
        {
            "execution_mode": "agentic",
            "agent_rounds": 2,
            "agent_tool_calls": 1,
            "agent_retrieved": True,
            "agent_rounds_exhausted": False,
            "agent_tokens_total": 18,
            "agent_model_seconds": 1.0,
            "tool_calls": [{"tool": "search", "total_seconds": 0.3}],
        },
        {
            "execution_mode": "agentic",
            "agent_rounds": 4,
            "agent_tool_calls": 4,
            "agent_retrieved": False,
            "agent_rounds_exhausted": True,
            "agent_tokens_total": 30,
            "agent_model_seconds": 2.0,
            "error": "exceeded",
            "tool_calls": [{"tool": "search", "total_seconds": 0.7}],
        },
    ]

    empty = aggregate_agent_metrics([fixed_diagnostics])
    assert empty == {}

    metrics = aggregate_agent_metrics([fixed_diagnostics, *agentic_diagnostics])
    assert metrics["agent_sample_count"] == 2
    assert metrics["agent_rounds_mean"] == 3.0
    assert metrics["agent_rounds_max"] == 4
    assert metrics["agent_tool_calls_mean"] == 2.5
    assert metrics["agent_rounds_exhausted_count"] == 1
    assert metrics["agent_no_retrieval_count"] == 1
    assert metrics["agent_error_count"] == 1
    assert metrics["agent_tokens_total"] == 48
    assert metrics["agent_tokens_mean"] == 24.0
    assert metrics["agent_model_seconds"] == 3.0
    assert metrics["agent_tool_seconds"] == pytest.approx(1.0)


def test_agentic_uses_configured_system_prompt():
    from langchain_core.messages import AIMessage

    adapter, _ = _adapter(
        result_mode="context",
        result_field="contexts",
        execution_mode="agentic",
        tools=("search",),
    )
    adapter.agent_system_prompt = "Always search exactly once."
    adapter.llm = FakeAgentModel([AIMessage(content="42")])
    adapter.prepare(Config(), [])
    adapter.answer({"question": "What?"}, Config())
    # bind_tools was called; the prompt is only visible via the first message
    # of the model invocation, which FakeAgentModel discards — so instead
    # assert via a capturing model.
    captured = {}

    class CapturingModel(FakeAgentModel):
        def invoke(self, messages):
            captured["system"] = str(messages[0].content)
            return super().invoke(messages)

    adapter.llm = CapturingModel([AIMessage(content="42")])
    adapter.answer({"question": "What?"}, Config())
    assert captured["system"] == "Always search exactly once."


def test_agentic_require_retrieval_fails_when_agent_skips_tools():
    from langchain_core.messages import AIMessage

    adapter, _ = _adapter(
        result_mode="context",
        result_field="contexts",
        execution_mode="agentic",
        tools=("search",),
    )
    adapter.agent_require_retrieval = True
    adapter.llm = FakeAgentModel(
        [AIMessage(content="I know this one: 42", usage_metadata={
            "input_tokens": 1, "output_tokens": 1, "total_tokens": 2})]
    )
    adapter.prepare(Config(), [])

    output = adapter.answer({"question": "What?"}, Config())

    assert output.answer_valid is False
    assert output.diagnostics["agent_skipped_retrieval"] is True
    assert "without retrieving" in output.diagnostics["error"]
    assert output.diagnostics["agent_retrieved"] is False


def test_agentic_require_retrieval_allows_real_retrieval():
    from langchain_core.messages import AIMessage

    adapter, _ = _adapter(
        FakeToolResult(structured={"contexts": [{"text": "Evidence"}]}),
        result_mode="context",
        result_field="contexts",
        execution_mode="agentic",
        tools=("search",),
    )
    adapter.agent_require_retrieval = True
    adapter.llm = FakeAgentModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "search", "args": {"query": "x"}, "id": "1"}],
            ),
            AIMessage(content="42"),
        ]
    )
    adapter.prepare(Config(), [])

    output = adapter.answer({"question": "What?"}, Config())

    assert output.answer == "42"
    assert output.diagnostics.get("agent_skipped_retrieval", False) is False
