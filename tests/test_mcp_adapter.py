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
