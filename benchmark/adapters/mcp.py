"""Model Context Protocol adapter for fixed-tool and agentic benchmarks."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from benchmark.adapters.base import AdapterCapabilities, RagSystemOutput
from benchmark.costing import estimate_cost_usd
from benchmark.generation import extract_concise_fallback, generate_answer, get_llm
from benchmark.prompt_templates import get_template

logger = logging.getLogger(__name__)

_SENSITIVE_ARGUMENT_MARKERS = ("authorization", "password", "secret", "token", "key")
_SAFE_CHILD_ENV = ("PATH", "LANG", "LC_ALL", "PYTHONPATH", "VIRTUAL_ENV")


class McpToolCallFailure(RuntimeError):
    """A transport-level tool failure with benchmark provenance attached."""

    def __init__(self, message: str, provenance: dict[str, Any]) -> None:
        super().__init__(message)
        self.provenance = provenance
        self.timed_out = bool(provenance.get("timeout_attempts"))


def _json_object(value: str | None, name: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


def _json_list(value: str | None, name: str) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise ValueError(f"{name} must be a JSON array of strings")
    return parsed


def _lookup(value: Any, path: str | None) -> Any:
    if not path:
        return value
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _as_texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            texts.extend(_as_texts(item))
        return texts
    if isinstance(value, dict):
        for key in ("text", "content", "page_content", "context", "answer"):
            if key in value:
                return _as_texts(value[key])
        return [json.dumps(value, ensure_ascii=False, sort_keys=True)]
    return [str(value)]


def _as_metadata(
    value: Any, count: int, tool_name: str, transport: str
) -> list[dict[str, Any]]:
    items = value if isinstance(value, list) else []
    metadata: list[dict[str, Any]] = []
    for rank in range(1, count + 1):
        item = items[rank - 1] if rank <= len(items) else None
        entry: dict[str, Any] = {
            "rank": rank,
            "mcp_tool": tool_name,
            "mcp_transport": transport,
        }
        if isinstance(item, dict):
            entry.update(
                {
                    key: val
                    for key, val in item.items()
                    if key
                    not in {"text", "content", "page_content", "context", "answer"}
                }
            )
            source = item.get("source") or item.get("source_id") or item.get("doc_id")
            if source:
                source_id = Path(str(source)).stem
                entry.setdefault("source_id", source_id)
                entry.setdefault("doc_id", source_id)
        metadata.append(entry)
    return metadata


def _serialize_result(result: Any) -> dict[str, Any]:
    model_dump = getattr(result, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json", by_alias=True, exclude_none=True)
    return {"content": str(result)}


def _result_payload(result: Any, field_path: str | None) -> Any:
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        selected = _lookup(structured, field_path)
        if selected is not None:
            return selected
    return [
        str(item.text)
        for item in getattr(result, "content", ())
        if getattr(item, "text", None) is not None
    ]


def _tool_is_error(result: Any) -> bool:
    return bool(getattr(result, "isError", False) or getattr(result, "is_error", False))


def _redact_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            "<redacted>"
            if any(marker in key.lower() for marker in _SENSITIVE_ARGUMENT_MARKERS)
            else value
        )
        for key, value in arguments.items()
    }


def _message_text(message: Any) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content or "")


def _message_usage(message: Any) -> tuple[int, int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0)
    if not total_tokens:
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


def aggregate_agent_metrics(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-sample agentic diagnostics into run-level agent metrics."""
    agent_items = [
        item for item in diagnostics if item.get("execution_mode") == "agentic"
    ]
    if not agent_items:
        return {}
    rounds = [int(item.get("agent_rounds", 0) or 0) for item in agent_items]
    tool_call_counts = [
        int(item.get("agent_tool_calls", 0) or 0) for item in agent_items
    ]
    tool_seconds = [
        sum(float(call.get("total_seconds", 0.0)) for call in item.get("tool_calls", []))
        for item in agent_items
    ]
    tokens = [int(item.get("agent_tokens_total", 0) or 0) for item in agent_items]
    return {
        "agent_sample_count": len(agent_items),
        "agent_rounds_mean": sum(rounds) / len(rounds),
        "agent_rounds_max": max(rounds),
        "agent_tool_calls_mean": sum(tool_call_counts) / len(tool_call_counts),
        "agent_tool_calls_max": max(tool_call_counts),
        "agent_rounds_exhausted_count": sum(
            bool(item.get("agent_rounds_exhausted")) for item in agent_items
        ),
        "agent_no_retrieval_count": sum(
            not bool(item.get("agent_retrieved")) for item in agent_items
        ),
        "agent_error_count": sum(bool(item.get("error")) for item in agent_items),
        "agent_tokens_total": sum(tokens),
        "agent_tokens_mean": sum(tokens) / len(tokens),
        "agent_model_seconds": sum(
            float(item.get("agent_model_seconds", 0.0) or 0.0)
            for item in agent_items
        ),
        "agent_tool_seconds": sum(tool_seconds),
    }


class PersistentMcpSession:
    """Own an MCP session on one background event loop for synchronous callers."""

    def __init__(
        self,
        *,
        transport: str,
        timeout_seconds: float,
        server_url: str | None,
        http_headers: dict[str, str] | None,
        command: str | None,
        command_args: list[str],
        env: dict[str, str] | None,
    ) -> None:
        self.transport = transport
        self.timeout_seconds = timeout_seconds
        self.server_url = server_url
        self.http_headers = http_headers
        self.command = command
        self.command_args = command_args
        self.env = env
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop_event: asyncio.Event | None = None
        self._lifecycle_error: BaseException | None = None
        self._session: Any = None

    def start(self) -> float:
        if self._thread is not None:
            return 0.0
        started = time.perf_counter()
        self._thread = threading.Thread(
            target=self._thread_main,
            name="mcp-session-loop",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(self.timeout_seconds):
            raise TimeoutError("Timed out starting MCP event loop")
        if self._lifecycle_error is not None:
            self._thread.join(timeout=self.timeout_seconds)
            self._loop = None
            self._thread = None
            raise self._lifecycle_error
        return time.perf_counter() - started

    def run(self, awaitable: Any) -> Any:
        if self._loop is None:
            close = getattr(awaitable, "close", None)
            if callable(close):
                close()
            raise RuntimeError("MCP session is not running")
        future = asyncio.run_coroutine_threadsafe(awaitable, self._loop)
        try:
            return future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"MCP operation exceeded {self.timeout_seconds:g}s"
            ) from exc

    def list_tools(self) -> list[Any]:
        return self.run(self._list_tools())

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return self.run(self._call_tool(name, arguments))

    def close(self) -> None:
        loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return
        if self._stop_event is not None:
            loop.call_soon_threadsafe(self._stop_event.set)
        thread.join(timeout=self.timeout_seconds)
        if thread.is_alive():
            raise TimeoutError("Timed out closing persistent MCP session")
        self._loop = None
        self._thread = None
        if self._lifecycle_error is not None:
            raise self._lifecycle_error

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            self._stop_event = asyncio.Event()
            loop.run_until_complete(self._open())
        except BaseException as exc:
            self._lifecycle_error = exc
            self._ready.set()
        finally:
            loop.close()

    async def _open(self) -> None:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError(
                "The MCP adapter requires the 'mcp' dependency. "
                "Install project requirements first."
            ) from exc
        try:
            from mcp.client.streamable_http import streamable_http_client
        except ImportError:
            # SDK 1.12 used the unseparated spelling; newer 1.x releases expose
            # streamable_http_client. Supporting both keeps the declared range valid.
            from mcp.client.streamable_http import (
                streamablehttp_client as streamable_http_client,
            )

        async with AsyncExitStack() as stack:
            if self.transport == "stdio":
                if not self.command:
                    raise ValueError("MCP_COMMAND is required for stdio transport")
                params = StdioServerParameters(
                    command=self.command,
                    args=self.command_args,
                    env=self.env,
                )
                transport_context = stdio_client(params)
            else:
                if not self.server_url:
                    raise ValueError(
                        "MCP_SERVER_URL is required for streamable_http transport"
                    )
                if self.http_headers:
                    import httpx

                    client = await stack.enter_async_context(
                        httpx.AsyncClient(
                            headers=self.http_headers,
                            timeout=self.timeout_seconds,
                        )
                    )
                    transport_context = streamable_http_client(
                        self.server_url, http_client=client
                    )
                else:
                    transport_context = streamable_http_client(self.server_url)

            streams = await stack.enter_async_context(transport_context)
            session = await stack.enter_async_context(
                ClientSession(
                    streams[0],
                    streams[1],
                    read_timeout_seconds=timedelta(seconds=self.timeout_seconds),
                )
            )
            await session.initialize()
            self._session = session
            self._ready.set()
            if self._stop_event is None:
                raise RuntimeError("MCP stop event was not initialized")
            await self._stop_event.wait()
        self._session = None

    async def _list_tools(self) -> list[Any]:
        if self._session is None:
            raise RuntimeError("MCP session has not been initialized")
        tools: list[Any] = []
        cursor = None
        while True:
            response = await self._session.list_tools(cursor=cursor)
            tools.extend(response.tools)
            cursor = getattr(response, "nextCursor", None) or getattr(
                response, "next_cursor", None
            )
            if not cursor:
                return tools

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self._session is None:
            raise RuntimeError("MCP session has not been initialized")
        return await self._session.call_tool(name, arguments=arguments)


@dataclass
class McpRagAdapter:
    """Benchmark a fixed MCP tool or an LLM-controlled multi-tool MCP agent."""

    transport: str
    tool_name: str
    result_mode: str
    execution_mode: str
    timeout_seconds: float
    question_argument: str
    top_k_argument: str | None
    tool_arguments: dict[str, Any]
    result_field: str | None
    allowed_tools: tuple[str, ...] = ()
    max_agent_rounds: int = 4
    max_retries: int = 1
    retry_backoff_seconds: float = 0.25
    continue_on_error: bool = True
    server_url: str | None = None
    http_headers: dict[str, str] | None = None
    command: str | None = None
    command_args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    llm: Any = None
    prompt_template: Any = None
    agent_system_prompt: str | None = None
    agent_require_retrieval: bool = False
    session_factory: Any = PersistentMcpSession

    name: str = "mcp"
    _runner: PersistentMcpSession | None = field(default=None, init=False, repr=False)
    _tools: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _connection_seconds: float = field(default=0.0, init=False, repr=False)
    _sample_number: int = field(default=0, init=False, repr=False)

    @classmethod
    def from_config(cls, config: Any) -> "McpRagAdapter":
        env_names = [
            item.strip() for item in str(config.mcp_env_vars).split(",") if item.strip()
        ]
        missing = [name for name in env_names if name not in os.environ]
        if missing:
            raise ValueError(
                "MCP_ENV_VARS references unset variables: " + ", ".join(missing)
            )
        child_env = {
            name: os.environ[name]
            for name in (*_SAFE_CHILD_ENV, *env_names)
            if name in os.environ
        }
        execution_mode = str(config.mcp_execution_mode).strip().lower()
        result_mode = str(config.mcp_result_mode).strip().lower()
        llm = None
        prompt_template = None
        if result_mode == "context" or execution_mode == "agentic":
            llm = get_llm(
                provider=config.llm_provider,
                model_name=config.llm_model,
                base_url=config.llm_base_url(),
                api_key=config.llm_api_key(),
                max_new_tokens=config.max_new_tokens,
            )
            prompt_template = get_template(config.prompt_template)

        return cls(
            transport=str(config.mcp_transport).strip().lower(),
            tool_name=str(config.mcp_tool_name).strip(),
            result_mode=result_mode,
            execution_mode=execution_mode,
            timeout_seconds=float(config.mcp_timeout_seconds),
            question_argument=str(config.mcp_question_argument).strip(),
            top_k_argument=(
                str(config.mcp_top_k_argument).strip()
                if config.mcp_top_k_argument
                else None
            ),
            tool_arguments=_json_object(
                config.mcp_tool_arguments_json, "MCP_TOOL_ARGUMENTS_JSON"
            ),
            result_field=config.mcp_result_field or None,
            allowed_tools=tuple(
                _json_list(config.mcp_allowed_tools_json, "MCP_ALLOWED_TOOLS_JSON")
            ),
            max_agent_rounds=int(config.mcp_max_agent_rounds),
            max_retries=int(config.mcp_max_retries),
            retry_backoff_seconds=float(config.mcp_retry_backoff_seconds),
            continue_on_error=bool(config.mcp_continue_on_error),
            server_url=config.mcp_server_url or None,
            http_headers={
                str(key): str(value)
                for key, value in _json_object(
                    config.mcp_http_headers_json, "MCP_HTTP_HEADERS_JSON"
                ).items()
            }
            or None,
            command=config.mcp_command or None,
            command_args=_json_list(config.mcp_args_json, "MCP_ARGS_JSON"),
            env=child_env or None,
            llm=llm,
            prompt_template=prompt_template,
            agent_system_prompt=(
                str(config.mcp_agent_system_prompt).strip()
                if getattr(config, "mcp_agent_system_prompt", None)
                else None
            ),
            agent_require_retrieval=bool(
                getattr(config, "mcp_agent_require_retrieval", False)
            ),
        )

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            retrieval=self.result_mode == "context" or self.execution_mode == "agentic",
            generation=True,
            references=self.result_mode == "context"
            or self.execution_mode == "agentic",
            token_usage=self.result_mode == "context"
            or self.execution_mode == "agentic",
            cleanup=True,
        )

    def prepare(self, config: Any, data: list[dict], corpus=None) -> None:
        del config, data, corpus
        self._runner = self.session_factory(
            transport=self.transport,
            timeout_seconds=self.timeout_seconds,
            server_url=self.server_url,
            http_headers=self.http_headers,
            command=self.command,
            command_args=self.command_args,
            env=self.env,
        )
        self._connection_seconds = self._runner.start()
        try:
            discovered = self._runner.list_tools()
            self._tools = {tool.name: tool for tool in discovered}
            if self.allowed_tools:
                missing = sorted(set(self.allowed_tools) - set(self._tools))
                if missing:
                    raise ValueError(
                        "MCP_ALLOWED_TOOLS_JSON contains unavailable tools: "
                        + ", ".join(missing)
                    )
                self._tools = {name: self._tools[name] for name in self.allowed_tools}
            if self.execution_mode == "fixed" and self.tool_name not in self._tools:
                available = ", ".join(sorted(self._tools)) or "none"
                raise ValueError(
                    f"MCP tool {self.tool_name!r} is unavailable; "
                    f"server tools: {available}"
                )
        except BaseException:
            self._runner.close()
            self._runner = None
            raise

    def cleanup(self, target: Any, config: Any) -> None:
        del target, config
        if self._runner is not None:
            self._runner.close()
            self._runner = None

    def answer(self, sample: dict, config: Any) -> RagSystemOutput:
        sample_number = self._sample_number
        self._sample_number += 1
        started = time.perf_counter()
        diagnostics: dict[str, Any] = {
            "execution_mode": self.execution_mode,
            "result_mode": self.result_mode,
            "cold_start": sample_number == 0,
            "connection_seconds": (
                self._connection_seconds if sample_number == 0 else 0.0
            ),
            "tool_calls": [],
            "retry_count": 0,
            "partial_completion": False,
        }
        try:
            if self.execution_mode == "agentic":
                output = self._answer_agentic(sample, config, diagnostics)
            else:
                output = self._answer_fixed(sample, config, diagnostics)
            diagnostics["total_seconds"] = time.perf_counter() - started
            output.diagnostics.update(diagnostics)
            return output
        except Exception as exc:
            diagnostics.update(
                {
                    "total_seconds": time.perf_counter() - started,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "timeout": isinstance(exc, TimeoutError)
                    or bool(getattr(exc, "timed_out", False)),
                    "partial_completion": bool(diagnostics["tool_calls"]),
                }
            )
            logger.warning("MCP sample failed: %s", exc)
            if not self.continue_on_error:
                raise
            return RagSystemOutput(
                answer="",
                total_seconds=diagnostics["total_seconds"],
                answer_valid=False,
                diagnostics=diagnostics,
            )

    def _answer_fixed(
        self, sample: dict, config: Any, diagnostics: dict[str, Any]
    ) -> RagSystemOutput:
        arguments = dict(self.tool_arguments)
        arguments[self.question_argument] = str(sample["question"])
        if self.top_k_argument:
            arguments[self.top_k_argument] = int(config.retrieval_top_k)
        try:
            result, provenance = self._call_tool(self.tool_name, arguments)
        except McpToolCallFailure as exc:
            diagnostics["tool_calls"].append(exc.provenance)
            diagnostics["retry_count"] += exc.provenance["attempts"] - 1
            raise
        diagnostics["tool_calls"].append(provenance)
        diagnostics["retry_count"] += provenance["attempts"] - 1
        if _tool_is_error(result):
            raise RuntimeError(self._tool_error_message(self.tool_name, result))

        payload = _result_payload(result, self.result_field)
        texts = _as_texts(payload)
        if not texts:
            diagnostics["empty_response"] = True
        raw_response = _serialize_result(result)
        metadata = _as_metadata(payload, len(texts), self.tool_name, self.transport)
        tool_seconds = float(provenance["total_seconds"])

        if self.result_mode == "answer":
            answer = "\n".join(texts).strip()
            structured = getattr(result, "structuredContent", None) or getattr(
                result, "structured_content", None
            )
            answer_metadata = (
                _as_metadata([structured], 1, self.tool_name, self.transport)
                if isinstance(structured, dict)
                else []
            )
            return RagSystemOutput(
                answer=answer,
                metadata=answer_metadata,
                raw_response={"tool_calls": [raw_response]},
                total_seconds=tool_seconds,
                raw_content=answer,
                answer_valid=bool(answer),
            )

        if self.llm is None or self.prompt_template is None:
            raise RuntimeError("MCP context mode generator was not initialized")
        generated = generate_answer(
            self.llm,
            str(sample["question"]),
            texts,
            system_prompt=self.prompt_template.system_prompt,
            human_template=self.prompt_template.human_template,
            strip_mode=config.llm_answer_strip_mode,
            value_fallback=config.llm_answer_value_fallback,
            ground_truth=sample.get("ground_truth"),
            prompt_template_name=config.prompt_template,
            cost_model_name=config.llm_model,
        )
        diagnostics["generation_seconds"] = generated.total_seconds
        return RagSystemOutput(
            answer=generated.answer,
            contexts=texts,
            metadata=metadata,
            raw_response={"tool_calls": [raw_response]},
            ttft_seconds=tool_seconds + generated.ttft_seconds,
            total_seconds=tool_seconds + generated.total_seconds,
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
        )

    def _answer_agentic(
        self, sample: dict, config: Any, diagnostics: dict[str, Any]
    ) -> RagSystemOutput:
        if self.llm is None:
            raise RuntimeError("MCP agentic mode LLM was not initialized")
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        schemas = [self._langchain_tool_schema(tool) for tool in self._tools.values()]
        model = self.llm.bind_tools(schemas)
        messages: list[Any] = [
            SystemMessage(
                content=self.agent_system_prompt
                or (
                    "Use the available MCP tools when evidence is needed. "
                    "Return only the final answer when you have enough evidence."
                )
            ),
            HumanMessage(content=str(sample["question"])),
        ]
        contexts: list[str] = []
        metadata: list[dict[str, Any]] = []
        raw_results: list[dict[str, Any]] = []
        input_tokens = output_tokens = total_tokens = 0
        model_seconds = 0.0
        final_text = ""
        rounds_used = 0
        round_log: list[dict[str, Any]] = []

        try:
            for round_number in range(1, self.max_agent_rounds + 1):
                rounds_used = round_number
                model_started = time.perf_counter()
                response = model.invoke(messages)
                model_seconds += time.perf_counter() - model_started
                usage = _message_usage(response)
                input_tokens += usage[0]
                output_tokens += usage[1]
                total_tokens += usage[2]
                messages.append(response)
                tool_calls = list(getattr(response, "tool_calls", ()) or ())
                if not tool_calls:
                    final_text = _message_text(response).strip()
                    break

                round_tools: list[str] = []
                round_log.append(
                    {"round": round_number, "tools": round_tools}
                )
                for tool_call in tool_calls:
                    name = str(tool_call.get("name", ""))
                    round_tools.append(name)
                    arguments = tool_call.get("args", {})
                    if name not in self._tools:
                        raise ValueError(
                            f"LLM requested disallowed MCP tool {name!r}"
                        )
                    if not isinstance(arguments, dict):
                        raise ValueError(
                            f"MCP tool arguments for {name!r} must be an object"
                        )
                    effective_arguments = {**self.tool_arguments, **arguments}
                    if name == self.tool_name and self.top_k_argument:
                        effective_arguments.setdefault(
                            self.top_k_argument, int(config.retrieval_top_k)
                        )
                    try:
                        result, provenance = self._call_tool(name, effective_arguments)
                    except McpToolCallFailure as exc:
                        diagnostics["tool_calls"].append(exc.provenance)
                        diagnostics["retry_count"] += exc.provenance["attempts"] - 1
                        raise
                    diagnostics["tool_calls"].append(provenance)
                    diagnostics["retry_count"] += provenance["attempts"] - 1
                    if _tool_is_error(result):
                        raise RuntimeError(self._tool_error_message(name, result))
                    raw = _serialize_result(result)
                    raw_results.append(raw)
                    payload = _result_payload(result, self.result_field)
                    call_contexts = _as_texts(payload)
                    offset = len(contexts)
                    contexts.extend(call_contexts)
                    for item in _as_metadata(
                        payload, len(call_contexts), name, self.transport
                    ):
                        item["rank"] = offset + int(item["rank"])
                        metadata.append(item)
                    messages.append(
                        ToolMessage(
                            content=json.dumps(raw, ensure_ascii=False, default=str),
                            tool_call_id=str(tool_call.get("id", name)),
                        )
                    )
            else:
                diagnostics["partial_completion"] = bool(diagnostics["tool_calls"])
                diagnostics["agent_rounds_exhausted"] = True
                raise RuntimeError(
                    f"MCP agent exceeded {self.max_agent_rounds} tool rounds"
                )
        finally:
            diagnostics["agent_rounds"] = rounds_used
            diagnostics["agent_tool_calls"] = len(diagnostics["tool_calls"])
            diagnostics["agent_round_log"] = round_log
            diagnostics["agent_tools_used"] = sorted(
                {str(call.get("tool", "unknown")) for call in diagnostics["tool_calls"]}
            )
            diagnostics["agent_retrieved"] = bool(contexts)
            diagnostics["agent_model_seconds"] = model_seconds
            diagnostics["agent_tokens_input"] = input_tokens
            diagnostics["agent_tokens_output"] = output_tokens
            diagnostics["agent_tokens_total"] = total_tokens
            diagnostics.setdefault("agent_rounds_exhausted", False)

        if not final_text:
            diagnostics["empty_response"] = True
        if self.agent_require_retrieval and not contexts:
            diagnostics["agent_skipped_retrieval"] = True
            raise RuntimeError(
                "MCP agent answered without retrieving "
                "(mcp_agent_require_retrieval=true)"
            )
        answer = final_text
        if config.llm_answer_value_fallback and answer:
            answer = extract_concise_fallback(answer) or answer
        diagnostics["generation_seconds"] = model_seconds
        elapsed = model_seconds + sum(
            float(call["total_seconds"]) for call in diagnostics["tool_calls"]
        )
        return RagSystemOutput(
            answer=answer,
            contexts=contexts,
            metadata=metadata,
            raw_response={"tool_calls": raw_results},
            ttft_seconds=elapsed,
            total_seconds=elapsed,
            token_count=output_tokens,
            tokens_per_second=(
                output_tokens / elapsed if output_tokens and elapsed else 0.0
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimate_cost_usd(
                model_name=config.llm_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            raw_content=final_text,
            answer_valid=bool(answer),
        )

    def _call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        if self._runner is None:
            raise RuntimeError("MCP adapter has not been prepared")
        attempt_timings: list[float] = []
        attempt_errors: list[str] = []
        timeout_attempts = 0
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                result = self._runner.call_tool(name, arguments)
                attempt_timings.append(time.perf_counter() - started)
                return result, {
                    "tool": name,
                    "arguments": _redact_arguments(arguments),
                    "attempts": attempt + 1,
                    "attempt_seconds": attempt_timings,
                    "total_seconds": sum(attempt_timings),
                    "is_error": _tool_is_error(result),
                    "attempt_errors": attempt_errors,
                    "timeout_attempts": timeout_attempts,
                }
            except Exception as exc:
                attempt_timings.append(time.perf_counter() - started)
                attempt_errors.append(f"{type(exc).__name__}: {exc}")
                timeout_attempts += int(isinstance(exc, TimeoutError))
                if attempt >= self.max_retries:
                    provenance = {
                        "tool": name,
                        "arguments": _redact_arguments(arguments),
                        "attempts": attempt + 1,
                        "attempt_seconds": attempt_timings,
                        "total_seconds": sum(attempt_timings),
                        "is_error": True,
                        "attempt_errors": attempt_errors,
                        "timeout_attempts": timeout_attempts,
                    }
                    raise McpToolCallFailure(str(exc), provenance) from exc
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise RuntimeError("unreachable")

    def _tool_error_message(self, name: str, result: Any) -> str:
        error_text = "\n".join(_as_texts(_result_payload(result, None)))
        return f"MCP tool {name!r} failed: {error_text or 'unknown error'}"

    @staticmethod
    def _langchain_tool_schema(tool: Any) -> dict[str, Any]:
        schema = getattr(tool, "inputSchema", None) or getattr(
            tool, "input_schema", None
        )
        return {
            "name": tool.name,
            "description": getattr(tool, "description", None) or "MCP tool",
            "parameters": schema or {"type": "object", "properties": {}},
        }
