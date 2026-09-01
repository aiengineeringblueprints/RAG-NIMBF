from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from types import ModuleType, SimpleNamespace

from benchmark.adapters.mcp import PersistentMcpSession


def test_streamable_http_session_is_initialized_once_and_reused(monkeypatch):
    events = []

    class FakeClientSession:
        def __init__(self, read, write, **kwargs):
            events.append(("session", read, write, kwargs))

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            events.append(("session_closed",))

        async def initialize(self):
            events.append(("initialize",))

        async def list_tools(self, cursor=None):
            events.append(("list_tools", cursor))
            return SimpleNamespace(
                tools=[SimpleNamespace(name="search")], nextCursor=None
            )

        async def call_tool(self, name, arguments):
            events.append(("call_tool", name, arguments))
            return SimpleNamespace(content=[], isError=False)

    @asynccontextmanager
    async def fake_http_client(url, **kwargs):
        events.append(("connect", url, kwargs))
        yield ("read", "write", "session-id")
        events.append(("transport_closed",))

    mcp_module = ModuleType("mcp")
    mcp_module.ClientSession = FakeClientSession
    mcp_module.StdioServerParameters = object
    client_module = ModuleType("mcp.client")
    stdio_module = ModuleType("mcp.client.stdio")
    stdio_module.stdio_client = None
    http_module = ModuleType("mcp.client.streamable_http")
    http_module.streamable_http_client = fake_http_client
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.client", client_module)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio_module)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", http_module)

    runner = PersistentMcpSession(
        transport="streamable_http",
        timeout_seconds=2,
        server_url="https://mcp.example.test/mcp",
        http_headers=None,
        command=None,
        command_args=[],
        env=None,
    )
    runner.start()
    assert [tool.name for tool in runner.list_tools()] == ["search"]
    runner.call_tool("search", {"query": "one"})
    runner.call_tool("search", {"query": "two"})
    runner.close()

    assert [event[0] for event in events].count("initialize") == 1
    assert [event[0] for event in events].count("connect") == 1
    assert [event[0] for event in events].count("call_tool") == 2
    assert events[-2:] == [("session_closed",), ("transport_closed",)]
