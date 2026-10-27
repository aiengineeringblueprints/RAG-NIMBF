"""The example corpus MCP server, driven over stdio like a real benchmark run."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from benchmark.adapters.mcp import PersistentMcpSession

SERVER = Path(__file__).resolve().parent.parent / "examples" / "corpus_mcp_server.py"

FILLER = "Routine maintenance notes describe inspections and handover records. " * 12
LONG_DOC = (
    "# Station Alpha\n\n"
    f"{FILLER}\n\n"
    f"{FILLER}\n\n"
    "The backup generator at Station Alpha is a hydrogen fuel cell.\n"
)


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "alpha.md").write_text(LONG_DOC, encoding="utf-8")
    (tmp_path / "beta.md").write_text(
        "# Station Beta\n\nStation Beta has no backup generator.\n", encoding="utf-8"
    )
    return tmp_path


def _search(corpus: Path, *server_args: str, **arguments):
    session = PersistentMcpSession(
        transport="stdio",
        timeout_seconds=30,
        server_url=None,
        http_headers=None,
        command=sys.executable,
        command_args=[str(SERVER), "--corpus", str(corpus), *server_args],
        env=None,
    )
    session.start()
    try:
        result = session.call_tool("search", arguments)
    finally:
        session.close()
    assert not result.isError, result
    return result.structuredContent["contexts"]


def test_search_returns_whole_documents_by_default(corpus):
    contexts = _search(corpus, query="hydrogen fuel cell generator", top_k=1)

    assert contexts[0]["source"] == "alpha.md"
    assert contexts[0]["text"] == LONG_DOC


def test_search_returns_chunks_matching_the_internal_chunk_budget(corpus):
    contexts = _search(
        corpus,
        "--chunk-size",
        "300",
        "--chunk-overlap",
        "50",
        query="hydrogen fuel cell generator",
        top_k=3,
    )

    assert all(len(item["text"]) <= 300 for item in contexts)
    best = contexts[0]
    assert best["source"] == "alpha.md"
    assert "hydrogen fuel cell" in best["text"]
    assert best["text"] != LONG_DOC
