from __future__ import annotations

import json
from dataclasses import dataclass

from benchmark.adapters import get_rag_adapter
from benchmark.adapters.http import HttpRagAdapter


@dataclass
class DummyConfig:
    name: str = "cfg"
    retrieval_top_k: int = 3
    prompt_template: str = "concise"
    dataset_name: str = "dummy"
    rag_system_adapter: str = "http"
    rag_http_endpoint_url: str = "http://rag.local/query"
    rag_http_timeout_seconds: float = 5.0
    rag_http_answer_field: str = "result.answer"
    rag_http_contexts_field: str = "result.contexts"
    rag_http_metadata_field: str = "result.metadata"
    rag_http_timings_field: str = "timings"
    rag_http_headers: str | None = None
    rag_http_auth_header: str | None = None
    rag_http_auth_value: str | None = None


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_http_adapter_normalizes_nested_response(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse({
            "result": {
                "answer": "42",
                "contexts": [{"text": "supporting context"}],
                "metadata": [{"doc_id": "doc-1"}],
            },
            "timings": {"ttft_seconds": 0.1, "total_seconds": 0.5},
            "token_count": 10,
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    adapter = HttpRagAdapter.from_config(DummyConfig())
    output = adapter.answer({"question": "What?", "ground_truth": "42"}, DummyConfig())

    assert captured["timeout"] == 5.0
    assert captured["body"]["question"] == "What?"
    assert output.answer == "42"
    assert output.contexts == ["supporting context"]
    assert output.metadata == [{"doc_id": "doc-1"}]
    assert output.ttft_seconds == 0.1
    assert output.total_seconds == 0.5
    assert output.token_count == 10
    assert output.tokens_per_second == 20.0


def test_get_rag_adapter_returns_none_for_internal():
    cfg = DummyConfig(rag_system_adapter="internal")
    assert get_rag_adapter(cfg) is None
