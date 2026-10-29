"""Enterprise RAG Blueprint benchmarked as an MCP host.

A fake transport stands in for the Blueprint ``/call_llm`` endpoint, so the
tests need neither the Blueprint codebase nor Chroma, an LLM or MCP servers.
"""

import base64
import hashlib
import hmac
import json
import time
from types import SimpleNamespace

import pytest

from examples.enterprise_rag_plugin import mcp_adapter
from examples.enterprise_rag_plugin.mcp_adapter import (
    EnterpriseRagMcpAdapter,
    TransportResponse,
    sign_identity,
)

SECRET = "s" * 40
ALIAS = "demo__get_retention_policy__0123456789ab"
ADD_ALIAS = "demo__add__ba9876543210"


def _config(**overrides):
    values = dict(
        enterprise_rag_blueprint_path=None,
        enterprise_rag_mcp_transport="inprocess",
        enterprise_rag_mcp_endpoint_url=None,
        enterprise_rag_mcp_enabled=True,
        enterprise_rag_mcp_roles="General",
        enterprise_rag_mcp_subject="bench-user",
        enterprise_rag_mcp_prompt_key="rag_source",
        enterprise_rag_mcp_used_model=None,
        enterprise_rag_mcp_timeout_seconds=30.0,
        enterprise_rag_mcp_index_corpus=False,
        enterprise_rag_mcp_apply_settings=False,
        enterprise_rag_mcp_query_refinement=True,
        enterprise_rag_mcp_guardrails=True,
        enterprise_rag_mcp_llm_api_key_env=None,
        embedding_model="nomic-embed-text:latest",
        # Real configs keep the provider prefix in llm_provider only.
        llm_model="qwen3:8b",
        llm_provider="ollama",
        retrieval_top_k=7,
        generation_temperature=0.0,
    )
    urls = {
        "llm_base_url": overrides.pop("llm_base_url", "http://llm-host:11434"),
        "llm_api_key": overrides.pop("llm_api_key", None),
        "embedding_base_url": overrides.pop("embedding_base_url", "http://emb-host:11434"),
        "embedding_api_key": overrides.pop("embedding_api_key", None),
    }
    values.update(overrides)
    for name, value in urls.items():
        values[name] = lambda value=value: value
    return SimpleNamespace(**values)


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, payload, headers):
        self.requests.append((payload, headers))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _blueprint_body(events, sources=(), documents=("handbook.pdf->Doc text",)):
    relevant = list(documents) + [
        f"MCP {s['server']}/{s['name']}->" + json.dumps({"content": ["30 days"]})
        for s in sources
    ]
    return {
        "result": "Retention is 30 days.",
        "relevant_documents": relevant,
        "prompt_key": "rag_source",
        "mcp": {"request_id": "r1", "events": list(events), "sources": list(sources)},
    }


def _adapter(*responses, **config_overrides):
    monkey_env = config_overrides.pop("secret", SECRET)
    transport = FakeTransport(*responses)
    adapter = EnterpriseRagMcpAdapter(
        _config(**config_overrides), transport=transport, identity_secret=monkey_env
    )
    adapter.prepare(adapter.config, data=[])
    return adapter, transport


def _verify(header):
    assert header.startswith("Bearer ")
    payload, signature = header[7:].split(".")
    expected = hmac.new(SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    assert hmac.compare_digest(signature, expected)
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


def test_signed_identity_matches_blueprint_contract():
    header = sign_identity(SECRET, "bench-user", ["General", "Legal & Compliance"], now=1000)

    claims = _verify(header)

    assert claims == {
        "sub": "bench-user",
        "roles": ["General", "Legal & Compliance"],
        "aud": "rag-mcp",
        "iat": 1000,
        "exp": 1060,
    }


def test_short_secret_is_rejected_before_the_run():
    adapter = EnterpriseRagMcpAdapter(
        _config(), transport=FakeTransport(), identity_secret="short"
    )
    with pytest.raises(ValueError, match="MCP_IDENTITY_SECRET"):
        adapter.prepare(adapter.config, data=[])


def test_mcp_request_carries_fresh_identity_and_mcp_flag():
    adapter, transport = _adapter(
        TransportResponse(200, _blueprint_body([])),
        enterprise_rag_mcp_roles="General, Legal & Compliance",
        enterprise_rag_mcp_used_model="qwen3",
    )

    adapter.answer({"question": "How long are documents kept?", "metadata": {}}, adapter.config)

    payload, headers = transport.requests[0]
    assert payload == {
        "question": "How long are documents kept?",
        "use_mcp": True,
        "prompt_key": "rag_source",
        "show_sources": True,
        "user_roles": ["General", "Legal & Compliance"],
        "chat_history": [],
        "used_model": "qwen3",
    }
    claims = _verify(headers["Authorization"])
    assert claims["sub"] == "bench-user"
    assert claims["roles"] == ["General", "Legal & Compliance"]
    assert abs(claims["iat"] - time.time()) < 5


def test_answer_and_evidence_are_normalized():
    sources = [
        {"server": "demo", "kind": "tool", "name": "get_retention_policy", "capability": ALIAS}
    ]
    adapter, _ = _adapter(TransportResponse(200, _blueprint_body([], sources=sources)))

    out = adapter.answer({"question": "q", "metadata": {}}, adapter.config)

    assert out.answer == "Retention is 30 days."
    assert out.answer_valid is True
    # The Blueprint shows the model each document's name, so the evaluator
    # sees it too and can check answers that cite their source.
    assert out.contexts[0] == "Document: handbook.pdf\nDoc text"
    assert "30 days" in out.contexts[1]
    assert not out.contexts[1].startswith("Document:")
    assert out.metadata[0]["evidence_type"] == "document"
    assert out.metadata[0]["doc_id"] == "handbook"
    assert out.metadata[1]["evidence_type"] == "mcp"
    assert out.metadata[1]["mcp_server"] == "demo"
    assert out.metadata[1]["mcp_capability"] == "get_retention_policy"
    assert "doc_id" not in out.metadata[1]
    assert out.total_seconds >= 0


def test_blueprint_events_feed_mcp_host_metrics():
    events = [
        {"server": "demo", "operation": "discover", "status": "ok"},
        {"server": "legal", "operation": "connect", "status": "denied"},
        {"server": "demo", "operation": "tool", "status": "ok",
         "capability": ALIAS, "duration_ms": 250},
        {"server": "demo", "operation": "tool", "status": "error",
         "capability": ADD_ALIAS, "duration_ms": 50},
        {"server": "demo", "operation": "tool", "status": "invalid_arguments",
         "capability": ADD_ALIAS},
        {"server": "demo", "operation": "resource", "status": "unavailable",
         "capability": "demo__blueprint___policy_re__aaaaaaaaaaaa"},
        {"server": "host", "operation": "call", "status": "budget_exhausted"},
        {"server": "host", "operation": "loop", "status": "budget_exhausted"},
    ]
    sources = [
        {"server": "demo", "kind": "tool", "name": "get_retention_policy", "capability": ALIAS}
    ]
    adapter, _ = _adapter(TransportResponse(200, _blueprint_body(events, sources=sources)))

    out = adapter.answer(
        {"question": "q", "metadata": {"expected_tools": ["get_retention_policy"]}},
        adapter.config,
    )
    metrics = adapter.aggregate_metrics([out.diagnostics])

    assert metrics["call_count"] == 5
    assert metrics["call_status_counts"] == {
        "ok": 1,
        "tool_error": 1,
        "invalid_arguments": 1,
        "unavailable": 1,
        "budget_exhausted": 1,
    }
    assert metrics["call_latency_max_seconds"] == pytest.approx(0.25)
    assert metrics["connect_denied_count"] == 1
    assert metrics["discovery_success_rate"] == 1.0
    assert metrics["budget_exhausted_sample_count"] == 1
    assert metrics["per_tool"]["demo/get_retention_policy"]["call_success_rate"] == 1.0
    # Aliases without a successful source fall back to the alias label.
    assert metrics["per_tool"]["demo/add"]["call_count"] == 2
    assert metrics["tool_selection_recall"] == 1.0
    assert adapter.diagnostic_stage_timings(out.diagnostics) == {
        "mcp_host_calls": pytest.approx(0.3)
    }


def test_expected_tools_accepts_comma_separated_metadata():
    adapter, _ = _adapter(TransportResponse(200, _blueprint_body([])))

    out = adapter.answer(
        {"question": "q", "metadata": {"expected_tools": "add, demo/get_retention_policy"}},
        adapter.config,
    )

    assert out.diagnostics["mcp_host"]["expected_tools"] == [
        "add",
        "demo/get_retention_policy",
    ]


def test_unknown_blueprint_status_is_counted_not_fatal():
    events = [{"server": "demo", "operation": "tool", "status": "exploded", "capability": ALIAS}]
    adapter, _ = _adapter(TransportResponse(200, _blueprint_body(events)))

    out = adapter.answer({"question": "q", "metadata": {}}, adapter.config)

    assert out.diagnostics["mcp_host"]["events"] == []
    assert out.diagnostics["untranslated_event_count"] == 1


@pytest.mark.parametrize(
    "status, failure_kind",
    [
        (400, "guardrail"),
        (401, "identity"),
        (422, "input"),
        (502, "model_or_tool_protocol"),
        (503, "configuration"),
        (504, "timeout"),
        (500, "http_error"),
    ],
)
def test_http_errors_become_failed_samples(status, failure_kind):
    adapter, _ = _adapter(TransportResponse(status, {"detail": "nope"}))

    out = adapter.answer(
        {"question": "q", "metadata": {"expected_tools": ["add"]}}, adapter.config
    )

    assert out.answer == ""
    assert out.answer_valid is False
    assert out.diagnostics["failure_kind"] == failure_kind
    assert out.diagnostics["error"] == f"HTTP {status}: nope"
    metrics = adapter.aggregate_metrics([out.diagnostics])
    assert metrics["failed_sample_count"] == 1
    assert metrics["tool_selection_sample_count"] == 0


def test_transport_exception_becomes_failed_sample():
    adapter, _ = _adapter(ConnectionError("refused"))

    out = adapter.answer({"question": "q", "metadata": {}}, adapter.config)

    assert out.answer_valid is False
    assert out.diagnostics["failure_kind"] == "transport"
    assert "refused" in out.diagnostics["error"]


def test_success_without_mcp_trace_is_flagged():
    body = _blueprint_body([])
    del body["mcp"]
    adapter, _ = _adapter(TransportResponse(200, body))

    out = adapter.answer({"question": "q", "metadata": {}}, adapter.config)

    assert out.diagnostics["failure_kind"] == "missing_mcp_trace"
    assert out.answer == "Retention is 30 days."


def test_mcp_disabled_uses_legacy_path_without_identity():
    body = {"result": "Plain answer", "relevant_documents": ["a.txt->text"]}
    adapter, transport = _adapter(
        TransportResponse(200, body), enterprise_rag_mcp_enabled=False, secret=""
    )

    out = adapter.answer(
        {"question": "q", "ground_truth": "Other answer", "metadata": {}}, adapter.config
    )

    payload, headers = transport.requests[0]
    assert payload["use_mcp"] is False
    assert "Authorization" not in headers
    assert out.answer == "Plain answer"
    assert "mcp_host" not in out.diagnostics
    metrics = adapter.aggregate_metrics([out.diagnostics])
    assert "call_count" not in metrics
    assert metrics["answer_match_rate"] == 0.0


def test_answer_match_and_category_feed_metrics():
    adapter, _ = _adapter(
        TransportResponse(200, _blueprint_body([])),
        TransportResponse(200, _blueprint_body([])),
        TransportResponse(502, {"detail": "boom"}),
    )
    samples = [
        # Explicit pattern groups: every group must match.
        {"question": "q", "ground_truth": "30 days",
         "metadata": {"category": "mcp_single", "answer_patterns": [["30 days"], ["retention"]]}},
        # Without patterns the ground truth itself is the pattern.
        {"question": "q", "ground_truth": "45 days", "metadata": {"category": "doc_single"}},
        {"question": "q", "ground_truth": "30 days", "metadata": {"category": "doc_single"}},
    ]

    outs = [adapter.answer(s, adapter.config) for s in samples]
    metrics = adapter.aggregate_metrics([o.diagnostics for o in outs])

    assert [o.diagnostics["answer_match"] for o in outs] == [True, False, False]
    assert [o.diagnostics["category"] for o in outs] == ["mcp_single", "doc_single", "doc_single"]
    assert metrics["answer_match_rate"] == pytest.approx(1 / 3)
    assert metrics["per_category"]["mcp_single"]["answer_match_rate"] == 1.0
    assert metrics["per_category"]["doc_single"]["sample_count"] == 2
    assert metrics["call_count"] == 0


def test_prepare_indexes_corpus_with_blueprint_embedding(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mcp_adapter,
        "replace_blueprint_index",
        lambda bundle, config, data, corpus=None: calls.append((bundle, data, corpus)),
    )
    adapter = EnterpriseRagMcpAdapter(
        _config(enterprise_rag_mcp_index_corpus=True),
        transport=FakeTransport(),
        identity_secret=SECRET,
    )
    adapter.set_components(SimpleNamespace(embedder="EMB", chunker=None))

    adapter.prepare(adapter.config, data=[{"q": 1}], corpus=[{"context": "c"}])

    assert calls == [(adapter.bundle, [{"q": 1}], [{"context": "c"}])]


def test_prepare_rejects_embedding_the_blueprint_cannot_query(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    adapter = EnterpriseRagMcpAdapter(
        _config(enterprise_rag_mcp_index_corpus=True, embedding_model="bge-m3"),
        transport=FakeTransport(),
        identity_secret=SECRET,
    )
    with pytest.raises(ValueError, match="nomic-embed-text"):
        adapter.prepare(adapter.config, data=[])


def test_ollama_llm_settings_map_to_blueprint_environment():
    env = mcp_adapter.blueprint_settings_env(_config())

    assert env == {
        "LLM_SOURCE_ANSWER": "openai",
        "MODEL_NAME": "qwen3:8b",
        # Ollama serves an OpenAI-compatible API under /v1.
        "OPENAI_BASE_URL": "http://llm-host:11434/v1",
        "OPENAI_API_KEY": "ollama",
        "LLM_TEMPERATURE": "0",
        "EMBEDDING_MODEL": "nomic-embed-text:latest",
        "OLLAMA_BASE_URL": "http://emb-host:11434",
        "OLLAMA_API_KEY": "",
        "RETRIEVER_TOP_K": "7",
        "DISABLE_QUERY_REFINEMENT": "false",
        "DISABLE_GUARDRAILS": "false",
    }


def test_ollama_behind_bearer_proxy_gets_keys_for_llm_and_embeddings():
    env = mcp_adapter.blueprint_settings_env(
        _config(
            llm_base_url="http://dgx-spark/ollama",
            llm_api_key="llm-token",
            embedding_base_url="http://dgx-spark/ollama",
            embedding_api_key="emb-token",
        )
    )

    assert env["OPENAI_BASE_URL"] == "http://dgx-spark/ollama/v1"
    assert env["OPENAI_API_KEY"] == "llm-token"
    assert env["OLLAMA_BASE_URL"] == "http://dgx-spark/ollama"
    assert env["OLLAMA_API_KEY"] == "emb-token"


def test_openai_compatible_llm_and_toggles_map_to_blueprint_environment():
    env = mcp_adapter.blueprint_settings_env(
        _config(
            llm_model="claude-opus-5-5",
            llm_provider="openai",
            llm_base_url="https://api.anthropic.com/v1/",
            llm_api_key="sk-test",
            generation_temperature=0.3,
            enterprise_rag_mcp_query_refinement=False,
            enterprise_rag_mcp_guardrails=False,
        )
    )

    assert env["MODEL_NAME"] == "claude-opus-5-5"
    assert env["OPENAI_BASE_URL"] == "https://api.anthropic.com/v1/"
    assert env["OPENAI_API_KEY"] == "sk-test"
    assert env["LLM_TEMPERATURE"] == "0.3"
    assert env["DISABLE_QUERY_REFINEMENT"] == "true"
    assert env["DISABLE_GUARDRAILS"] == "true"


def test_llm_key_can_come_from_a_named_environment_variable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    config = _config(
        llm_model="claude-opus-5-5",
        llm_provider="openai",
        llm_api_key="shared-key",
        enterprise_rag_mcp_llm_api_key_env="ANTHROPIC_API_KEY",
    )

    assert mcp_adapter.blueprint_settings_env(config)["OPENAI_API_KEY"] == "sk-ant-test"


def test_missing_named_llm_key_fails_before_the_run(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    adapter = EnterpriseRagMcpAdapter(
        _config(
            enterprise_rag_mcp_apply_settings=True,
            enterprise_rag_mcp_llm_api_key_env="ANTHROPIC_API_KEY",
        ),
        transport=FakeTransport(),
        identity_secret=SECRET,
    )

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        adapter.prepare(adapter.config, data=[])


def test_applied_settings_hold_only_during_the_request(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "blueprint-default")
    monkeypatch.delenv("RETRIEVER_TOP_K", raising=False)
    seen = {}

    class EnvRecordingTransport(FakeTransport):
        def post(self, payload, headers):
            seen["MODEL_NAME"] = mcp_adapter.os.environ.get("MODEL_NAME")
            seen["RETRIEVER_TOP_K"] = mcp_adapter.os.environ.get("RETRIEVER_TOP_K")
            return super().post(payload, headers)

    transport = EnvRecordingTransport(TransportResponse(200, _blueprint_body([])))
    adapter = EnterpriseRagMcpAdapter(
        _config(enterprise_rag_mcp_apply_settings=True),
        transport=transport,
        identity_secret=SECRET,
    )
    adapter.prepare(adapter.config, data=[])

    adapter.answer({"question": "q", "metadata": {}}, adapter.config)

    assert seen == {"MODEL_NAME": "qwen3:8b", "RETRIEVER_TOP_K": "7"}
    assert transport.requests[0][0]["used_model"] == "openai"
    assert mcp_adapter.os.environ["MODEL_NAME"] == "blueprint-default"
    assert "RETRIEVER_TOP_K" not in mcp_adapter.os.environ


def test_without_applied_settings_the_blueprint_environment_is_untouched(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "blueprint-default")
    seen = []

    class EnvRecordingTransport(FakeTransport):
        def post(self, payload, headers):
            seen.append(mcp_adapter.os.environ.get("MODEL_NAME"))
            return super().post(payload, headers)

    transport = EnvRecordingTransport(TransportResponse(200, _blueprint_body([])))
    adapter = EnterpriseRagMcpAdapter(_config(), transport=transport, identity_secret=SECRET)
    adapter.prepare(adapter.config, data=[])

    adapter.answer({"question": "q", "metadata": {}}, adapter.config)

    assert seen == ["blueprint-default"]
    assert "used_model" not in transport.requests[0][0]


def test_applied_settings_index_with_any_ollama_embedding(monkeypatch):
    monkeypatch.setattr(mcp_adapter, "replace_blueprint_index", lambda *a, **k: None)
    adapter = EnterpriseRagMcpAdapter(
        _config(
            enterprise_rag_mcp_index_corpus=True,
            enterprise_rag_mcp_apply_settings=True,
            embedding_model="bge-m3",
        ),
        transport=FakeTransport(),
        identity_secret=SECRET,
    )

    adapter.prepare(adapter.config, data=[])


def test_without_applied_settings_index_must_match_blueprint_embedding(monkeypatch):
    monkeypatch.setattr(mcp_adapter, "replace_blueprint_index", lambda *a, **k: None)
    monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3")
    adapter = EnterpriseRagMcpAdapter(
        _config(enterprise_rag_mcp_index_corpus=True, embedding_model="bge-m3:latest"),
        transport=FakeTransport(),
        identity_secret=SECRET,
    )

    adapter.prepare(adapter.config, data=[])


def test_inprocess_transport_rejects_blueprint_without_per_request_settings(
    monkeypatch, tmp_path
):
    import sys

    (tmp_path / "chain").mkdir()
    (tmp_path / "chain" / "handle_llms.py").write_text("MODEL_NAME = 'fixed'\n")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "handle_llms", raising=False)
    config = _config(
        enterprise_rag_blueprint_path=str(tmp_path), enterprise_rag_mcp_apply_settings=True
    )

    with pytest.raises(ValueError, match="update the Blueprint"):
        mcp_adapter.InProcessTransport(config, timeout_seconds=5)


def test_http_transport_posts_json(monkeypatch):
    captured = {}

    class _Response:
        status = 200

        def read(self):
            return b'{"result": "ok"}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(mcp_adapter.urllib.request, "urlopen", fake_urlopen)
    transport = mcp_adapter.HttpTransport("http://chain:8080/call_llm", timeout_seconds=9)

    response = transport.post({"question": "q"}, {"Authorization": "Bearer x"})

    assert response == TransportResponse(200, {"result": "ok"})
    assert captured["url"] == "http://chain:8080/call_llm"
    assert captured["body"] == {"question": "q"}
    assert captured["headers"]["Authorization"] == "Bearer x"
    assert captured["timeout"] == 9


def test_http_transport_returns_error_status_with_detail(monkeypatch):
    import io
    import urllib.error

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 401, "Unauthorized", {}, io.BytesIO(b'{"detail": "bad"}')
        )

    monkeypatch.setattr(mcp_adapter.urllib.request, "urlopen", fake_urlopen)
    transport = mcp_adapter.HttpTransport("http://chain:8080/call_llm", timeout_seconds=9)

    assert transport.post({}, {}) == TransportResponse(401, {"detail": "bad"})


def test_adapter_is_registered():
    from benchmark.adapters import RAG_ADAPTER_REGISTRY

    assert "enterprise_rag_mcp" in RAG_ADAPTER_REGISTRY
