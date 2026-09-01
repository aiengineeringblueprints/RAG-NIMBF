from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from benchmark.adapters.base import (
    AdapterGenerationResult,
    PreparedTarget,
    RetrievalResult,
)
from benchmark.adapters.ragflow import (
    DatasetSpec,
    GenerationSpec,
    RagflowAdapter,
    RagflowApiError,
    RagflowClient,
    RagflowIndexingError,
    RagflowError,
    RetrievalSpec,
    TransportResponse,
    UploadDocument,
    deterministic_resource_name,
)


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def request(self, method, url, headers, body, timeout_seconds):
        self.requests.append((method, url, dict(headers), body, timeout_seconds))
        response = self.responses.pop(0)
        if callable(response):
            return response(method, url, headers, body, timeout_seconds)
        return response


def response(payload, status=200):
    return TransportResponse(status, json.dumps(payload).encode())


def envelope(data=None, code=0, message=""):
    return response({"code": code, "data": data, "message": message})


def test_client_reads_key_from_env_and_redacts_it(monkeypatch):
    monkeypatch.setenv("PRIVATE_RAG_KEY", "very-secret-value")
    client = RagflowClient.from_env(
        "http://rag.local", "PRIVATE_RAG_KEY", transport=FakeTransport([])
    )

    assert "very-secret-value" not in repr(client)
    assert "<redacted>" in repr(client)


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://rag.local",
        "http://user:password@rag.local",
        "http://rag.local?token=secret",
        "http://rag.local#fragment",
    ],
)
def test_client_rejects_unsafe_base_urls(base_url):
    with pytest.raises(ValueError):
        RagflowClient(base_url, "secret", transport=FakeTransport([]))


def test_deterministic_name_is_stable_and_setting_sensitive():
    first = deterministic_resource_name("Benchmark Run", {"chunk": 128})
    second = deterministic_resource_name("Benchmark Run", {"chunk": 128})
    changed = deterministic_resource_name("Benchmark Run", {"chunk": 256})

    assert first == second
    assert first != changed
    assert first.startswith("Benchmark-Run-")
    assert len(first) <= 128


def test_client_retrieval_normalizes_chunks_and_retries_transient_status():
    transport = FakeTransport(
        [
            response({"message": "busy"}, status=503),
            envelope(
                {
                    "chunks": [
                        {
                            "id": "chunk-1",
                            "content": "Evidence",
                            "document_id": "doc-1",
                            "kb_id": "kb-1",
                            "similarity": 0.9,
                            "vector_similarity": 0.8,
                            "term_similarity": 1.0,
                        }
                    ],
                    "total": 1,
                }
            ),
        ]
    )
    client = RagflowClient(
        "http://rag.local",
        "secret",
        transport=transport,
        max_retries=1,
        retry_backoff_seconds=0,
        sleep=lambda _: None,
    )

    result = client.retrieve("question", ["kb-1"], RetrievalSpec(result_k=5))

    assert len(transport.requests) == 2
    sent = json.loads(transport.requests[-1][3])
    assert sent["page_size"] == 5
    assert sent["top_k"] == 1024
    assert result.chunks[0].chunk_id == "chunk-1"
    assert result.chunks[0].score == 0.9


def test_generation_requests_references_usage_and_does_not_retry():
    transport = FakeTransport([response({"message": "busy"}, status=503)])
    client = RagflowClient(
        "http://rag.local",
        "secret",
        transport=transport,
        max_retries=3,
        retry_backoff_seconds=0,
        sleep=lambda _: None,
    )

    with pytest.raises(RagflowApiError):
        client.generate("chat-1", "question", GenerationSpec())

    assert len(transport.requests) == 1
    payload = json.loads(transport.requests[0][3])
    assert payload["stream"] is False
    assert payload["reference"] is True


def test_generation_normalizes_references_and_usage():
    transport = FakeTransport(
        [
            response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "Answer",
                                "reference": [
                                    {
                                        "id": "chunk-1",
                                        "content": "Evidence",
                                        "document_id": "doc-1",
                                        "dataset_id": "kb-1",
                                        "similarity": 0.75,
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 3,
                        "total_tokens": 10,
                    },
                }
            )
        ]
    )
    client = RagflowClient("http://rag.local", "secret", transport=transport)

    result = client.generate("chat-1", "question", GenerationSpec())

    assert result.answer == "Answer"
    assert result.chunks[0].text == "Evidence"
    assert result.input_tokens == 7
    assert result.output_tokens == 3
    assert result.total_tokens == 10


def test_upload_uses_multipart_and_sanitizes_filename():
    transport = FakeTransport([envelope([{"id": "doc-1", "name": "secret.txt"}])])
    client = RagflowClient("http://rag.local", "secret", transport=transport)

    client.upload_documents(
        "kb-1", [UploadDocument("../../secret.txt", b"contents", "text/plain")]
    )

    _, _, headers, body, _ = transport.requests[0]
    assert headers["Content-Type"].startswith("multipart/form-data; boundary=")
    assert b'filename="secret.txt"' in body
    assert b"../" not in body
    assert b"contents" in body


def test_wait_for_documents_fails_closed_on_indexing_failure():
    transport = FakeTransport(
        [
            envelope(
                {
                    "docs": [
                        {"id": "doc-1", "run": "FAIL", "progress_msg": "parser failed"}
                    ]
                }
            )
        ]
    )
    client = RagflowClient("http://rag.local", "secret", transport=transport)

    with pytest.raises(RagflowIndexingError, match="parser failed"):
        client.wait_for_documents("kb-1", ["doc-1"])


def test_managed_adapter_preserves_stable_source_provenance():
    transport = FakeTransport(
        [
            response({"status": "ok"}),
            envelope([]),
            envelope({"id": "kb-1", "name": "benchmark"}),
            envelope({"docs": []}),
            envelope([{"id": "doc-1", "name": "00000-NF-001.txt", "run": "UNSTART"}]),
            envelope(None),
            envelope(
                {
                    "docs": [
                        {
                            "id": "doc-1",
                            "name": "00000-NF-001.txt",
                            "run": "DONE",
                            "progress": 1.0,
                        }
                    ]
                }
            ),
            envelope([]),
            envelope({"id": "chat-1", "name": "chat"}),
            envelope(
                {
                    "chunks": [
                        {
                            "id": "chunk-1",
                            "content": "Northstar evidence",
                            "document_id": "doc-1",
                            "similarity": 0.9,
                        }
                    ],
                    "total": 1,
                }
            ),
            response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "Answer",
                                "reference": [
                                    {
                                        "id": "chunk-1",
                                        "content": "Northstar evidence",
                                        "document_name": "00000-NF-001.txt",
                                        "similarity": 0.9,
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 2,
                        "total_tokens": 6,
                    },
                }
            ),
        ]
    )
    client = RagflowClient("http://rag.local", "secret", transport=transport)
    adapter = RagflowAdapter(
        client,
        DatasetSpec(),
        RetrievalSpec(),
        GenerationSpec(),
        poll_interval_seconds=0,
    )
    corpus = [{"context": "Northstar evidence", "metadata": {"source_id": "NF-001"}}]

    target = adapter.prepare(object(), [], corpus)
    retrieval = adapter.retrieve(target, {"question": "What?"}, object())
    generation = adapter.generate(target, {"question": "What?"}, object())

    assert isinstance(target, PreparedTarget)
    assert isinstance(retrieval, RetrievalResult)
    assert retrieval.chunks[0].source_id == "NF-001"
    assert retrieval.chunks[0].as_metadata()["doc_id"] == "NF-001"
    assert isinstance(generation, AdapterGenerationResult)
    assert generation.retrieval.chunks[0].source_id == "NF-001"
    assert generation.total_tokens == 6


def test_corpus_content_changes_dataset_resource_identity():
    # Exercise the public prepare path's effective identity without making HTTP calls.
    import benchmark.adapters.ragflow as module

    first = module._corpus_digest([{"context": "one", "metadata": {"source_id": "A"}}])
    second = module._corpus_digest([{"context": "two", "metadata": {"source_id": "A"}}])
    assert (
        DatasetSpec(corpus_digest=first).resource_name
        != DatasetSpec(corpus_digest=second).resource_name
    )


def test_retrieval_stage_prepares_without_chat_or_llm_dependency():
    transport = FakeTransport(
        [
            response({"status": "ok"}),
            envelope([]),
            envelope({"id": "kb-1"}),
            envelope({"docs": []}),
            envelope([{"id": "doc-1", "name": "00000-NF-001.txt", "run": "DONE"}]),
        ]
    )
    adapter = RagflowAdapter(
        RagflowClient("http://rag.local", "secret", transport=transport),
        DatasetSpec(),
        RetrievalSpec(),
        GenerationSpec(),
    )

    @dataclass
    class Config:
        benchmark_stage: str = "retrieve"

    target = adapter.prepare(
        Config(),
        [],
        [{"context": "Evidence", "metadata": {"source_id": "NF-001"}}],
    )

    assert target.target_id == "kb-1"
    assert not any("/chats" in request[1] for request in transport.requests)
    with pytest.raises(RagflowError, match="without a chat"):
        adapter.generate(target, {"question": "What?"}, Config())


def test_registry_exposes_both_provider_names():
    from benchmark.adapters import RAG_ADAPTER_REGISTRY

    assert (
        RAG_ADAPTER_REGISTRY["ragflow"].__func__ is RagflowAdapter.from_config.__func__
    )
    assert (
        RAG_ADAPTER_REGISTRY["optimaiserag"].__func__
        is RagflowAdapter.from_config.__func__
    )
