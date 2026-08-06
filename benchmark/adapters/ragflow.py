"""Managed RAGFlow/optimAIseRAG integration using its public HTTP API.

The client intentionally has no dependency on the vendored RAGFlow SDK.  This
keeps the benchmark process isolated from the server checkout and makes the
HTTP contract straightforward to mock in unit tests.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from benchmark.adapters.base import (
    AdapterCapabilities,
    AdapterGenerationResult,
    PreparedTarget,
    RetrievedChunk as CanonicalChunk,
    RetrievalResult as CanonicalRetrievalResult,
    RagSystemOutput,
)
from benchmark.prompt_templates import get_template


class RagflowError(RuntimeError):
    """Base error for a failed RAGFlow operation."""


class RagflowApiError(RagflowError):
    """The server returned an HTTP or RAGFlow envelope error."""


class RagflowIndexingError(RagflowError):
    """One or more documents failed or were cancelled during indexing."""


class RagflowTimeoutError(RagflowError):
    """A request or asynchronous indexing operation exceeded its deadline."""


@dataclass(frozen=True)
class TransportResponse:
    status: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)


class HttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
    ) -> TransportResponse: ...


def _url_origin(url: str) -> tuple[str, str, int | None]:
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme.lower()
    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80 if scheme == "http" else None
    return scheme, (parsed.hostname or "").lower(), port


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Fail closed before an authenticated request crosses an origin."""

    def __init__(self, original_url: str) -> None:
        self._origin = _url_origin(original_url)
        super().__init__()

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if _url_origin(newurl) != self._origin:
            raise urllib.error.HTTPError(
                newurl,
                403,
                "RAGFlow cross-origin redirect rejected",
                headers,
                fp,
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class UrllibTransport:
    """Small standard-library transport; useful when the SDK is unavailable."""

    def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
    ) -> TransportResponse:
        request = urllib.request.Request(
            url, data=body, headers=dict(headers), method=method
        )
        opener = urllib.request.build_opener(_SameOriginRedirectHandler(url))
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                return TransportResponse(
                    status=int(response.status),
                    body=response.read(),
                    headers=dict(response.headers.items()),
                )
        except urllib.error.HTTPError as exc:
            return TransportResponse(
                status=exc.code,
                body=exc.read(),
                headers=dict(exc.headers.items()) if exc.headers else {},
            )
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RagflowTimeoutError(
                "RAGFlow HTTP request failed or timed out"
            ) from exc


@dataclass(frozen=True)
class UploadDocument:
    """A document upload kept in memory to avoid temporary-file side effects."""

    name: str
    content: bytes
    content_type: str | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "UploadDocument":
        resolved = Path(path).expanduser().resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"RAGFlow upload target is not a file: {resolved}")
        return cls(
            name=resolved.name,
            content=resolved.read_bytes(),
            content_type=mimetypes.guess_type(resolved.name)[0],
        )


@dataclass(frozen=True)
class DatasetSpec:
    embedding_model: str | None = None
    chunk_method: str = "naive"
    parser_config: Mapping[str, Any] = field(default_factory=dict)
    name_prefix: str = "benchmark"
    description: str = "Managed by Benchmarking Framework"
    corpus_digest: str | None = None

    @property
    def resource_name(self) -> str:
        return deterministic_resource_name(
            self.name_prefix,
            {
                "embedding_model": self.embedding_model,
                "chunk_method": self.chunk_method,
                "parser_config": dict(self.parser_config),
                "corpus_digest": self.corpus_digest,
            },
        )


@dataclass(frozen=True)
class RetrievalSpec:
    candidate_k: int = 1024
    result_k: int = 10
    similarity_threshold: float = 0.2
    vector_similarity_weight: float = 0.3
    keyword_enabled: bool = False
    reranker_id: str | None = None
    highlight: bool = False


@dataclass(frozen=True)
class GenerationSpec:
    model: str = "model"
    model_name: str | None = None
    temperature: float = 0.1
    top_p: float = 0.3
    presence_penalty: float = 0.4
    frequency_penalty: float = 0.7
    context_k: int = 6
    candidate_k: int = 1024
    similarity_threshold: float = 0.2
    vector_similarity_weight: float = 0.3
    reranker_id: str | None = None
    prompt: str | None = None
    empty_response: str = ""
    show_quote: bool = True
    name_prefix: str = "benchmark-chat"

    def chat_payload(self, dataset_ids: Sequence[str]) -> dict[str, Any]:
        llm = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }
        if self.model_name:
            llm["model_name"] = self.model_name
        prompt: dict[str, Any] = {
            "similarity_threshold": self.similarity_threshold,
            # RAGFlow's chat API names the complement, unlike /retrieval.
            "keywords_similarity_weight": 1.0 - self.vector_similarity_weight,
            "top_n": self.context_k,
            "top_k": self.candidate_k,
            "rerank_model": self.reranker_id or "",
            "empty_response": self.empty_response,
            "show_quote": self.show_quote,
            "variables": [{"key": "knowledge", "optional": False}],
        }
        if self.prompt is not None:
            prompt["prompt"] = self.prompt
        return {"dataset_ids": list(dataset_ids), "llm": llm, "prompt": prompt}

    def resource_name(self, dataset_ids: Sequence[str]) -> str:
        return deterministic_resource_name(
            self.name_prefix, self.chat_payload(dataset_ids)
        )


@dataclass(frozen=True)
class PreparedRagflowTarget:
    dataset_id: str
    chat_id: str | None
    document_ids: tuple[str, ...]
    created_dataset: bool = False
    created_chat: bool = False


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    rank: int
    chunk_id: str | None = None
    document_id: str | None = None
    dataset_id: str | None = None
    document_name: str | None = None
    score: float | None = None
    vector_score: float | None = None
    keyword_score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    chunks: tuple[RetrievedChunk, ...]
    total: int
    raw_response: Mapping[str, Any]
    total_seconds: float


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    chunks: tuple[RetrievedChunk, ...]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    raw_response: Mapping[str, Any]
    total_seconds: float


def deterministic_resource_name(prefix: str, settings: Mapping[str, Any]) -> str:
    """Build a stable, API-safe name from the effective settings."""
    safe_prefix = re.sub(r"[^A-Za-z0-9._-]+", "-", prefix).strip("-._")
    safe_prefix = safe_prefix or "benchmark"
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str)
    suffix = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{safe_prefix[:115]}-{suffix}"[:128]


class RagflowClient:
    """Typed client for the public RAGFlow/optimAIseRAG benchmark surface."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.25,
        transport: HttpTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        parsed_base_url = urllib.parse.urlsplit(base_url)
        if (
            parsed_base_url.scheme not in {"http", "https"}
            or not parsed_base_url.hostname
        ):
            raise ValueError("RAGFlow base_url must be an absolute http(s) URL")
        if parsed_base_url.username or parsed_base_url.password:
            raise ValueError("RAGFlow base_url must not contain credentials")
        if parsed_base_url.query or parsed_base_url.fragment:
            raise ValueError("RAGFlow base_url must not contain a query or fragment")
        if not api_key:
            raise ValueError("RAGFlow API key must not be empty")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._transport = transport or UrllibTransport()
        self._sleep = sleep
        self._monotonic = monotonic

    @classmethod
    def from_env(
        cls,
        base_url: str,
        api_key_env: str = "RAG_MANAGED_API_KEY",
        **kwargs: Any,
    ) -> "RagflowClient":
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"Required API key environment variable is unset: {api_key_env}"
            )
        return cls(base_url, api_key, **kwargs)

    def __repr__(self) -> str:
        return (
            f"RagflowClient(base_url={self.base_url!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, api_key=<redacted>)"
        )

    def health(self) -> Mapping[str, Any]:
        # Health lives outside the /api/v1 prefix and intentionally needs no auth.
        return self._request_json(
            "GET",
            "/v1/system/healthz",
            api_prefix=False,
            authenticated=False,
            envelope=False,
            retry_safe=True,
        )

    def ensure_ready(self) -> None:
        status = self.health()
        if str(status.get("status", "")).lower() != "ok":
            raise RagflowApiError("RAGFlow dependency health check is not OK")

    def list_datasets(self, *, name: str | None = None) -> list[dict[str, Any]]:
        data = self._request_json(
            "GET", "/datasets", query={"name": name} if name else None
        )
        if not isinstance(data, list):
            raise RagflowApiError("RAGFlow dataset list response has an invalid shape")
        return [item for item in data if isinstance(item, dict)]

    def ensure_dataset(self, spec: DatasetSpec) -> tuple[dict[str, Any], bool]:
        matches = [
            item
            for item in self.list_datasets(name=spec.resource_name)
            if item.get("name") == spec.resource_name
        ]
        if matches:
            return matches[0], False
        payload: dict[str, Any] = {
            "name": spec.resource_name,
            "description": spec.description,
            "chunk_method": spec.chunk_method,
            "parser_config": dict(spec.parser_config),
        }
        if spec.embedding_model:
            payload["embedding_model"] = spec.embedding_model
        data = self._request_json(
            "POST", "/datasets", payload=payload, retry_safe=False
        )
        if not isinstance(data, dict) or not data.get("id"):
            raise RagflowApiError(
                "RAGFlow create-dataset response omitted the dataset ID"
            )
        return data, True

    def upload_documents(
        self, dataset_id: str, documents: Sequence[UploadDocument]
    ) -> list[dict[str, Any]]:
        if not documents:
            return []
        body, content_type = _multipart_documents(documents)
        data = self._request_json(
            "POST",
            f"/datasets/{_path_id(dataset_id)}/documents",
            body=body,
            content_type=content_type,
            retry_safe=False,
        )
        if not isinstance(data, list):
            raise RagflowApiError("RAGFlow upload response has an invalid shape")
        return [item for item in data if isinstance(item, dict)]

    def list_documents(self, dataset_id: str) -> list[dict[str, Any]]:
        all_docs: list[dict[str, Any]] = []
        page = 1
        while True:
            data = self._request_json(
                "GET",
                f"/datasets/{_path_id(dataset_id)}/documents",
                query={"page": page, "page_size": 100},
            )
            if not isinstance(data, dict) or not isinstance(data.get("docs"), list):
                raise RagflowApiError(
                    "RAGFlow document list response has an invalid shape"
                )
            batch = [item for item in data["docs"] if isinstance(item, dict)]
            all_docs.extend(batch)
            if len(batch) < 100:
                return all_docs
            page += 1

    def parse_documents(self, dataset_id: str, document_ids: Sequence[str]) -> None:
        if not document_ids:
            return
        self._request_json(
            "POST",
            f"/datasets/{_path_id(dataset_id)}/chunks",
            payload={"document_ids": list(document_ids)},
            retry_safe=False,
        )

    def wait_for_documents(
        self,
        dataset_id: str,
        document_ids: Sequence[str],
        *,
        timeout_seconds: float = 1800.0,
        poll_interval_seconds: float = 2.0,
    ) -> list[dict[str, Any]]:
        pending = set(document_ids)
        completed: dict[str, dict[str, Any]] = {}
        deadline = self._monotonic() + timeout_seconds
        while pending:
            docs = self.list_documents(dataset_id)
            by_id = {str(doc.get("id")): doc for doc in docs if doc.get("id")}
            for document_id in list(pending):
                doc = by_id.get(document_id)
                if not doc:
                    continue
                state = str(doc.get("run") or "").upper()
                progress = float(doc.get("progress") or 0.0)
                if state in {"FAIL", "CANCEL"}:
                    message = str(doc.get("progress_msg") or "no server detail")
                    raise RagflowIndexingError(
                        f"RAGFlow document {document_id} ended as {state}: {message}"
                    )
                if state == "DONE" or progress >= 1.0:
                    completed[document_id] = doc
                    pending.remove(document_id)
            if not pending:
                break
            if self._monotonic() >= deadline:
                raise RagflowTimeoutError(
                    f"Timed out waiting for {len(pending)} RAGFlow document(s)"
                )
            self._sleep(poll_interval_seconds)
        return [completed[document_id] for document_id in document_ids]

    def retrieve(
        self,
        question: str,
        dataset_ids: Sequence[str],
        spec: RetrievalSpec,
        *,
        document_ids: Sequence[str] | None = None,
    ) -> RetrievalResult:
        payload: dict[str, Any] = {
            "question": question,
            "dataset_ids": list(dataset_ids),
            "page": 1,
            "page_size": spec.result_k,
            "similarity_threshold": spec.similarity_threshold,
            "vector_similarity_weight": spec.vector_similarity_weight,
            "top_k": spec.candidate_k,
            "keyword": spec.keyword_enabled,
            "highlight": spec.highlight,
        }
        if document_ids:
            payload["document_ids"] = list(document_ids)
        if spec.reranker_id:
            payload["rerank_id"] = spec.reranker_id
        started = self._monotonic()
        data, raw = self._request_envelope(
            "POST", "/retrieval", payload=payload, retry_safe=True
        )
        elapsed = self._monotonic() - started
        if not isinstance(data, dict):
            raise RagflowApiError("RAGFlow retrieval response has an invalid shape")
        chunks = _normalize_chunks(data.get("chunks"))
        return RetrievalResult(
            chunks=chunks,
            total=int(data.get("total") or len(chunks)),
            raw_response=raw,
            total_seconds=elapsed,
        )

    def list_chats(self, *, name: str | None = None) -> list[dict[str, Any]]:
        data = self._request_json(
            "GET", "/chats", query={"name": name} if name else None
        )
        if not isinstance(data, list):
            raise RagflowApiError("RAGFlow chat list response has an invalid shape")
        return [item for item in data if isinstance(item, dict)]

    def ensure_chat(
        self, dataset_ids: Sequence[str], spec: GenerationSpec
    ) -> tuple[dict[str, Any], bool]:
        name = spec.resource_name(dataset_ids)
        payload = {"name": name, **spec.chat_payload(dataset_ids)}
        matches = [
            item for item in self.list_chats(name=name) if item.get("name") == name
        ]
        if matches:
            chat = matches[0]
            self._request_json(
                "PUT",
                f"/chats/{_path_id(str(chat['id']))}",
                payload=payload,
                retry_safe=False,
            )
            return chat, False
        data = self._request_json("POST", "/chats", payload=payload, retry_safe=False)
        if not isinstance(data, dict) or not data.get("id"):
            raise RagflowApiError("RAGFlow create-chat response omitted the chat ID")
        return data, True

    def generate(
        self, chat_id: str, question: str, spec: GenerationSpec
    ) -> GenerationResult:
        payload = {
            "model": spec.model,
            "messages": [{"role": "user", "content": question}],
            "stream": False,
            "reference": True,
        }
        started = self._monotonic()
        raw = self._request_json(
            "POST",
            f"/chats_openai/{_path_id(chat_id)}/chat/completions",
            # Generation is billable and the API exposes no idempotency key.
            payload=payload,
            envelope=False,
            retry_safe=False,
        )
        elapsed = self._monotonic() - started
        if not isinstance(raw, dict):
            raise RagflowApiError("RAGFlow completion response has an invalid shape")
        choices = raw.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
        ):
            raise RagflowApiError("RAGFlow completion response omitted choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RagflowApiError("RAGFlow completion response omitted the message")
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        answer = str(message.get("content") or "")
        return GenerationResult(
            answer=answer,
            chunks=_normalize_chunks(message.get("reference")),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            raw_response=raw,
            total_seconds=elapsed,
        )

    def delete_dataset(self, dataset_id: str) -> None:
        self._request_json(
            "DELETE", "/datasets", payload={"ids": [dataset_id]}, retry_safe=False
        )

    def delete_chat(self, chat_id: str) -> None:
        self._request_json(
            "DELETE", "/chats", payload={"ids": [chat_id]}, retry_safe=False
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        body: bytes | None = None,
        query: Mapping[str, Any] | None = None,
        content_type: str = "application/json",
        api_prefix: bool = True,
        authenticated: bool = True,
        envelope: bool = True,
        retry_safe: bool | None = None,
    ) -> Any:
        data, _ = self._request_envelope(
            method,
            path,
            payload=payload,
            body=body,
            query=query,
            content_type=content_type,
            api_prefix=api_prefix,
            authenticated=authenticated,
            envelope=envelope,
            retry_safe=retry_safe,
        )
        return data

    def _request_envelope(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        body: bytes | None = None,
        query: Mapping[str, Any] | None = None,
        content_type: str = "application/json",
        api_prefix: bool = True,
        authenticated: bool = True,
        envelope: bool = True,
        retry_safe: bool | None = None,
    ) -> tuple[Any, Mapping[str, Any]]:
        if payload is not None and body is not None:
            raise ValueError("Use either payload or body, not both")
        encoded = body
        if payload is not None:
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        prefix = "/api/v1" if api_prefix else ""
        url = f"{self.base_url}{prefix}{path}"
        if query:
            values = {key: value for key, value in query.items() if value is not None}
            url = f"{url}?{urllib.parse.urlencode(values)}"
        headers = {"Accept": "application/json", "Content-Type": content_type}
        if authenticated:
            headers["Authorization"] = f"Bearer {self._api_key}"
        method = method.upper()
        can_retry = method == "GET" if retry_safe is None else retry_safe
        attempts = self.max_retries + 1 if can_retry else 1
        response: TransportResponse | None = None
        for attempt in range(attempts):
            try:
                response = self._transport.request(
                    method, url, headers, encoded, self.timeout_seconds
                )
            except RagflowTimeoutError:
                if attempt + 1 >= attempts:
                    raise
                self._sleep(self.retry_backoff_seconds * (2**attempt))
                continue
            if (
                response.status not in {429, 500, 502, 503, 504}
                or attempt + 1 >= attempts
            ):
                break
            self._sleep(self.retry_backoff_seconds * (2**attempt))
        assert response is not None
        try:
            raw = json.loads(response.body.decode("utf-8")) if response.body else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RagflowApiError(
                f"RAGFlow returned non-JSON content (HTTP {response.status})"
            ) from exc
        if not isinstance(raw, dict):
            raise RagflowApiError("RAGFlow response must be a JSON object")
        if not 200 <= response.status < 300:
            message = str(raw.get("message") or "request failed")
            raise RagflowApiError(f"RAGFlow HTTP {response.status}: {message}")
        if envelope:
            code = raw.get("code")
            if code != 0:
                raise RagflowApiError(
                    f"RAGFlow API code {code}: {raw.get('message') or 'request failed'}"
                )
            return raw.get("data"), raw
        # OpenAI-compatible errors can still use the RAGFlow envelope.
        if raw.get("code") not in (None, 0):
            raise RagflowApiError(
                f"RAGFlow API code {raw.get('code')}: {raw.get('message') or 'request failed'}"
            )
        return raw, raw


class RagflowAdapter:
    """Framework-compatible managed adapter with an explicit provider API."""

    name = "ragflow"

    def __init__(
        self,
        client: RagflowClient,
        dataset: DatasetSpec,
        retrieval: RetrievalSpec,
        generation: GenerationSpec,
        *,
        indexing_timeout_seconds: float = 1800.0,
        poll_interval_seconds: float = 2.0,
        cleanup_resources: bool = False,
    ) -> None:
        self.client = client
        self.dataset_spec = dataset
        self.retrieval_spec = retrieval
        self.generation_spec = generation
        self.indexing_timeout_seconds = indexing_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.cleanup_resources = cleanup_resources
        self.target: PreparedRagflowTarget | None = None
        self._source_by_document_id: dict[str, str] = {}
        self._source_by_document_name: dict[str, str] = {}

    @classmethod
    def from_config(cls, config: Any) -> "RagflowAdapter":
        """Map provider-neutral benchmark settings to RAGFlow's API names."""
        if not config.rag_managed_base_url:
            raise ValueError("RAG_MANAGED_BASE_URL is required for the ragflow adapter")
        options = _json_object(
            config.rag_managed_options_json, "RAG_MANAGED_OPTIONS_JSON"
        )
        ingestion = _json_object(
            config.ingestion_options_json, "INGESTION_OPTIONS_JSON"
        )
        allowed = {
            "max_retries",
            "retry_backoff_seconds",
            "dataset_description",
            "health_check",
        }
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ValueError(
                "Unsupported RAG_MANAGED_OPTIONS_JSON key(s) for ragflow: "
                + ", ".join(unknown)
            )
        if "parser_config" in ingestion:
            extra_ingestion = sorted(set(ingestion) - {"parser_config"})
            if extra_ingestion:
                raise ValueError(
                    "When INGESTION_OPTIONS_JSON contains parser_config it cannot "
                    "also contain: " + ", ".join(extra_ingestion)
                )
            parser_config = ingestion["parser_config"]
        else:
            parser_config = ingestion
        if not isinstance(parser_config, dict):
            raise ValueError("INGESTION_OPTIONS_JSON.parser_config must be an object")
        prefix = str(config.rag_managed_dataset_prefix)
        if not config.rag_managed_reuse_resources:
            prefix = f"{prefix}-{secrets.token_hex(5)}"
        template = get_template(config.prompt_template)
        prompt = f"{template.system_prompt}\n\nKnowledge base:\n{{knowledge}}"
        reranker = (
            config.reranker_model
            if getattr(config, "reranker_model", "none") != "none"
            else None
        )
        client = RagflowClient.from_env(
            config.rag_managed_base_url,
            api_key_env=config.rag_managed_api_key_env,
            timeout_seconds=config.rag_managed_request_timeout_seconds,
            max_retries=int(options.get("max_retries", 2)),
            retry_backoff_seconds=float(options.get("retry_backoff_seconds", 0.25)),
        )
        adapter = cls(
            client,
            DatasetSpec(
                embedding_model=config.embedding_model,
                chunk_method=config.ingestion_method or "naive",
                parser_config=parser_config,
                name_prefix=prefix,
                description=str(
                    options.get("dataset_description")
                    or "Managed by Benchmarking Framework"
                ),
            ),
            RetrievalSpec(
                candidate_k=config.retrieval_candidate_k,
                result_k=config.retrieval_top_k,
                similarity_threshold=config.retrieval_similarity_threshold,
                vector_similarity_weight=config.retrieval_vector_similarity_weight,
                keyword_enabled=config.retrieval_keyword_enabled,
                reranker_id=reranker,
            ),
            GenerationSpec(
                model="model",
                model_name=config.llm_model,
                temperature=config.generation_temperature,
                top_p=config.generation_top_p,
                presence_penalty=config.generation_presence_penalty,
                frequency_penalty=config.generation_frequency_penalty,
                context_k=config.generation_context_k,
                candidate_k=config.retrieval_candidate_k,
                similarity_threshold=config.retrieval_similarity_threshold,
                vector_similarity_weight=config.retrieval_vector_similarity_weight,
                reranker_id=reranker,
                prompt=str(prompt),
                empty_response="",
                show_quote=True,
                name_prefix=f"{prefix}-chat",
            ),
            indexing_timeout_seconds=config.rag_managed_poll_timeout_seconds,
            poll_interval_seconds=config.rag_managed_poll_interval_seconds,
            cleanup_resources=config.rag_managed_cleanup,
        )
        adapter._health_check = bool(options.get("health_check", True))
        return adapter

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            ingestion=True,
            retrieval=True,
            generation=True,
            chunk_configuration=True,
            embedding_configuration=True,
            reranking=True,
            references=True,
            token_usage=True,
            cleanup=True,
        )

    def supports_components(self) -> dict[str, bool]:
        return {
            "chunker": False,
            "embedder": False,
            "retriever": False,
            "reranker": False,
            "llm": False,
            "prompt": False,
        }

    def prepare(
        self, config: Any, data: list[dict], corpus: list[dict] | None = None
    ) -> PreparedTarget:
        stage = str(getattr(config, "benchmark_stage", "all")).lower()
        target = self.prepare_run(
            corpus or data, require_chat=stage not in {"index", "retrieve"}
        )
        return PreparedTarget(
            target_id=target.chat_id or target.dataset_id,
            dataset_ids=(target.dataset_id,),
            document_ids=target.document_ids,
            metadata={
                "provider": "ragflow",
                "chat_id": target.chat_id,
                "created_dataset": target.created_dataset,
                "created_chat": target.created_chat,
            },
        )

    def prepare_run(
        self,
        corpus: Sequence[Mapping[str, Any]],
        *,
        require_chat: bool = True,
    ) -> PreparedRagflowTarget:
        if getattr(self, "_health_check", True):
            self.client.ensure_ready()
        effective_dataset_spec = replace(
            self.dataset_spec, corpus_digest=_corpus_digest(corpus)
        )
        dataset, created_dataset = self.client.ensure_dataset(effective_dataset_spec)
        dataset_id = str(dataset["id"])
        created_chat = False
        chat: dict[str, Any] | None = None
        try:
            uploads = [
                _corpus_document(item, index) for index, item in enumerate(corpus)
            ]
            source_by_name = {
                upload.name: _corpus_source_id(item, index)
                for index, (upload, item) in enumerate(zip(uploads, corpus))
            }
            expected = {item.name: item for item in uploads}
            existing = {
                str(doc.get("name")): doc
                for doc in self.client.list_documents(dataset_id)
            }
            missing = [
                document for name, document in expected.items() if name not in existing
            ]
            added = self.client.upload_documents(dataset_id, missing)
            for doc in added:
                existing[str(doc.get("name"))] = doc
            unresolved = [name for name in expected if name not in existing]
            if unresolved:
                raise RagflowApiError(
                    f"RAGFlow did not return {len(unresolved)} expected uploaded document(s)"
                )
            documents = [existing[name] for name in expected]
            self._source_by_document_id = {
                str(doc["id"]): source_by_name[str(doc["name"])] for doc in documents
            }
            self._source_by_document_name = {
                str(doc["name"]): source_by_name[str(doc["name"])] for doc in documents
            }
            for doc in documents:
                if str(doc.get("run") or "").upper() in {"FAIL", "CANCEL"}:
                    raise RagflowIndexingError(
                        f"Existing deterministic document {doc.get('name')} is {doc.get('run')}"
                    )
            to_parse = [
                str(doc["id"])
                for doc in documents
                if str(doc.get("run") or "").upper() != "DONE"
                and float(doc.get("progress") or 0.0) < 1.0
            ]
            self.client.parse_documents(dataset_id, to_parse)
            if to_parse:
                self.client.wait_for_documents(
                    dataset_id,
                    to_parse,
                    timeout_seconds=self.indexing_timeout_seconds,
                    poll_interval_seconds=self.poll_interval_seconds,
                )
            if require_chat:
                chat, created_chat = self.client.ensure_chat(
                    [dataset_id], self.generation_spec
                )
        except Exception as exc:
            try:
                if chat is not None and created_chat:
                    self.client.delete_chat(str(chat["id"]))
                if created_dataset:
                    self.client.delete_dataset(dataset_id)
            except RagflowError as cleanup_exc:
                exc.add_note(f"RAGFlow rollback also failed: {cleanup_exc}")
            raise
        self.target = PreparedRagflowTarget(
            dataset_id=dataset_id,
            chat_id=str(chat["id"]) if chat is not None else None,
            document_ids=tuple(str(doc["id"]) for doc in documents),
            created_dataset=created_dataset,
            created_chat=created_chat,
        )
        return self.target

    def retrieve(
        self, target: PreparedTarget, sample: dict, config: Any
    ) -> CanonicalRetrievalResult:
        provider_target = self._resolve_target(target)
        result = self.client.retrieve(
            str(sample["question"]), [provider_target.dataset_id], self.retrieval_spec
        )
        return self._canonical_retrieval(result)

    def retrieve_question(self, question: str) -> RetrievalResult:
        target = self._require_target()
        return self.client.retrieve(question, [target.dataset_id], self.retrieval_spec)

    def generate(
        self,
        target: PreparedTarget,
        sample: dict,
        config: Any,
        retrieval: CanonicalRetrievalResult | None = None,
    ) -> AdapterGenerationResult:
        provider_target = self._resolve_target(target)
        if provider_target.chat_id is None:
            raise RagflowError(
                "This RAGFlow target was prepared without a chat; generation is "
                "unavailable for BENCHMARK_STAGE=index/retrieve"
            )
        result = self.client.generate(
            provider_target.chat_id, str(sample["question"]), self.generation_spec
        )
        return AdapterGenerationResult(
            answer=result.answer,
            retrieval=self._canonical_retrieval_from_chunks(
                result.chunks, result.total_seconds, result.raw_response
            ),
            raw_response=dict(result.raw_response),
            total_seconds=result.total_seconds,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
            raw_content=result.answer,
            answer_valid=bool(result.answer.strip()),
        )

    def answer(self, sample: dict, config: Any = None) -> RagSystemOutput:
        target = self._require_target()
        if target.chat_id is None:
            raise RagflowError("RAGFlow target has no chat configured for generation")
        result = self.client.generate(
            target.chat_id, str(sample["question"]), self.generation_spec
        )
        contexts = [chunk.text for chunk in result.chunks]
        metadata = [dict(chunk.metadata) for chunk in result.chunks]
        return RagSystemOutput(
            answer=result.answer,
            contexts=contexts,
            metadata=metadata,
            raw_response=dict(result.raw_response),
            total_seconds=result.total_seconds,
            token_count=result.output_tokens,
            tokens_per_second=(
                result.output_tokens / result.total_seconds
                if result.output_tokens and result.total_seconds > 0
                else 0.0
            ),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
            raw_content=result.answer,
            answer_valid=bool(result.answer.strip()),
        )

    def cleanup(self, target: PreparedTarget | None = None, config: Any = None) -> None:
        if not self.cleanup_resources:
            return
        provider_target = (
            self._resolve_target(target) if target is not None else self.target
        )
        if provider_target is None:
            return
        # Never delete reused resources; ownership may belong to another run.
        if provider_target.created_chat and provider_target.chat_id is not None:
            self.client.delete_chat(provider_target.chat_id)
        if provider_target.created_dataset:
            self.client.delete_dataset(provider_target.dataset_id)
        self.target = None

    def _require_target(self) -> PreparedRagflowTarget:
        if self.target is None:
            raise RagflowError("RagflowAdapter.prepare_run() must be called first")
        return self.target

    def _resolve_target(self, target: PreparedTarget | None) -> PreparedRagflowTarget:
        current = self._require_target()
        expected_id = current.chat_id or current.dataset_id
        if target is not None and target.target_id != expected_id:
            raise RagflowError("Prepared target does not belong to this adapter run")
        return current

    def _canonical_retrieval(self, result: RetrievalResult) -> CanonicalRetrievalResult:
        return self._canonical_retrieval_from_chunks(
            result.chunks, result.total_seconds, result.raw_response
        )

    def _canonical_retrieval_from_chunks(
        self,
        chunks: Sequence[RetrievedChunk],
        total_seconds: float,
        raw_response: Mapping[str, Any],
    ) -> CanonicalRetrievalResult:
        canonical = []
        for chunk in chunks:
            source_id = (
                self._source_by_document_id.get(chunk.document_id or "")
                or self._source_by_document_name.get(chunk.document_name or "")
                or chunk.document_name
            )
            metadata = dict(chunk.metadata)
            if source_id:
                metadata.update({"source_id": source_id, "doc_id": source_id})
            canonical.append(
                CanonicalChunk(
                    text=chunk.text,
                    rank=chunk.rank,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_id=source_id,
                    score=chunk.score,
                    metadata=metadata,
                )
            )
        return CanonicalRetrievalResult(
            chunks=tuple(canonical),
            total_seconds=total_seconds,
            raw_response=dict(raw_response),
        )


def _path_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("RAGFlow resource ID contains invalid characters")
    return value


def _safe_filename(value: str) -> str:
    name = Path(value).name.replace("\x00", "")
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    if not name:
        raise ValueError("RAGFlow upload filename is empty after sanitization")
    return name[:180]


def _multipart_documents(documents: Sequence[UploadDocument]) -> tuple[bytes, str]:
    boundary = f"benchmark-{secrets.token_hex(16)}"
    chunks: list[bytes] = []
    for document in documents:
        name = _safe_filename(document.name)
        media_type = (
            document.content_type
            or mimetypes.guess_type(name)[0]
            or "application/octet-stream"
        )
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'.encode(),
                f"Content-Type: {media_type}\r\n\r\n".encode(),
                document.content,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _normalize_chunks(value: Any) -> tuple[RetrievedChunk, ...]:
    if isinstance(value, dict):
        value = value.get("chunks", [])
    if not isinstance(value, list):
        return ()
    normalized: list[RetrievedChunk] = []
    for rank, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        text = item.get("content") or item.get("content_with_weight") or ""
        known = {
            "id",
            "chunk_id",
            "content",
            "content_with_weight",
            "document_id",
            "doc_id",
            "dataset_id",
            "kb_id",
            "document_name",
            "document_keyword",
            "docnm_kwd",
            "similarity",
            "vector_similarity",
            "term_similarity",
        }
        metadata = {key: val for key, val in item.items() if key not in known}
        metadata.update(
            {
                "chunk_id": item.get("id") or item.get("chunk_id"),
                "document_id": item.get("document_id") or item.get("doc_id"),
                "dataset_id": item.get("dataset_id") or item.get("kb_id"),
                "document_name": item.get("document_name")
                or item.get("document_keyword")
                or item.get("docnm_kwd"),
                "rank": rank,
                "score": item.get("similarity"),
            }
        )
        normalized.append(
            RetrievedChunk(
                text=str(text),
                rank=rank,
                chunk_id=_optional_str(item.get("id") or item.get("chunk_id")),
                document_id=_optional_str(
                    item.get("document_id") or item.get("doc_id")
                ),
                dataset_id=_optional_str(item.get("dataset_id") or item.get("kb_id")),
                document_name=_optional_str(
                    item.get("document_name")
                    or item.get("document_keyword")
                    or item.get("docnm_kwd")
                ),
                score=_optional_float(item.get("similarity")),
                vector_score=_optional_float(item.get("vector_similarity")),
                keyword_score=_optional_float(item.get("term_similarity")),
                metadata=metadata,
            )
        )
    return tuple(normalized)


def _corpus_document(item: Mapping[str, Any], index: int) -> UploadDocument:
    context = item.get("context", "")
    if isinstance(context, list):
        text = "\n\n".join(str(part) for part in context)
    else:
        text = str(context)
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    identity = metadata.get("doc_id") or metadata.get("source_id")
    if not identity:
        identity = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    filename = _safe_filename(f"{index:05d}-{identity}.txt")
    return UploadDocument(filename, text.encode("utf-8"), "text/plain; charset=utf-8")


def _corpus_source_id(item: Mapping[str, Any], index: int) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    source_id = metadata.get("source_id") or metadata.get("doc_id")
    if source_id:
        return str(source_id)
    context = item.get("context", "")
    canonical = json.dumps(context, sort_keys=True, separators=(",", ":"), default=str)
    return f"source-{index:05d}-{hashlib.sha256(canonical.encode()).hexdigest()[:12]}"


def _corpus_digest(corpus: Sequence[Mapping[str, Any]]) -> str:
    """Hash complete ordered source identities and contents for safe reuse."""
    digest = hashlib.sha256()
    for index, item in enumerate(corpus):
        record = {
            "source_id": _corpus_source_id(item, index),
            "context": item.get("context", ""),
        }
        digest.update(
            json.dumps(
                record, sort_keys=True, separators=(",", ":"), default=str
            ).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _json_object(value: str | None, label: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must contain valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return dict(parsed)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
