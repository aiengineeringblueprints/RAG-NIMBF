"""Enterprise RAG Blueprint benchmarked as an MCP host.

The Blueprint itself connects to its configured MCP servers and runs the
model/tool loop. This adapter only sends benchmark questions to the
Blueprint ``/call_llm`` endpoint with ``use_mcp: true`` and translates the
``mcp.events`` trace of every answer into :mod:`benchmark.mcp_host` events,
so the run reports tool-call success rates, latencies, budget exhaustion and
tool selection next to the usual answer-quality metrics.

Transports:

- ``inprocess``: imports the Blueprint chain API from
  ``ENTERPRISE_RAG_BLUEPRINT_PATH`` and calls it through FastAPI's test
  client, so the full endpoint (identity check, guardrails, MCP loop) runs
  without Docker.
- ``http``: posts to a running Blueprint chain service.

Each MCP request carries a freshly signed identity assertion built with
``MCP_IDENTITY_SECRET``, the same secret the Blueprint frontend uses.

Activation (bash):

    ENTERPRISE_RAG_BLUEPRINT_PATH=../Enterprise_RAG_Blueprint \\
    RAG_ADAPTER_MODULES=examples.enterprise_rag_plugin.mcp_adapter \\
    python main.py experiments/enterprise-rag-mcp-host.yaml
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from benchmark.adapters import register_rag_adapter
from benchmark.adapters.base import RagSystemOutput
from benchmark.adapters.components import ComponentBundle
from benchmark.answer_match import answer_matches
from benchmark.providers import parse_model_id
from benchmark.mcp_host import (
    McpHostEvent,
    aggregate_by_category,
    aggregate_mcp_host_metrics,
    mcp_host_diagnostics,
    mcp_host_stage_timings,
)
from examples.enterprise_rag_plugin.adapter import (
    ensure_blueprint_importable,
    replace_blueprint_index,
)

log = logging.getLogger(__name__)

# The Blueprint embeds queries with EMBEDDING_MODEL (chain/retriever.py), by
# default this Ollama model; an index built with another model is unreadable.
BLUEPRINT_EMBEDDING_MODEL = "nomic-embed-text"

_STATUS_MAP = {
    "ok": "ok",
    "error": "tool_error",
    "invalid_arguments": "invalid_arguments",
    "denied": "denied",
    "unavailable": "unavailable",
    "result_too_large": "result_too_large",
    "budget_exhausted": "budget_exhausted",
}
_FAILURE_KINDS = {
    400: "guardrail",
    401: "identity",
    422: "input",
    502: "model_or_tool_protocol",
    503: "configuration",
    504: "timeout",
}


@dataclass(frozen=True)
class TransportResponse:
    status: int
    body: dict[str, Any]


def sign_identity(
    secret: str, subject: str, roles: list[str], now: int | None = None
) -> str:
    """Build the Blueprint frontend's HMAC identity assertion header value."""
    issued = int(time.time()) if now is None else int(now)
    claims = {
        "sub": subject,
        "roles": list(roles),
        "aud": "rag-mcp",
        "iat": issued,
        "exp": issued + 60,
    }
    payload = (
        base64.urlsafe_b64encode(json.dumps(claims, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )
    signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"Bearer {payload}.{signature}"


def _disabled_flag(enabled: bool) -> str:
    return "false" if enabled else "true"


def blueprint_settings_env(config: Any) -> dict[str, str]:
    """Blueprint environment that runs it with this benchmark config.

    The Blueprint reads these variables per request (chain/handle_llms.py,
    chain/retriever.py) and talks to every LLM through an OpenAI-compatible
    client, so Ollama models go through Ollama's ``/v1`` API.
    """
    llm_provider = config.llm_provider
    llm_name = config.llm_model.removeprefix(f"{llm_provider}:")
    llm_key = config.llm_api_key()
    key_env = getattr(config, "enterprise_rag_mcp_llm_api_key_env", None)
    if key_env:
        llm_key = os.environ.get(key_env)
        if not llm_key:
            raise ValueError(
                f"enterprise_rag_mcp_llm_api_key_env names {key_env}, which is not set"
            )
    if llm_provider == "ollama":
        llm_url = config.llm_base_url().rstrip("/") + "/v1"
        llm_key = llm_key or "ollama"
    else:
        llm_url = config.llm_base_url()
        # The OpenAI client needs some key, even for endpoints without auth.
        llm_key = llm_key or "EMPTY"
    return {
        "LLM_SOURCE_ANSWER": "openai",
        "MODEL_NAME": llm_name,
        "OPENAI_BASE_URL": llm_url,
        "OPENAI_API_KEY": llm_key,
        "LLM_TEMPERATURE": f"{config.generation_temperature:g}",
        "EMBEDDING_MODEL": parse_model_id(config.embedding_model)[1],
        "OLLAMA_BASE_URL": config.embedding_base_url(),
        # Sent as Bearer by the Blueprint; empty means no auth header.
        "OLLAMA_API_KEY": config.embedding_api_key() or "",
        "RETRIEVER_TOP_K": str(config.retrieval_top_k),
        "DISABLE_QUERY_REFINEMENT": _disabled_flag(config.enterprise_rag_mcp_query_refinement),
        "DISABLE_GUARDRAILS": _disabled_flag(config.enterprise_rag_mcp_guardrails),
    }


@contextlib.contextmanager
def _scoped_env(values: dict[str, str]) -> Iterator[None]:
    """Set environment variables for one in-process request, then restore them."""
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _ollama_model(name: str) -> str:
    name = parse_model_id(name)[1]
    return name[: -len(":latest")] if name.endswith(":latest") else name


def _json_body(raw: bytes) -> dict[str, Any]:
    try:
        parsed = json.loads(raw.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"detail": raw.decode("utf-8", "replace")[:500]}
    return parsed if isinstance(parsed, dict) else {"detail": str(parsed)[:500]}


class HttpTransport:
    """POST to a running Blueprint chain service."""

    def __init__(self, endpoint_url: str, timeout_seconds: float) -> None:
        self.endpoint_url = endpoint_url
        self.timeout_seconds = timeout_seconds

    def post(self, payload: dict[str, Any], headers: dict[str, str]) -> TransportResponse:
        request = urllib.request.Request(
            self.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                return TransportResponse(int(resp.status), _json_body(resp.read()))
        except urllib.error.HTTPError as exc:
            return TransportResponse(int(exc.code), _json_body(exc.read()))


class InProcessTransport:
    """Call the Blueprint chain API in this process via FastAPI's test client."""

    def __init__(self, config: Any, timeout_seconds: float) -> None:
        ensure_blueprint_importable(config)
        root = getattr(config, "enterprise_rag_blueprint_path", None) or os.environ.get(
            "ENTERPRISE_RAG_BLUEPRINT_PATH"
        )
        if root and (Path(root) / ".env").is_file():
            from dotenv import load_dotenv

            # Values already set (framework .env, shell) take precedence.
            load_dotenv(Path(root) / ".env", override=False)
        if getattr(config, "enterprise_rag_mcp_apply_settings", False):
            import handle_llms  # Blueprint chain/handle_llms.py

            # Older Blueprints read MODEL_NAME etc. once at import and would
            # silently ignore the per-request settings.
            if not hasattr(handle_llms, "retriever_top_k"):
                raise ValueError(
                    "This Blueprint checkout reads its model settings only at "
                    "import; update the Blueprint or set "
                    "enterprise_rag_mcp_apply_settings: false"
                )
        from chain_api import app  # Blueprint chain/chain_api.py
        from fastapi.testclient import TestClient

        self._client = TestClient(app, raise_server_exceptions=False)
        self._client.timeout = timeout_seconds

    def post(self, payload: dict[str, Any], headers: dict[str, str]) -> TransportResponse:
        response = self._client.post("/call_llm", json=payload, headers=headers)
        return TransportResponse(response.status_code, _json_body(response.content))


def _expected_tools(metadata: dict[str, Any]) -> list[str] | None:
    raw = metadata.get("expected_tools")
    if raw is None:
        return None
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [str(item) for item in raw]


def _answer_labels(sample: dict, answer: str) -> dict[str, Any]:
    """Category and deterministic answer match for one sample, when labelled."""
    metadata = sample.get("metadata") or {}
    labels: dict[str, Any] = {}
    if metadata.get("category"):
        labels["category"] = str(metadata["category"])
    patterns = metadata.get("answer_patterns")
    if not patterns and sample.get("ground_truth"):
        patterns = [[str(sample["ground_truth"])]]
    if patterns:
        labels["answer_match"] = answer_matches(answer, patterns)
    return labels


def _capability_name(server: str, alias: str, names: dict[str, str]) -> str:
    """Resolve a Blueprint capability alias (``server__label__hash``)."""
    if alias in names:
        return names[alias]
    if alias.startswith(f"{server}__"):
        label = alias[len(server) + 2 :].rsplit("__", 1)[0]
        if label:
            return label
    return alias


def translate_events(
    events: list[dict[str, Any]], sources: list[dict[str, Any]]
) -> tuple[list[McpHostEvent], int]:
    """Map Blueprint ``mcp.events`` onto MCP host events.

    Returns the translated events and the number of events whose status or
    operation is unknown to this adapter version.
    """
    names = {
        str(s["capability"]): str(s["name"])
        for s in sources
        if s.get("capability") and s.get("name")
    }
    translated: list[McpHostEvent] = []
    skipped = 0
    for event in events:
        status = _STATUS_MAP.get(str(event.get("status")))
        operation = str(event.get("operation"))
        if status is None or operation not in {
            "connect", "discover", "tool", "resource", "call", "loop"
        }:
            skipped += 1
            continue
        server = str(event.get("server", "unknown"))
        alias = event.get("capability")
        duration_ms = event.get("duration_ms")
        translated.append(
            McpHostEvent(
                server=server,
                kind=operation,
                status=status,
                name=_capability_name(server, str(alias), names) if alias else None,
                duration_seconds=(
                    float(duration_ms) / 1000.0 if duration_ms is not None else None
                ),
            )
        )
    return translated, skipped


def _evidence(
    relevant_documents: list[Any], sources: list[dict[str, Any]]
) -> tuple[list[str], list[dict[str, Any]]]:
    mcp_labels = {
        f"MCP {s.get('server')}/{s.get('name')}": s for s in sources if s.get("name")
    }
    contexts: list[str] = []
    metadata: list[dict[str, Any]] = []
    for rank, entry in enumerate(relevant_documents, start=1):
        label, sep, content = str(entry).partition("->")
        if not sep:
            label, content = "unknown", str(entry)
        source = mcp_labels.get(label)
        if source is not None or label.startswith("MCP "):
            server, _, name = label[4:].partition("/")
            metadata.append(
                {
                    "rank": rank,
                    "evidence_type": "mcp",
                    "mcp_server": source.get("server") if source else server,
                    "mcp_capability": source.get("name") if source else name,
                }
            )
        else:
            source_id = Path(label).stem
            metadata.append(
                {
                    "rank": rank,
                    "evidence_type": "document",
                    "source": label,
                    "source_id": source_id,
                    "doc_id": source_id,
                }
            )
            # Same label the Blueprint puts before each document in the prompt
            # (chain/load_chain.py format_docs), so citations are checkable.
            content = f"Document: {label}\n{content}"
        contexts.append(content)
    return contexts, metadata


class EnterpriseRagMcpAdapter:
    """Benchmark the Blueprint's own MCP host behaviour end to end."""

    name = "enterprise_rag_mcp"

    def __init__(
        self,
        config: Any,
        transport: Any = None,
        identity_secret: str | None = None,
    ) -> None:
        self.config = config
        self.bundle = ComponentBundle()
        self._transport = transport
        self._identity_secret = identity_secret
        self._secret = ""

    def supports_components(self) -> dict[str, bool]:
        # Chunker/embedder only build the index; the Blueprint answers with
        # its own retriever, prompts and model.
        return {
            "chunker": True,
            "embedder": True,
            "retriever": False,
            "reranker": False,
            "llm": False,
            "prompt": False,
        }

    def set_components(self, bundle: ComponentBundle) -> None:
        self.bundle = bundle

    @property
    def _roles(self) -> list[str]:
        return [
            role.strip()
            for role in str(self.config.enterprise_rag_mcp_roles).split(",")
            if role.strip()
        ]

    def prepare(
        self,
        config: Any,
        data: list[dict],
        corpus: list[dict] | None = None,
    ) -> None:
        index = bool(config.enterprise_rag_mcp_index_corpus)
        # The in-process transport loads the Blueprint .env, so it must exist
        # before indexing reads INDEX_NAME / VECTORDB_DIR / EMBEDDING_MODEL.
        if self._transport is None:
            timeout = float(config.enterprise_rag_mcp_timeout_seconds)
            if config.enterprise_rag_mcp_transport == "http":
                self._transport = HttpTransport(
                    str(config.enterprise_rag_mcp_endpoint_url), timeout
                )
            else:
                self._transport = InProcessTransport(config, timeout)
        # Resolved after the in-process transport loaded the Blueprint .env.
        self._secret = (
            self._identity_secret
            if self._identity_secret is not None
            else os.environ.get("MCP_IDENTITY_SECRET", "")
        )
        if config.enterprise_rag_mcp_enabled and len(self._secret) < 32:
            raise ValueError(
                "MCP_IDENTITY_SECRET must be set (at least 32 characters) and "
                "match the Blueprint's secret"
            )
        if config.enterprise_rag_mcp_apply_settings:
            blueprint_settings_env(config)  # fail before indexing, not per sample
        if index and not config.enterprise_rag_mcp_apply_settings:
            blueprint_model = os.environ.get("EMBEDDING_MODEL") or BLUEPRINT_EMBEDDING_MODEL
            if _ollama_model(config.embedding_model) != _ollama_model(blueprint_model):
                raise ValueError(
                    f"The Blueprint queries its index with {blueprint_model!r}; set "
                    f"embedding_models to it (got {config.embedding_model!r}), enable "
                    "enterprise_rag_mcp_apply_settings or disable "
                    "enterprise_rag_mcp_index_corpus"
                )
        if index:
            replace_blueprint_index(self.bundle, config, data, corpus=corpus)

    def answer(self, sample: dict, config: Any) -> RagSystemOutput:
        if self._transport is None:
            raise RuntimeError("EnterpriseRagMcpAdapter.prepare() was not called")
        use_mcp = bool(config.enterprise_rag_mcp_enabled)
        payload: dict[str, Any] = {
            "question": str(sample.get("question", "")),
            "use_mcp": use_mcp,
            "prompt_key": config.enterprise_rag_mcp_prompt_key,
            "show_sources": True,
            "user_roles": self._roles,
            "chat_history": [],
        }
        settings = (
            blueprint_settings_env(config)
            if config.enterprise_rag_mcp_apply_settings
            else {}
        )
        if config.enterprise_rag_mcp_used_model:
            payload["used_model"] = config.enterprise_rag_mcp_used_model
        elif settings:
            payload["used_model"] = settings["LLM_SOURCE_ANSWER"]
        headers: dict[str, str] = {}
        if use_mcp:
            headers["Authorization"] = sign_identity(
                self._secret, config.enterprise_rag_mcp_subject, self._roles
            )
        expected = _expected_tools(sample.get("metadata") or {})

        started = time.perf_counter()
        try:
            with _scoped_env(settings):
                response = self._transport.post(payload, headers)
        except Exception as exc:  # noqa: BLE001 -- one failed sample must not stop the run
            return self._failed(
                sample,
                time.perf_counter() - started,
                use_mcp,
                expected,
                error=f"{type(exc).__name__}: {exc}",
                failure_kind="transport",
            )
        elapsed = time.perf_counter() - started

        if response.status != 200:
            detail = response.body.get("detail", "")
            return self._failed(
                sample,
                elapsed,
                use_mcp,
                expected,
                error=f"HTTP {response.status}: {detail}",
                failure_kind=_FAILURE_KINDS.get(response.status, "http_error"),
            )

        body = response.body
        trace = body.get("mcp") if isinstance(body.get("mcp"), dict) else None
        sources = list((trace or {}).get("sources") or [])
        contexts, metadata = _evidence(list(body.get("relevant_documents") or []), sources)
        answer = str(body.get("result") or "")
        diagnostics: dict[str, Any] = {"use_mcp": use_mcp, **_answer_labels(sample, answer)}
        if use_mcp:
            events, skipped = translate_events(list((trace or {}).get("events") or []), sources)
            diagnostics.update(mcp_host_diagnostics(events, expected_tools=expected))
            diagnostics["untranslated_event_count"] = skipped
            diagnostics["request_id"] = (trace or {}).get("request_id")
            if trace is None:
                diagnostics["failure_kind"] = "missing_mcp_trace"
                diagnostics["error"] = "Blueprint response has no mcp trace"
        return RagSystemOutput(
            answer=answer,
            contexts=contexts,
            metadata=metadata,
            raw_response=body,
            total_seconds=elapsed,
            ttft_seconds=elapsed,
            raw_content=answer,
            answer_valid=bool(answer.strip()),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _failed(
        sample: dict,
        elapsed: float,
        use_mcp: bool,
        expected: list[str] | None,
        *,
        error: str,
        failure_kind: str,
    ) -> RagSystemOutput:
        log.warning("Blueprint request failed (%s): %s", failure_kind, error)
        diagnostics: dict[str, Any] = {"use_mcp": use_mcp}
        if use_mcp:
            diagnostics.update(mcp_host_diagnostics([], expected_tools=expected))
        diagnostics.update({"error": error, "failure_kind": failure_kind})
        diagnostics.update(_answer_labels(sample, ""))
        return RagSystemOutput(
            answer="",
            total_seconds=elapsed,
            answer_valid=False,
            diagnostics=diagnostics,
        )

    def diagnostic_stage_timings(self, diagnostics: dict[str, Any]) -> dict[str, float]:
        if "mcp_host" not in diagnostics:
            return {}
        return mcp_host_stage_timings(diagnostics)

    def aggregate_metrics(self, diagnostics: list[dict[str, Any]]) -> dict[str, Any] | None:
        summary = {
            **(aggregate_mcp_host_metrics(diagnostics) or {}),
            **(aggregate_by_category(diagnostics) or {}),
        }
        return summary or None


register_rag_adapter("enterprise_rag_mcp", lambda c: EnterpriseRagMcpAdapter(c))
