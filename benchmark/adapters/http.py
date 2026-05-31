"""HTTP adapter for black-box RAG systems."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from benchmark.adapters.base import RagSystemOutput


def _lookup(data: dict[str, Any], path: str, default: Any = None) -> Any:
    """Read a dotted field path from a response dictionary."""
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _as_contexts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        contexts: list[str] = []
        for item in value:
            if isinstance(item, str):
                contexts.append(item)
            elif isinstance(item, dict):
                text = (
                    item.get("text")
                    or item.get("content")
                    or item.get("page_content")
                    or item.get("context")
                )
                if text is not None:
                    contexts.append(str(text))
            elif item is not None:
                contexts.append(str(item))
        return contexts
    return [str(value)]


def _as_metadata(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    return []


@dataclass(frozen=True)
class HttpRagAdapter:
    """Call an external RAG service over JSON HTTP POST."""

    endpoint_url: str
    timeout_seconds: float = 60.0
    answer_field: str = "answer"
    contexts_field: str = "contexts"
    metadata_field: str = "metadata"
    timings_field: str = "timings"
    headers: dict[str, str] | None = None

    name: str = "http"

    @classmethod
    def from_config(cls, config: Any) -> "HttpRagAdapter":
        if not config.rag_http_endpoint_url:
            raise ValueError(
                "RAG_HTTP_ENDPOINT_URL is required when RAG_SYSTEM_ADAPTER=http"
            )
        headers = {"Content-Type": "application/json"}
        if config.rag_http_headers:
            try:
                parsed = json.loads(config.rag_http_headers)
            except json.JSONDecodeError as exc:
                raise ValueError("RAG_HTTP_HEADERS must be valid JSON") from exc
            if not isinstance(parsed, dict):
                raise ValueError("RAG_HTTP_HEADERS must be a JSON object")
            headers.update({str(k): str(v) for k, v in parsed.items()})
        if config.rag_http_auth_header and config.rag_http_auth_value:
            headers[config.rag_http_auth_header] = config.rag_http_auth_value
        return cls(
            endpoint_url=config.rag_http_endpoint_url,
            timeout_seconds=config.rag_http_timeout_seconds,
            answer_field=config.rag_http_answer_field,
            contexts_field=config.rag_http_contexts_field,
            metadata_field=config.rag_http_metadata_field,
            timings_field=config.rag_http_timings_field,
            headers=headers,
        )

    def prepare(
        self,
        config: Any,
        data: list[dict],
        corpus: list[dict] | None = None,
    ) -> None:
        return None

    def answer(self, sample: dict, config: Any) -> RagSystemOutput:
        payload = {
            "question": sample["question"],
            "metadata": sample.get("metadata", {}),
            "ground_truth": sample.get("ground_truth"),
            "config": {
                "name": config.name,
                "retrieval_top_k": config.retrieval_top_k,
                "prompt_template": config.prompt_template,
                "dataset_name": config.dataset_name,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint_url,
            data=body,
            headers=self.headers or {"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                response_body = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"HTTP RAG adapter request failed: {exc}") from exc
        total = time.perf_counter() - started

        try:
            raw = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("HTTP RAG adapter response was not valid JSON") from exc
        if not isinstance(raw, dict):
            raise RuntimeError("HTTP RAG adapter response must be a JSON object")

        answer = _lookup(raw, self.answer_field, "")
        contexts = _as_contexts(_lookup(raw, self.contexts_field, []))
        metadata = _as_metadata(_lookup(raw, self.metadata_field, []))
        timings = _lookup(raw, self.timings_field, {})
        if not isinstance(timings, dict):
            timings = {}

        total_seconds = float(timings.get("total_seconds") or total)
        token_count = int(raw.get("token_count") or timings.get("token_count") or 0)
        input_tokens = int(raw.get("input_tokens") or timings.get("input_tokens") or 0)
        output_tokens = int(raw.get("output_tokens") or timings.get("output_tokens") or token_count or 0)
        total_tokens = int(raw.get("total_tokens") or timings.get("total_tokens") or input_tokens + output_tokens)
        estimated_cost_raw = (
            raw["estimated_cost_usd"]
            if "estimated_cost_usd" in raw
            else timings.get("estimated_cost_usd")
        )
        estimated_cost_usd = float(estimated_cost_raw) if estimated_cost_raw is not None else None
        tokens_per_second = float(
            raw.get("tokens_per_second")
            or timings.get("tokens_per_second")
            or (token_count / total_seconds if total_seconds > 0 and token_count else 0)
        )
        raw_content = str(raw.get("raw_content") or answer or "")
        answer_text = str(answer or "")

        return RagSystemOutput(
            answer=answer_text,
            contexts=contexts,
            metadata=metadata,
            raw_response=raw,
            ttft_seconds=float(timings.get("ttft_seconds") or 0.0),
            total_seconds=total_seconds,
            token_count=token_count,
            tokens_per_second=tokens_per_second,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
            raw_content=raw_content,
            raw_reasoning=raw.get("raw_reasoning"),
            answer_valid=bool(answer_text.strip()),
        )
