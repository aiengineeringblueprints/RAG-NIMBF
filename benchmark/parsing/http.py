"""HTTP adapter for OpenAI-compatible document parsing endpoints."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from benchmark.parsing.base import ParseResult, ParsedPage

DEFAULT_PAGE_PROMPT = (
    "Convert this document page to Markdown. "
    "Preserve headings, lists, and tables. Output Markdown only."
)


@dataclass(frozen=True)
class HttpParserAdapter:
    """Parse documents page by page via an OpenAI-compatible chat endpoint.

    Each page is sent as one ``chat/completions`` request. Pages with image
    data are sent as ``image_url`` content parts (data URI); text-only pages
    are sent as plain text. The assistant message content is taken verbatim
    as the page Markdown.
    """

    endpoint_url: str
    model: str = ""
    page_prompt: str = DEFAULT_PAGE_PROMPT
    timeout_seconds: float = 60.0
    headers: dict[str, str] = field(
        default_factory=lambda: {"Content-Type": "application/json"}
    )
    parser_name: str = "http"
    parser_version: str | None = None

    name: str = "http"

    @classmethod
    def from_config(cls, config: Any) -> "HttpParserAdapter":
        endpoint_url = getattr(config, "parser_http_endpoint_url", None)
        if not endpoint_url:
            raise ValueError(
                "PARSER_HTTP_ENDPOINT_URL is required when the parser "
                "adapter is http"
            )
        headers = {"Content-Type": "application/json"}
        raw_headers = getattr(config, "parser_http_headers", None)
        if raw_headers:
            try:
                parsed = json.loads(raw_headers)
            except json.JSONDecodeError as exc:
                raise ValueError("PARSER_HTTP_HEADERS must be valid JSON") from exc
            if not isinstance(parsed, dict):
                raise ValueError("PARSER_HTTP_HEADERS must be a JSON object")
            headers.update({str(k): str(v) for k, v in parsed.items()})
        return cls(
            endpoint_url=endpoint_url,
            model=str(getattr(config, "parser_http_model", "") or ""),
            page_prompt=str(
                getattr(config, "parser_http_prompt", "") or DEFAULT_PAGE_PROMPT
            ),
            timeout_seconds=float(
                getattr(config, "parser_http_timeout_seconds", 60.0)
            ),
            headers=headers,
            parser_version=getattr(config, "parser_version", None),
        )

    def _content_parts(self, page: dict) -> list[dict[str, Any]]:
        image = page.get("image_base64")
        if image:
            media_type = str(page.get("image_media_type") or "image/png")
            data_uri = f"data:{media_type};base64,{image}"
            return [
                {"type": "text", "text": self.page_prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]
        return [{"type": "text", "text": self.page_prompt}, {"type": "text", "text": str(page.get("text") or "")}]

    def _parse_page(self, page: dict) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": self._content_parts(page)}
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint_url,
            data=body,
            headers=self.headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                req, timeout=self.timeout_seconds
            ) as resp:
                response_body = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"HTTP parser adapter request failed: {exc}") from exc
        try:
            raw = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "HTTP parser adapter response was not valid JSON"
            ) from exc
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if not choices:
            raise RuntimeError(
                "HTTP parser adapter response has no choices"
            )
        message = choices[0].get("message") or {}
        markdown = str(message.get("content") or "")
        return markdown, raw

    def parse(self, document: dict, config: Any = None) -> ParseResult:
        started = time.perf_counter()
        pages: list[ParsedPage] = []
        last_raw: dict[str, Any] | None = None
        for page in document.get("pages", []):
            markdown, raw = self._parse_page(page)
            pages.append(
                ParsedPage(page_number=int(page.get("page_number", len(pages) + 1)), markdown=markdown)
            )
            last_raw = raw
        return ParseResult(
            document_id=str(document.get("document_id", "")),
            pages=tuple(pages),
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            total_seconds=time.perf_counter() - started,
            raw_response=last_raw,
        )
