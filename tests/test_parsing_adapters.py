"""Tests for the parser adapter seam: registry, contract, HTTP + plugin."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

import pytest

from benchmark.parsing import (
    PARSER_ADAPTER_REGISTRY,
    ParseResult,
    get_parser_adapter,
    register_parser_adapter,
)
from benchmark.parsing.base import ParsedPage
from benchmark.parsing.http import HttpParserAdapter


@dataclass
class DummyParserConfig:
    parser_adapter: str = ""
    parser_version: str | None = None
    parser_http_endpoint_url: str | None = None
    parser_http_model: str = "olmocr"
    parser_http_prompt: str = "Parse this page to markdown."
    parser_http_timeout_seconds: float = 60.0
    parser_http_headers: str | None = None
    parser_plugin_module: str | None = None
    parser_plugin_attribute: str | None = None


class StubParser:
    name = "stub"
    parser_version: str | None = "0.1"

    def parse(self, document, config=None):
        return ParseResult(
            document_id=document["document_id"],
            pages=(
                ParsedPage(page_number=1, markdown="# Hello"),
                ParsedPage(page_number=2, markdown="World"),
            ),
            parser_name=self.name,
            parser_version=self.parser_version,
        )


def test_parser_registry_registers_and_resolves_by_name():
    original = PARSER_ADAPTER_REGISTRY.copy()
    try:
        seen = {}

        def factory(config):
            seen["config"] = config
            return StubParser()

        register_parser_adapter("stub", factory)

        cfg = DummyParserConfig(parser_adapter="stub")
        parser = get_parser_adapter(cfg)

        assert isinstance(parser, StubParser)
        assert seen["config"] is cfg
    finally:
        PARSER_ADAPTER_REGISTRY.clear()
        PARSER_ADAPTER_REGISTRY.update(original)


def test_register_parser_adapter_normalizes_names():
    original = PARSER_ADAPTER_REGISTRY.copy()
    try:
        register_parser_adapter("  STUB  ", lambda config: StubParser())
        parser = get_parser_adapter(DummyParserConfig(parser_adapter="stub"))
        assert isinstance(parser, StubParser)
    finally:
        PARSER_ADAPTER_REGISTRY.clear()
        PARSER_ADAPTER_REGISTRY.update(original)


def test_get_parser_adapter_returns_none_when_unconfigured():
    assert get_parser_adapter(DummyParserConfig()) is None


def test_get_parser_adapter_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported PARSER_ADAPTER"):
        get_parser_adapter(DummyParserConfig(parser_adapter="nope"))


def test_parse_result_markdown_only_contract():
    result = ParseResult(
        document_id="doc-1",
        pages=(ParsedPage(page_number=1, markdown="# Page one"),),
        parser_name="stub",
    )
    assert result.markdown == "# Page one"
    assert result.pages[0].blocks is None


def test_parse_result_joins_pages_in_order():
    result = ParseResult(
        document_id="doc-1",
        pages=(
            ParsedPage(page_number=1, markdown="one"),
            ParsedPage(page_number=2, markdown="two"),
        ),
        parser_name="stub",
    )
    assert result.markdown == "one\n\ntwo"


class FakePageResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_http_parser_adapter_uses_openai_compatible_endpoint(monkeypatch):
    captured: dict = {"bodies": []}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["bodies"].append(json.loads(req.data.decode("utf-8")))
        return FakePageResponse({
            "choices": [
                {"message": {"content": "# Parsed\n\ntext"}}
            ]
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    adapter = HttpParserAdapter.from_config(
        DummyParserConfig(parser_http_endpoint_url="http://parser.local/v1/chat/completions")
    )
    document = {
        "document_id": "doc-1",
        "pages": [
            {"page_number": 1, "image_base64": "QUJD", "image_media_type": "image/png"},
            {"page_number": 2, "text": "plain page"},
        ],
    }
    result = adapter.parse(document)

    assert captured["url"] == "http://parser.local/v1/chat/completions"
    assert captured["timeout"] == 60.0
    assert len(captured["bodies"]) == 2

    first_body = captured["bodies"][0]
    assert first_body["model"] == "olmocr"
    first_content = first_body["messages"][0]["content"]
    assert first_content[0] == {"type": "text", "text": "Parse this page to markdown."}
    assert first_content[1]["type"] == "image_url"
    assert first_content[1]["image_url"]["url"].startswith("data:image/png;base64,QUJD")
    # Second call uses the plain page text.
    second_content = captured["bodies"][1]["messages"][0]["content"]
    assert second_content[1] == {"type": "text", "text": "plain page"}

    assert result.document_id == "doc-1"
    assert result.parser_name == "http"
    assert [page.markdown for page in result.pages] == ["# Parsed\n\ntext", "# Parsed\n\ntext"]
    assert result.markdown == "# Parsed\n\ntext\n\n# Parsed\n\ntext"


def test_http_parser_adapter_requires_endpoint():
    with pytest.raises(ValueError, match="PARSER_HTTP_ENDPOINT_URL"):
        HttpParserAdapter.from_config(DummyParserConfig())


def test_http_parser_adapter_captures_identity_and_version():
    adapter = HttpParserAdapter.from_config(
        DummyParserConfig(
            parser_http_endpoint_url="http://parser.local/v1/chat/completions",
            parser_version="olmocr-1.0",
        )
    )
    assert adapter.parser_name == "http"
    assert adapter.parser_version == "olmocr-1.0"


def test_plugin_parser_flavor_loads_local_class(tmp_path, monkeypatch):
    plugin = tmp_path / "my_parser_plugin.py"
    plugin.write_text(
        "class LocalParser:\n"
        "    name = 'local'\n"
        "    parser_version = '9.9'\n"
        "\n"
        "    def __init__(self, config=None):\n"
        "        self.config = config\n"
        "\n"
        "    def parse(self, document, config=None):\n"
        "        from benchmark.parsing.base import ParseResult, ParsedPage\n"
        "        return ParseResult(\n"
        "            document_id=document['document_id'],\n"
        "            pages=(ParsedPage(page_number=1, markdown='local!'),),\n"
        "            parser_name=self.name,\n"
        "            parser_version=self.parser_version,\n"
        "        )\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    cfg = DummyParserConfig(
        parser_plugin_module="my_parser_plugin",
        parser_plugin_attribute="LocalParser",
    )
    parser = get_parser_adapter(cfg)

    assert parser is not None
    assert parser.name == "local"
    result = parser.parse({"document_id": "doc-9"})
    assert result.markdown == "local!"
    assert result.parser_version == "9.9"
    assert "my_parser_plugin" in sys.modules


def test_plugin_parser_flavor_rejects_missing_attribute(tmp_path, monkeypatch):
    plugin = tmp_path / "other_parser_plugin.py"
    plugin.write_text("X = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    cfg = DummyParserConfig(
        parser_plugin_module="other_parser_plugin",
        parser_plugin_attribute="DoesNotExist",
    )
    with pytest.raises(ValueError, match="other_parser_plugin"):
        get_parser_adapter(cfg)
