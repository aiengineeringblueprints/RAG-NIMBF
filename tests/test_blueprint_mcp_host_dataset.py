"""Integrity of the blueprint_mcp_host dataset and its MCP fixture server."""

import asyncio
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

from benchmark.answer_match import answer_matches, answer_matches_normalized, normalize_answer

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "blueprint_mcp_host"


def _load(name):
    # Loaded by path: the top-level ``datasets`` name belongs to HuggingFace.
    sys.path.insert(0, str(ROOT))
    try:
        spec = importlib.util.spec_from_file_location(f"bmh_{name}", ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ROOT))


generate = _load("generate")
data = _load("fixture_data")
ROWS = [json.loads(line) for line in (ROOT / "questions.jsonl").read_text().splitlines()]
DECOYS = data.DECOY_TOOLS
DOC_TEXTS = {p.stem: p.read_text(encoding="utf-8") for p in (ROOT / "corpus").glob("*.md")}
CORPUS = normalize_answer("\n".join(DOC_TEXTS.values()))


def test_committed_files_match_the_generator():
    expected = generate.questions()
    assert [r["question"] for r in ROWS] == [q["question"] for q in expected]
    assert [r["metadata"]["answer_patterns"] for r in ROWS] == [
        q["answer_patterns"] for q in expected
    ]
    documents = generate.documents()
    assert {doc_id: text for doc_id, (_kind, text) in documents.items()} == DOC_TEXTS


def test_dataset_size_and_balance():
    categories = Counter(r["metadata"]["category"] for r in ROWS)
    assert categories == Counter(generate.CATEGORY_TARGETS)
    assert len(ROWS) == 1000
    assert len({r["question"] for r in ROWS}) == len(ROWS)
    assert len({r["metadata"]["case_id"] for r in ROWS}) == len(ROWS)
    subtypes = Counter(r["metadata"]["subtype"] for r in ROWS)
    assert len(subtypes) >= 40
    # Few sub-types are capped by their entity count; the rest stay balanced.
    assert sum(count < 10 for count in subtypes.values()) <= 4


def test_corpus_is_large_and_diverse():
    docs = generate.documents()
    kinds = Counter(kind for kind, _text in docs.values())
    assert len(docs) >= 200
    assert len(kinds) >= 9
    assert all(count >= 8 for count in kinds.values())


def _words(text):
    return len(text.split())


def test_document_lengths_vary_and_include_long_manuals():
    lengths = sorted(_words(text) for text in DOC_TEXTS.values())
    assert lengths[0] >= 100
    assert lengths[len(lengths) // 2] >= 450
    assert lengths[-1] >= 2000
    assert sum(length >= 1500 for length in lengths) >= 10
    assert sum(length < 400 for length in lengths) >= 10


def test_documents_mix_formats_and_structures():
    texts = list(DOC_TEXTS.values())
    assert sum(bool(re.search(r"^\|[- |]+\|$", t, re.M)) for t in texts) >= 150  # tables
    assert sum(bool(re.search(r"^- ", t, re.M)) for t in texts) >= 150  # bullet lists
    assert sum(bool(re.search(r"^\*\*Q:", t, re.M)) for t in texts) >= 40  # FAQ entries
    structures = {tuple(re.findall(r"^## (.+)$", t, re.M)) for t in texts}
    assert len(structures) >= 80


def test_boilerplate_is_not_shared_verbatim_across_documents():
    paragraphs = Counter(
        paragraph
        for text in DOC_TEXTS.values()
        for paragraph in {p.strip() for p in text.split("\n\n")}
        if len(paragraph) > 120
    )
    assert max(paragraphs.values()) <= 12


def test_current_values_compete_with_older_values_in_the_same_document():
    for code, product in data.PRODUCTS.items():
        text = DOC_TEXTS[generate.PRODUCT_DOC[code]]
        prices = set(re.findall(r"(\d+) EUR", text))
        assert str(product["price_eur"]) in prices and len(prices) >= 2, code
    for prefix, *_rest in data.FAMILIES:
        text = DOC_TEXTS[generate.FAMILY_DOC[prefix]]
        for product in data.PRODUCTS.values():
            if product["family"] == prefix:
                section = text.split(f"### {product['name']}\n", 1)[1].split("\n### ", 1)[0]
                versions = set(re.findall(r"\b\d+\.\d+\.\d+\b", section))
                assert product["firmware"] in versions and len(versions) >= 2, product["code"]


def test_questions_with_competing_values_ask_for_the_current_one():
    qualifiers = ("current", "latest", "final", "approved")
    competing = {"product": ("price", "weigh"), "customer": ("tier", "SLA"),
                 "incident": ("downtime",), "supplier": ("lead time",)}
    for row in ROWS:
        words = competing.get(row["metadata"]["subtype"], ())
        if any(w in row["question"] for w in words):
            assert any(q in row["question"] for q in qualifiers), row["question"]


def test_every_real_tool_is_expected_and_decoys_never():
    expected = Counter(t for r in ROWS for t in r["metadata"]["expected_tools"])
    assert set(expected) == set(data.TOOLS) - DECOYS
    assert all(count >= 30 for count in expected.values())


def test_labels_are_consistent_per_category():
    for row in ROWS:
        meta = row["metadata"]
        tools, docs = meta["expected_tools"], meta["gold_doc_ids"]
        assert meta["requires_mcp"] == bool(tools)
        if meta["category"].startswith("doc_"):
            assert not tools and docs
        elif meta["category"] == "mcp_single":
            assert len(tools) == 1 and not docs
        elif meta["category"] == "mcp_chain":
            assert len(tools) == 2 and not docs
        else:
            assert tools and docs
        for doc_id in docs:
            assert (ROOT / "corpus" / f"{doc_id}.md").is_file()


def test_ground_truth_satisfies_its_own_patterns():
    for row in ROWS:
        assert answer_matches(row["ground_truth"], row["metadata"]["answer_patterns"]), row


def test_single_document_answers_are_stated_in_their_document():
    for row in ROWS:
        if row["metadata"]["category"] != "doc_single":
            continue
        (doc_id,) = row["metadata"]["gold_doc_ids"]
        text = (ROOT / "corpus" / f"{doc_id}.md").read_text(encoding="utf-8")
        assert answer_matches(text, row["metadata"]["answer_patterns"]), row["question"]


def test_tool_answers_are_not_in_the_corpus():
    for row in ROWS:
        if row["metadata"]["expected_tools"]:
            assert not answer_matches_normalized(CORPUS, row["metadata"]["answer_patterns"]), row["question"]


SKU = next(iter(data.STOCK))


async def _server_session_results():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_server.py")])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name: t.description for t in (await session.list_tools()).tools}
            ok = await session.call_tool("get_stock_level", {"sku": SKU})
            bad = await session.call_tool("get_stock_level", {"sku": "K9"})
            return tools, ok, bad


def test_mcp_server_serves_the_fixture_tools():
    tools, ok, bad = asyncio.run(asyncio.wait_for(_server_session_results(), 30))

    assert set(tools) == set(data.TOOLS)
    assert all(description and description.strip() for description in tools.values())
    assert not ok.isError
    assert json.loads(ok.content[0].text)["units"] == data.STOCK[SKU]["units"]
    assert bad.isError
    assert "Unknown SKU" in bad.content[0].text
