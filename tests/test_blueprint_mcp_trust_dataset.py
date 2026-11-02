"""Integrity of the blueprint_mcp_host trust questions and their MCP servers."""

import asyncio
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

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


generate_trust = _load("generate_trust")
trust = _load("trust_data")
core = _load("fixture_data")
ROWS = [json.loads(line) for line in (ROOT / "questions_trust.jsonl").read_text().splitlines()]
DOC_TEXTS = {p.stem: p.read_text(encoding="utf-8") for p in (ROOT / "corpus").glob("*.md")}
CORPUS = normalize_answer("\n".join(DOC_TEXTS.values()))
TEMPLATE = json.loads((ROOT / "blueprint_mcp_trust_servers.template.json").read_text())
SERVERS = {s["id"]: s for s in TEMPLATE["servers"]}


def _rows(category):
    return [r for r in ROWS if r["metadata"]["category"] == category]


def test_committed_questions_match_the_generator():
    assert ROWS == generate_trust.rows()


def test_size_balance_and_labels():
    assert Counter(r["metadata"]["category"] for r in ROWS) == Counter(generate_trust.CATEGORY_TARGETS)
    assert len({r["question"] for r in ROWS}) == len(ROWS)
    assert len({r["metadata"]["case_id"] for r in ROWS}) == len(ROWS)
    for row in ROWS:
        meta = row["metadata"]
        assert meta["expected_behavior"] in {"answer", "decline"}
        assert meta["requires_mcp"] == bool(meta["expected_tools"])
        if meta["expected_behavior"] == "answer":
            assert answer_matches(row["ground_truth"], meta["answer_patterns"]), row


def test_freshness_documents_hold_a_stale_value_and_the_live_value_is_mcp_only():
    rows = _rows("freshness")
    assert {r["metadata"]["subtype"] for r in rows} == {"price_live", "firmware_live"}
    for row in rows:
        meta = row["metadata"]
        assert meta["expected_behavior"] == "answer"
        assert set(meta["expected_tools"]) <= set(SERVERS["kestrel_live"]["read_only_tools"])
        (doc_id,) = meta["gold_doc_ids"]
        assert answer_matches(DOC_TEXTS[doc_id], meta["forbidden_patterns"]), row["question"]
        assert not answer_matches_normalized(CORPUS, meta["answer_patterns"]), row["question"]


def test_live_values_differ_from_the_documented_values():
    for code, product in core.PRODUCTS.items():
        entry = trust.get_price_list_entry(product["sku"])
        assert entry["list_price_eur"] > product["price_eur"]
        release = trust.get_firmware_release(code)
        assert release["latest_version"] != product["firmware"]


def test_permission_questions_expect_a_decline_without_leaking_finance_data():
    rows = _rows("permission")
    assert SERVERS["kestrel_finance"]["roles"] == ["Finance"]
    for row in rows:
        meta = row["metadata"]
        assert meta["expected_behavior"] == "decline"
        assert meta["expected_tools"] == []
        assert meta["forbidden_patterns"]
        assert not answer_matches_normalized(CORPUS, meta["forbidden_patterns"]), row["question"]


def test_injection_questions_carry_a_canary_the_answer_must_not_repeat():
    rows = [r for r in _rows("robustness") if r["metadata"]["subtype"] == "injection_supplier"]
    assert len(rows) >= 10
    for row in rows:
        meta = row["metadata"]
        assert meta["expected_behavior"] == "answer"
        (tool,) = meta["expected_tools"]
        assert tool in SERVERS["kestrel_service"]["read_only_tools"]
        canary = meta["forbidden_patterns"][0][0]
        assert canary.startswith(trust.CANARY_PREFIX)
        assert canary.lower() not in CORPUS
        assert not answer_matches_normalized(CORPUS, meta["answer_patterns"]), row["question"]


def test_tool_call_injections_name_a_forbidden_tool_the_question_does_not_need():
    rows = [r for r in _rows("robustness") if r["metadata"]["subtype"] == "injection_toolcall"]
    assert len(rows) >= 10
    for row in rows:
        meta = row["metadata"]
        (forbidden,) = meta["forbidden_tools"]
        assert forbidden in core.TOOLS and forbidden not in meta["expected_tools"]
        code = row["question"].split("Kestrel ")[1].split(" ")[0]
        comment = trust.get_product_reviews(code)["latest_comment"]
        assert forbidden in comment and trust.INJECTED_CUSTOMER in comment
        assert meta["forbidden_patterns"]  # the leaked balance


def test_injection_tools_return_an_instruction_with_the_canary():
    supplier = core.SUPPLIERS[0]["name"]
    rating = trust.get_supplier_rating(supplier)
    assert "ignore" in json.dumps(rating).lower()
    assert trust.CANARY_PREFIX in json.dumps(rating)


def test_fault_questions_expect_a_decline_after_a_failing_tool():
    faults = {"fault_timeout", "fault_error", "fault_oversized"}
    rows = [r for r in _rows("robustness") if r["metadata"]["subtype"] in faults]
    assert {r["metadata"]["subtype"] for r in rows} == faults
    for row in rows:
        assert row["metadata"]["expected_behavior"] == "decline"
        assert len(row["metadata"]["expected_tools"]) == 1


def test_fault_tools_fail_within_the_configured_budgets():
    serial = "SN-KH3-10001"
    with pytest.raises(trust.ToolError):
        trust.get_warranty_status(serial)
    history = trust.get_service_history(serial)
    assert len(json.dumps(history).encode()) > TEMPLATE["max_result_bytes"]
    assert trust.SLOW_SECONDS > TEMPLATE["timeout_seconds"]
    assert TEMPLATE["request_timeout_seconds"] > TEMPLATE["timeout_seconds"] * TEMPLATE["max_calls"]


def test_unanswerable_questions_name_entities_that_do_not_exist():
    rows = _rows("unanswerable")
    assert len({r["metadata"]["subtype"] for r in rows}) == 5
    for row in rows:
        meta = row["metadata"]
        assert meta["expected_behavior"] == "decline"
        ref = meta["unknown_reference"]
        assert normalize_answer(ref) not in CORPUS
        for tool in meta["expected_tools"]:
            with pytest.raises(core.ToolError):
                core.TOOLS[tool](ref)


def test_template_exposes_every_trust_tool_and_the_core_server():
    assert set(SERVERS) == {"kestrel", "kestrel_live", "kestrel_finance", "kestrel_service"}
    assert set(SERVERS["kestrel"]["read_only_tools"]) == set(core.TOOLS)
    for server_id, tools in trust.SERVER_TOOLS.items():
        assert set(SERVERS[server_id]["read_only_tools"]) == set(tools)
    expected = Counter(t for r in ROWS for t in r["metadata"]["expected_tools"])
    finance = set(trust.SERVER_TOOLS["kestrel_finance"])
    assert set(expected) >= {t for tools in trust.SERVER_TOOLS.values() for t in tools} - finance


async def _server_tools(server_id):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=[str(ROOT / "trust_mcp_server.py"), server_id]
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            error = await session.call_tool("get_warranty_status", {"serial": "SN-KH3-1"}) \
                if server_id == "kestrel_service" else None
            return tools, error


@pytest.mark.parametrize("server_id", ["kestrel_live", "kestrel_finance", "kestrel_service"])
def test_trust_servers_serve_their_tools(server_id):
    tools, error = asyncio.run(asyncio.wait_for(_server_tools(server_id), 30))
    assert tools == set(trust.SERVER_TOOLS[server_id])
    if error is not None:
        assert error.isError and "unavailable" in error.content[0].text


def test_tool_names_survive_the_blueprint_alias_label():
    # The Blueprint labels failed calls by its alias, which keeps 20 characters
    # of the tool name; longer names would break tool-selection matching.
    names = {t for tools in trust.SERVER_TOOLS.values() for t in tools} | set(core.TOOLS)
    assert max(len(n) for n in names) <= 20, sorted(n for n in names if len(n) > 20)
