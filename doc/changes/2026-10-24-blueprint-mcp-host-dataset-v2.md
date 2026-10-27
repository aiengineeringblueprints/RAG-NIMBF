# Larger MCP host benchmark: fixture server, 59 questions, answer matching

Date: 2026-10-24

## Summary
The first `blueprint_mcp_host` dataset (3 documents, 8 questions, 2 demo
tools) could not separate good from bad tool use. It is replaced by a
generated dataset with 25 documents, 59 questions in five categories and a
framework-owned MCP fixture server with six data tools and a decoy. A
deterministic answer match and a per-category breakdown make MCP off vs. on
comparable without an LLM judge. Follows
[2026-10-24-mcp-host-benchmarking](2026-10-24-mcp-host-benchmarking.md).

## Changes

**Answer matching** (`benchmark/answer_match.py`, new): `answer_matches(answer,
patterns)` with pattern groups, Unicode/Markdown/thousands normalization,
word-boundary matching and `re:` regex alternatives.

**Category breakdown** (`benchmark/mcp_host.py`): `aggregate_by_category()`
returns `answer_match_count`, `answer_match_rate` and `per_category`
(answer match plus call and tool-selection rates when traced).

**Adapter** (`examples/enterprise_rag_plugin/mcp_adapter.py`): stores
`category` and `answer_match` (from `metadata.answer_patterns`, fallback the
ground truth) per sample; failed requests count as misses;
`aggregate_metrics()` merges host and answer metrics, so MCP-off runs now
report answer metrics instead of `None`.

**Dataset** (`datasets/blueprint_mcp_host/`): `fixture_data.py`,
`generate.py`, `mcp_server.py`, regenerated `corpus/` and `questions.jsonl`,
rewritten `README.md`. Manifest `experiments/enterprise-rag-mcp-host.yaml`
now samples all 59 questions.

**Docs**: `doc/mcp-host-benchmarking.md` (answer patterns, new metrics,
fixture-server config, reference result).

## Behaviour changes and migration
- `enterprise_rag_mcp` `adapter_metrics` gain `answer_match_*` and
  `per_category`; with MCP off they are no longer `None`.
- The dataset's MCP labels now refer to the fixture server
  (`datasets/blueprint_mcp_host/mcp_server.py`), not the Blueprint demo
  server. Update the Blueprint `MCP_CONFIG_FILE` accordingly.

## Verification
- New tests: `tests/test_answer_match.py` (15),
  `tests/test_blueprint_mcp_host_dataset.py` (7: generator reproducibility,
  balance, label consistency, ground truth vs. patterns, answers present in
  their documents and absent for MCP-only questions, live stdio server with
  tool error), 2 category tests in `tests/test_mcp_host.py`, 1 adapter test.
  Full suite: 683 passed.
- Live run, 118 Blueprint requests (MCP off and on, 59 each). Results are in
  the guide's reference table: answer match 24 % → 80 % (re-scored with the
  final patterns, originally 76 %), MCP on: `call_success_rate` 0.97,
  `tool_selection_f1` 0.97, decoy never called.
- Dataset-design checks: the integrity test caught bare-number patterns
  that also occur in the corpus, so counts and prices now require units or
  decimals. Two correct answers ("96 Kestrel K7 sensors") were scored as
  misses, so `re:` patterns were added and the answers re-scored offline.
  Naming the product in the boilerplate paragraphs was tried and reverted:
  a direct retriever check showed identical top-5 recall with and without it.

## Follow-ups
- Document misses come from the Blueprint's dense-only retrieval failing to
  separate product codes. Hybrid/keyword retrieval in the Blueprint would
  show up in `doc_single`, `doc_multi` and `mixed`.
- No fault-injection server yet (slow, failing, oversized,
  prompt-injecting tools).
