# MCP host benchmark scaled to 1000 questions and 240 documents

Date: 2026-10-24

## Summary
The `blueprint_mcp_host` dataset grows from 25 documents / 59 questions to
240 documents of ten kinds and 1000 questions in 51 sub-types, so category
and sub-type results are statistically meaningful and the corpus is diverse
enough to stress retrieval. The MCP fixture server gains two data tools and a
second decoy. Follows
[2026-10-24-blueprint-mcp-host-dataset-v2](2026-10-24-blueprint-mcp-host-dataset-v2.md).

## Changes

**Fixture data** (`datasets/blueprint_mcp_host/fixture_data.py`, rewritten):
seeded generation of 160 employees, 40 products in 10 families, 8 sites,
20 policies (hand-written facts with questions and patterns), 40 customers,
40 projects, 40 incidents, 20 suppliers, 15 courses, superseded policies and
discontinued products as distractors; MCP tables for 300 orders, stock,
tickets, project status, invoices, 9 currencies and 12 shipping countries.
New tools `get_project_status`, `get_invoice_balance`; decoys
`get_weather`, `get_stock_price` (`DECOY_TOOLS`).

**MCP server** (`mcp_server.py`): exposes the new tools.

**Generator** (`generate.py`, rewritten): per-kind document templates with
two phrasings per sentence and cross-references between documents;
candidate builders per sub-type; filters (document answers must occur in the
gold document, tool answers must not occur in the corpus); even sampling over
sub-types to `CATEGORY_TARGETS`; seeded shuffle of the questions file.
Questions carry a new `metadata.subtype`.

**Answer matching** (`benchmark/answer_match.py`): new public
`normalize_answer()` and `answer_matches_normalized()` so large texts are
normalized once.

**Manifest** (`experiments/enterprise-rag-mcp-host.yaml`): `sample_size: 200`
by default (first 200 rows of the shuffled file); 1000 for the full run.

**Docs**: dataset `README.md` (corpus table, tools, categories, retrieval
difficulty), `doc/mcp-host-benchmarking.md` (dataset, server config, previous
reference result marked as such), `doc/00-README.md`. Deferred work is now
tracked in `doc/experiment-todos.md` §7 (full live run, confidence
intervals, unanswerable/error-recovery/decoy-sweep/fault-injection/multi-turn
categories, ToolQA and CRAG adapters, Blueprint trace and retrieval gaps).

## Behaviour changes and migration
- The Blueprint `MCP_CONFIG_FILE` must expose all ten fixture tools
  (`read_only_tools`), see the guide.
- The previous 59-question reference result is not comparable with the new
  dataset.
- The dataset test now regenerates 1000 questions; the full test suite takes
  about 45 s instead of about 22 s.

## Verification
- `tests/test_blueprint_mcp_host_dataset.py` (9 tests): generator
  reproducibility, 1000 unique questions with exact category targets, ≥ 40
  sub-types, ≥ 200 documents of ≥ 9 kinds, every data tool expected ≥ 30
  times and decoys never, label consistency, ground truth vs. patterns,
  document answers present in their documents, tool answers absent from the
  corpus, live stdio server with tool error. `tests/test_answer_match.py`
  gains a pre-normalized matching test. Full suite: 686 passed.
- Generation checks caught hand-written pattern bugs (plural mismatch,
  prefix patterns blocked by word boundaries); they were fixed and the
  filter now drops 0 of 1469 document candidates.
- Dense-retrieval difficulty check (Ollama `nomic-embed-text`, top 5, chunks
  512/64, no LLM): gold documents found for 97 % of policy and 94 % of
  project questions but only 9 % of customer and product questions and 7 %
  of `doc_multi` pairs; numbers in the dataset README.
- No live Blueprint run of the new version yet (cost: about 9–10k LLM calls
  for the full set with MCP off and on).

## Follow-ups
See `doc/experiment-todos.md` §7.
