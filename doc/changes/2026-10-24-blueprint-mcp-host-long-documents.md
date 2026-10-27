# Longer, varied documents for the MCP host benchmark

Date: 2026-10-24

## Summary
The `blueprint_mcp_host` documents were short (about 1,000 characters) and all
shared one "Facts + identical filler" template, so chunking did not matter and
reading was trivial once a document was retrieved. They are now built like
real enterprise documents: realistic sections per kind, mixed formats, very
different lengths including long manuals, and older competing values inside
the same document. Questions stay the same 1000 in 51 sub-types. Follows
[2026-10-24-blueprint-mcp-host-dataset-v3](2026-10-24-blueprint-mcp-host-dataset-v3.md).

## Changes

**Text bank** (`datasets/blueprint_mcp_host/text_bank.py`, new): paragraphs,
FAQ entries, troubleshooting rows, parameters, agenda items etc. per document
kind, filled with each document's own names; no digits.

**Document builder** (`generate.py`): a `Doc` Markdown builder with a
per-document random generator. Per kind: realistic sections, key facts as
table/list/prose, heading synonyms, optional sections in random order,
verbosity levels (brief/normal/detailed), 10 products as installation and
operating manuals, revision histories and estimates with older values,
cross-references ("Related documents"), and a closing document-control
section naming the document's subject. `documents()` now returns
`{doc_id: (kind, markdown)}`; the old `render(title, kind, facts)` is gone.
Filler dates avoid MCP-only restock dates.

**Questions**: wording asks for the *current* list price, weight, support tier,
SLA and lead time, the *final* deadline and downtime, wherever the document
now holds an older value; affected `doc_multi` and `mixed` questions too.

**Answer matching** (`benchmark/answer_match.py`): substring pre-check
before the regex (same results, generation about 2× faster on the larger
corpus).

**Docs**: dataset `README.md` (document structure, retrieval difficulty with
document- and answer-chunk-level numbers), `doc/mcp-host-benchmarking.md`.

## Behaviour changes and migration
- Corpus content changed completely (same document IDs and facts); indexes
  built from the previous corpus must be rebuilt.
- Some question texts changed (qualifiers); results are not comparable with
  runs on the previous version.

## Verification
- New tests in `tests/test_blueprint_mcp_host_dataset.py`, written first and
  failing on the old corpus: length spread (min ≥ 100, median ≥ 450, max ≥
  2000, ≥ 10 long and ≥ 10 short documents), formats (≥ 150 documents with
  tables, ≥ 150 with lists, ≥ 40 with FAQ, ≥ 80 distinct section layouts), no
  paragraph over 120 characters shared by more than 12 documents, competing
  older prices and firmware versions present next to the current ones, and
  qualifiers in questions about those facts. All existing integrity checks
  (answers in gold documents, tool answers absent from the corpus, labels,
  server) still pass. Full suite: 691 passed.
- Result: 240 documents, 107–2,051 words (median 472), 238 distinct section
  layouts, 2,014 chunks at
  512/64 (previously about 600).
- Retrieval check, dense top 5 (`nomic-embed-text`, no LLM): answer chunk
  found for 37 % of `doc_single` questions (46 % on the short documents).
  Release notes fall from 53 % to 0 % because firmware-mentioning datasheets
  outrank them; sites (62 → 83 %) and incidents (31 → 37 %) got easier. Full
  table in the dataset README.

## Follow-ups
- Re-run the Blueprint live on a sample to see how reading accuracy changes
  now that documents contain competing values (`doc/experiment-todos.md` §7).
- Chunk-size sweep (256/512/1024) on this corpus with the internal pipeline,
  now that chunking matters.
