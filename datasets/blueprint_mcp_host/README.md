# Blueprint MCP host fixture

A synthetic benchmark for RAG systems that call MCP servers themselves
(adapter `enterprise_rag_mcp`, guide: `doc/mcp-host-benchmarking.md`). It
pairs a 240-document corpus of a fictional instrument maker with a read-only
MCP server whose data the answering model cannot know, and labels each of
1000 questions with the tools and documents a correct answer needs. All
content is fictional (CC0-1.0).

## Files

- `fixture_data.py`: every record, generated from a fixed seed: 160
  employees, 40 products in 10 families, 8 sites, 20 policies, 40 customers,
  40 projects, 40 incidents, 20 suppliers, 15 courses, and the MCP-only
  tables (300 orders, stock, tickets, project status, invoices, exchange
  rates, shipping). Tool functions are pure.
- `text_bank.py`: prose building blocks per document kind (paragraphs, FAQ
  entries, troubleshooting rows, agenda items, ...). Contains no digits, so
  filler can neither contradict a fact nor leak an MCP-only value.
- `generate.py`: renders `corpus/` and `questions.jsonl`. MCP ground truths
  are computed with the same functions the server exposes.
- `mcp_server.py`: stdio MCP server (FastMCP) exposing the tool functions.
- `corpus/`: 240 Markdown documents (see below).
- `questions.jsonl`: 1000 questions in seeded random order. Local loaders take
  the first `sample_size` rows, so any prefix is a mixed sample.

Regenerate after editing `fixture_data.py` or `generate.py` (about 20 s):

```bash
python datasets/blueprint_mcp_host/generate.py
```

`tests/test_blueprint_mcp_host_dataset.py` fails when committed files and the
generator drift apart, and checks every label against corpus and server.

## Corpus

| Documents | IDs | Kind | Count |
| --- | --- | --- | ---: |
| Product datasheets | KI-001..040 | `datasheet` | 40 |
| Firmware release notes per product family | KI-051..060 | `release_notes` | 10 |
| Policies | KI-101..120 | `policy` | 20 |
| Superseded policies (distractors) | KI-901..903 | `policy` | 3 |
| Sites | KI-201..208 | `site` | 8 |
| Customer profiles | KI-301..340 | `customer` | 40 |
| Project briefs | KI-401..440 | `project` | 40 |
| Incident reports | KI-501..540 | `incident` | 40 |
| Supplier profiles | KI-601..620 | `supplier` | 20 |
| Training courses | KI-701..715 | `course` | 15 |
| Discontinued predecessor products (distractors) | KI-911..914 | `datasheet` | 4 |

Documents are built like real ones, deterministically per document ID:

- **Realistic sections per kind**: e.g. datasheets with overview,
  specifications, ordering, installation, maintenance, troubleshooting, FAQ
  and revision history; incident reports with summary table, timeline, impact,
  root cause and corrective actions; project briefs with facts, budget,
  milestones, risks and minutes.
- **Mixed formats**: Markdown tables, bullet and numbered lists, prose, FAQ
  blocks, dated meeting notes. Key facts are rendered as a table, a list or
  prose, chosen per document; section names come from synonyms; optional
  sections appear in a per-document order (238 distinct section layouts
  across 240 documents).
- **Varied length**: each document is brief, normal or detailed; 10 products
  are documented as long installation and operating manuals. Length ranges
  from about 100 to 2,050 words (median about 470).
- **Competing values in the same document**: revision histories with an old
  list price and weight, firmware tables with older versions, contract
  histories with the previous support tier and SLA, initial budget and
  downtime estimates, temporary lead times during a shortage. Questions about
  these facts ask for the *current*, *latest*, *final* or *approved* value.
- **Cross-references**: projects name customers, leads and products;
  incidents name customers and products; customer profiles list their
  projects and incidents; every document ends with document-control text that
  names its own subject, so no paragraph repeats verbatim across documents.

Live numbers (budget spent, stock, invoices, extensions) are only available
through the MCP server.

## MCP tools

| Tool | Data |
| --- | --- |
| `search_directory(query)` | employees: title, department, office, extension |
| `get_order_status(order_id)` | orders: customer ID, product, SKU, quantity, status, carrier, date, weight, value |
| `get_stock_level(sku)` | stock per SKU (SKU only, so product names need the datasheet) |
| `convert_currency(amount, from_currency, to_currency)` | fixed company rates for 9 currencies |
| `count_open_tickets(product_code)` | open tickets and most common issue per product |
| `get_shipping_quote(weight_kg, destination_country)` | base fee plus per-kg rate, 12 countries |
| `get_project_status(project_code)` | budget spent, percent complete, hours, risk |
| `get_invoice_balance(customer_id)` | outstanding EUR and overdue invoices (ID only, so names need the profile) |
| `get_weather(city)`, `get_stock_price(ticker)` | decoys: no question needs them |

Unknown inputs return MCP tool errors, so recovery shows up in the call
metrics.

## Questions

| Category | Count | Evidence | Expected tools | Sub-types |
| --- | ---: | --- | --- | ---: |
| `doc_single` | 300 | one document | none | 9 (one per document kind) |
| `doc_multi` | 100 | two documents plus arithmetic | none (no calculator exposed) | 8 |
| `mcp_single` | 250 | one tool | one | 13 |
| `mcp_chain` | 100 | one tool's output feeds a second tool | two | 7 |
| `mixed` | 250 | a document fact is the input of a tool call | one | 14 |

Every question carries `case_id`, `category`, `subtype`, `expected_tools`,
`requires_mcp`, `gold_doc_ids` and `answer_patterns` (groups of
alternatives; every group must appear in the answer; `re:` marks a regular
expression). Counts and prices require a unit or decimals.

Generation filters candidates: a document answer must occur in its gold
document, and a tool answer must not occur anywhere in the corpus (so the
tool is really needed). Sub-types are sampled evenly; a few (restock dates,
policy roles, site heads) are capped by their entity count.

Expose all ten tools and no resources in the RAG system's MCP configuration,
otherwise the tool-selection labels do not apply.

## Retrieval difficulty (reference)

Dense top-5 retrieval (`nomic-embed-text`, chunks 512/64, about 2,000
chunks, raw question, no query rewriting), 2026-10-24. "Gold document" counts
questions whose gold documents all appear in the top 5; "answer chunk" counts
`doc_single` questions where a retrieved chunk of the gold document contains
the answer.

| Questions | Gold document in top 5 | Answer chunk in top 5 |
| --- | ---: | ---: |
| `doc_single` policy | 91 % | 91 % |
| `doc_single` project | 91 % | 74 % |
| `doc_single` site | 83 % | 83 % |
| `doc_single` incident | 51 % | 37 % |
| `doc_single` course | 34 % | 31 % |
| `doc_single` supplier | 18 % | 18 % |
| `doc_single` product | 6 % | 6 % |
| `doc_single` customer | 3 % | 3 % |
| `doc_single` release notes | 6 % | 0 % |
| `doc_single` total | | 37 % |
| `doc_multi` (both documents) | 6 % | |
| `mixed` (one document) | 30 % | |

Documents identified by proper names or product codes are hard for
dense-only retrieval, and datasheets that mention firmware outrank the
release notes that hold the version tables. Hybrid or keyword retrieval
should show up directly in these categories.
