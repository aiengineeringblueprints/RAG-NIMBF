# Parser adapters are a separate adapter family from RAG adapters

We are adding OCR/document-parsing benchmarking. Decided: parsers get their own
registered adapter family (`register_parser_adapter`, contract: document in →
`ParseResult` of per-page Markdown out, structured blocks optional) beside the
RAG adapters, with its own metric module group (CER/WER, TEDS, vendored
OmniDocBench matching). We deliberately did NOT stretch the existing RAG
adapter lifecycle (prepare → retrieve → generate) to cover parsing — the two
contracts share serving patterns (HTTP, Python plugin) but not lifecycles, and
merging the terms would muddy both. Parsing attaches to RAG experiments only
through the `corpus_parser` config knob (ADR-neutral knob, corpus = parsed
Markdown), which preserves same-run provenance for a later parsing-metrics ↔
RAG-outcomes correlation study (an open gap in the literature, see
`doc/11-ocr-benchmarking-sota.md` §5).

Considered and rejected: overloading "Adapter" for both families; making
structured blocks (bbox/category/table-HTML) mandatory in the ParseResult
contract (would exclude Markdown-only parsers; OmniDocBench itself converts
Markdown tables to HTML for TEDS).

Consequences: CONTEXT.md distinguishes RAG adapter vs Parser adapter; the
`Adapter` glossary entry stays scoped to RAG systems; parsing evaluation is a
self-contained module group that can later merge into a unified evaluation
module (ADR-004's deferred batch) without rework.
