# Spec: OCR / document-parsing benchmarking capabilities

Label: `ready-for-agent`
Design basis: grilling session 2026-09-16 + `doc/11-ocr-benchmarking-sota.md` (primary-source research) + ADR-005.

## Problem Statement

The framework benchmarks RAG systems end to end, but has no way to evaluate the document-ingestion side of the pipeline: how well OCR/document-parsing systems (PDF → Markdown) actually parse documents, and how parsing quality affects downstream RAG quality. There is no parser leaderboard capability, no parsing metric, no parser integration seam, and no way to route corpus construction through a parser with provenance.

## Solution

Two staged capabilities. (1) A parser-quality benchmark: document parsers run through a new Parser adapter family, outputs scored against document ground truth with text metrics (CER/WER, normalized edit distance) and TEDS for tables, after GT↔prediction alignment using vendored OmniDocBench matching with the algorithm version pinned in run metadata. Results flow through the existing worker loop, run dirs, and MLflow. (2) A `corpus_parser` config knob that routes corpus construction of ordinary RAG experiments through a parser, preserving same-run provenance so a parsing-metrics ↔ RAG-outcomes correlation experiment can be run later (an open gap in the literature).

## User Stories

1. As a benchmark author, I want to register a document parser as a Parser adapter, so that parsers are integrated exactly like other external systems rather than ad hoc.
2. As a benchmark author, I want to point an experiment at an OpenAI-compatible parsing endpoint over HTTP, so that hosted parsers (olmOCR servers, vLLM-served VLM parsers, Mistral-OCR-style APIs) are benchmarkable without custom code.
3. As a benchmark author, I want to run a local Python parsing tool (Docling, Marker, MinerU) in-process, so that I can benchmark parsers without deploying a service.
4. As a benchmark author, I want parser outputs scored with CER/WER and normalized edit distance per page and per document, so that text-quality comparison across parsers is table-stakes.
5. As a benchmark author, I want tables in parsed Markdown scored with TEDS (markdown tables converted to HTML), so that table-parsing quality is comparable with published OmniDocBench numbers.
6. As a researcher, I want the GT↔prediction matching algorithm and its version recorded in run metadata, so that my scores are interpretable and comparable within a pinned harness version.
7. As a researcher, I want OmniDocBench ground truth loaded through the dataset layer, so that the richest available annotation set anchors my experiments.
8. As a benchmark author preparing vendor-facing results, I want to run DP-Bench (MIT) and olmOCR-Bench (Apache-2.0), so that license-clean claims are possible.
9. As a researcher, I want OmniDocBench's research-only license flagged in experiment configs, so that research-only data never silently backs commercial claims.
10. As a benchmark author, I want parser runs orchestrated by the existing worker with checkpoints, run dirs, and MLflow tracking, so that parsing benchmarks behave like every other benchmark in this framework.
11. As a framework maintainer, I want parsing metrics and matching in their own module group, so that they can later merge into a unified evaluation module (ADR-004) without rework.
12. As a benchmark author comparing parsers, I want a per-parser comparison report (per-category, per-page scores) in the run dir, so that leaderboard-style tables come out of a run.
13. As a benchmark author, I want a `corpus_parser` knob in a RAG manifest, so that corpus construction goes through a chosen parser while chunking/retrieval/evaluation run unchanged.
14. As a researcher, I want parser identity and version pinned in run metadata for RAG runs with corpus parsing, so that the provenance chain from parsing to answer quality is complete.
15. As a researcher, I want the same corpus reproducible across parsers, so that a later correlation experiment (parsing metrics ↔ retrieval recall ↔ answer quality) needs no new infrastructure.
16. As a framework maintainer, I want the ParseResult contract to accept Markdown-only parsers, so that simple hosted parsers are not excluded by structural requirements.

## Implementation Decisions

- **Parser adapter family**: a separate registry entry point beside the RAG adapters (ADR-005). Contract: document in → ParseResult (per-page Markdown; structured blocks optional). Two flavors: HTTP (OpenAI-compatible endpoints) and in-process Python plugin. MCP flavor deferred.
- **Metric scope v1**: CER/WER + normalized edit distance (jiwer + RapidFuzz), TEDS for tables (vendored Apache-2.0 from OmniDocBench/PubTabNet, including the markdown-table → HTML conversion). Deferred: layout mAP, reading order, CDM, charts, olmOCR-style unit tests.
- **Alignment**: vendor OmniDocBench's quick_match matching semantics (Apache-2.0); pin the matching-algorithm version in run metadata; scores are not comparable across matching versions.
- **Module group**: parsing adapters + alignment + metrics live together in their own module group, deliberately separate from the fragmented evaluation modules (ADR-004 defers that merge; this group becomes a ready-made deep module for it).
- **Ingestion**: the `corpus_parser` config knob (manifest field + env fallback + test, per AGENTS.md convention) routes corpus construction through a parser; parser identity/version pinned in run metadata; chunking → retrieval → evaluation unchanged.
- **Datasets v1**: OmniDocBench v1.6 (primary; research-only license flagged in config), DP-Bench (MIT), olmOCR-Bench (Apache-2.0). PubTabNet/DocLayNet deferred with their metrics.
- **Reporting**: parser runs produce per-parser, per-category, per-page score tables in the run dir and MLflow, via the existing worker/tracking path.
- **Correlation experiment**: explicitly out of scope for v1 implementation, but the `corpus_parser` provenance chain is designed so it needs no rework.

## Testing Decisions

- Tests exercise external behavior at three seams: the Parser adapter interface (fake parser in-process), the parsing-metrics module (pure functions: given prediction + GT → score, including matching), and the worker-driven parser run (fake tracker + temp dirs, tiny GT set).
- Metric tests use small fixture pairs with hand-computed expected scores (including normalization edge cases: case, whitespace, unicode, blank pages).
- Alignment tests cover paragraph splitting/truncation merge cases modeled on OmniDocBench's quick_match behavior.
- Ingestion-knob tests follow the AGENTS.md convention: config field + env fallback + matrix expansion.
- Prior art: existing adapter protocol tests, managed-adapter lifecycle tests, dataset adapter tests, and orchestration tests define the house style.
- Stage gate: worker-level tests with a stub parser over a tiny fixture GT subset producing known scores (fake tracker, temp dirs). No real benchmark runs are executed by agents; the user validates with real experiments themselves.

## Out of Scope

- The correlation experiment itself (parsing metrics ↔ RAG outcomes).
- CDM/formula metrics, layout mAP, reading-order metrics, chart metrics.
- olmOCR-style unit-test checks.
- MCP parser flavor; structured-blocks-required parsers; DocLayNet/PubTabNet/Fox datasets.
- Any change to existing RAG adapter contracts or the worker loop's semantics.

## Further Notes

- Gated on the adapter-seam campaign: first implementation ticket is blocked on campaign tickets #3 and #4 landing, so Parser adapters enter through the settled adapter seam and the worker-only loop.
- Primary research with sources and verification notes: `doc/11-ocr-benchmarking-sota.md`.
