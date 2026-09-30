# OCR-07: corpus_parser ingestion knob for RAG experiments

**What to build:** The `corpus_parser` config knob (manifest field + env fallback + test, per AGENTS.md convention): when set, corpus construction for a RAG experiment routes through the named Parser adapter — documents are parsed to Markdown, then the existing chunk → index → retrieve → generate → evaluate pipeline runs unchanged. Parser identity and version are pinned into run metadata, completing the provenance chain from parsing to answer quality.

**Blocked by:** OCR-01 (Parser adapter protocol).

**Status:** ready-for-agent

- [ ] Manifest knob + env fallback + matrix expansion + test, per the every-knob convention
- [ ] Corpus built from parsed documents is indistinguishable downstream from a directly loaded text corpus (same chunking/index path)
- [ ] Provenance (parser name, version, dataset license) in run metadata
- [ ] RAG-level test: stub parser + tiny document set → full pipeline runs and evaluates unchanged
