# OCR-01: Parser adapter protocol, registry, HTTP + Python-plugin flavors

**What to build:** A second registered adapter family for document parsers (ADR-005): a parser registry entry point, the ParseResult contract (document in → per-page Markdown; structured blocks optional), and two real flavors — an HTTP adapter for OpenAI-compatible parsing endpoints and an in-process Python-plugin adapter for local tools (Docling/Marker/MinerU). Registered parsers are first-class citizens alongside the existing adapter registry.

**Blocked by:** adapter-seam campaign #3 (Orchestrator drives every system through the adapter seam) and #4 (Worker becomes the only orchestration loop) — Parser adapters enter through the settled seam. Otherwise none.

**Status:** ready-for-agent

- [ ] Parser registry with registration by name; parser resolution symmetrical to the RAG adapter registry
- [ ] ParseResult contract: per-page Markdown; optional structured blocks; Markdown-only parsers fully supported
- [ ] HTTP flavor works against an OpenAI-compatible endpoint (tested with a stub server)
- [ ] Python-plugin flavor loads a local parser class (tested with a stub parser)
- [ ] Parser identity + version capturable for run metadata
- [ ] Tests at the Parser adapter interface only; house style per existing adapter protocol tests
